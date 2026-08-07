"""Hyperparameter search over the training run.

Trains one net per point of a grid, ranks them, and writes every result to disk
as it goes. Three things about it are deliberate:

**The datasets are loaded and encoded once.** Encoding 5.6M positions costs more
than a 30-epoch training run, so a search that shelled out to `train.py` per
trial would spend most of its time re-encoding the same data. Trials call
`train.run_training()` directly against tensors that stay on the GPU.

**Trials are scored under a fixed weighting, not their own.** `train.py` picks
its best epoch by the validation loss *of the blend it is training on*. That is
right for a single run and wrong for a search: a trial that down-weights the
endgame set would score better simply because it moved the goalposts, and
`--weights` is one of the axes most worth searching. So every trial is ranked by
`--score-weights`, which is fixed across the whole search, while `--grid
blend=...` varies only what the trial trains on.

  The default `--score-weights` is 1 per set, i.e. each dataset's mean loss
  counts equally regardless of its size. That is a choice, not a fact - it says
  the endgame holdout matters as much as the midgame one. If you disagree, set
  it, but set it once for the whole search.

**Results are written after every trial.** `results.jsonl` is appended as each
trial finishes, so a search that is killed halfway keeps everything it did, and
`--resume` picks up from there. (The data generator's habit of holding
everything in memory until the end is exactly the trap being avoided here.)

    # the default grid: blend ratio x learning rate x epochs, 24 trials
    python src/python/nnue/grid_search.py \
        --data src/python/nnue/data/selfplay_1m.npz src/python/nnue/data/db_positions.npz

    # a specific sweep
    python src/python/nnue/grid_search.py --data ... \
        --grid blend=1:0.25,1:0.5,1:1 lr=1e-3,2e-3 --repeats 2

    # price the shapes first with bench_width.py, then search the affordable ones
    python src/python/nnue/grid_search.py --data ... --grid l1=128,256,512 l2=16,32

    # what would run, without running it
    python src/python/nnue/grid_search.py --data ... --grid lr=1e-3,2e-3 --dry-run

**What this cannot tell you.** Validation loss is a proxy. STATUS.md is right
that the only number that decides anything is games, and the search has no
access to that: it cannot see that L1=1024 costs node rate, and it cannot see
that 20 cp of midgame MAE may or may not be worth 24% of the search. Use it to
pick two or three candidates, then settle them with `bot_vs_bot.py`.

**`lam=` is worse than a proxy - it is a rigged one, and this is a trap.**
Every trial is scored at one fixed `--score-lambda` so that trials remain
comparable, but that fixed target is itself a lambda, and the trial trained at
that same lambda is the one optimising it. So a lambda sweep ranked here will
essentially always crown `--score-lambda`'s own value: at the default of 1.0 it
reports that lambda 1.0 fits the search score best, which is true, circular, and
says nothing at all about strength. The whole point of mixing in the game result
is to fit the search score *slightly worse* on purpose. Sweep `lam=` here only
to see the size of that cost - never to choose the value. That choice is
`bot_vs_bot.py`'s alone.
"""

import argparse
import itertools
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_selfplay_data
from model import CheckersNet
from train import TrainConfig, load_dataset, piece_breakdown, run_training, split

NNUE_DIR = os.path.dirname(os.path.abspath(__file__))

# axes that map straight onto a TrainConfig field, and how to parse a value.
# 'blend' is handled separately because one value is a list, not a scalar.
AXES = {
    'lr': float,
    'batch': int,
    'epochs': int,
    'l1': int,
    'l2': int,
    'weight_decay': float,
    'pct_start': float,
    'val_frac': float,
    'seed': int,
    'lam': float,
    'piece_balance': float,
    'piece_cap': float,
    # rigged in exactly the way `lam` is, and for the same reason: the target
    # moves with it, so trials at different scales are not comparable on
    # val_loss. Sweep it here only to see mae_cp; let games decide the value.
    'eval_scale': float,
}

# blend ratio x learning rate x epochs. Chosen because those are the three
# things STATUS.md flags as unknown: the blend ratio was never swept (1:0.5 was
# the first value tried), and the loss curve was still descending at epoch 30,
# which usually means the schedule ended too early rather than that the net had
# converged.
DEFAULT_GRID = {
    'blend': ['1:0.25', '1:0.5', '1:1', '1:2'],
    'lr': [1e-3, 2e-3, 4e-3],
    'epochs': [30, 60],
}


# ---------------------------------------------------------------------------
# grid construction
# ---------------------------------------------------------------------------
def parse_axis(spec, n_sets):
    """'lr=1e-3,2e-3' -> ('lr', [0.001, 0.002])."""
    if '=' not in spec:
        raise ValueError(f"axis {spec!r} is not name=v1,v2,...")
    name, values = spec.split('=', 1)
    name = name.strip().replace('-', '_')
    parts = [v.strip() for v in values.split(',') if v.strip()]
    if not parts:
        raise ValueError(f"axis {name!r} has no values")

    if name == 'blend':
        out = []
        for p in parts:
            w = [float(x) for x in p.split(':')]
            if len(w) != n_sets:
                raise ValueError(
                    f"blend {p!r} has {len(w)} weights for {n_sets} datasets")
            out.append(w)
        return name, out

    if name not in AXES:
        raise ValueError(f"unknown axis {name!r}; known: "
                         f"blend, {', '.join(sorted(AXES))}")
    return name, [AXES[name](p) for p in parts]


def build_trials(axes, mode, n_random, seed):
    """Cartesian product, or a random sample of it."""
    names = list(axes)
    points = [dict(zip(names, combo))
              for combo in itertools.product(*(axes[n] for n in names))]
    if mode == 'random' and n_random < len(points):
        random.Random(seed).shuffle(points)
        points = points[:n_random]
    return points


def describe_point(point):
    bits = []
    for k, v in point.items():
        if k == 'blend':
            bits.append('blend=' + ':'.join(f'{x:g}' for x in v))
        elif isinstance(v, float):
            bits.append(f'{k}={v:g}')
        else:
            bits.append(f'{k}={v}')
    return ' '.join(bits)


def point_key(point):
    """Stable identity for --resume. Order-independent."""
    return json.dumps(point, sort_keys=True, default=list)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def headline(kind, m):
    """The one number per set that a human reads off the table."""
    if kind == 'wld':
        return f"{m['acc']:.2%} wdl"
    return f"{m['mae_cp']:.1f} cp"


def aggregate(rows):
    """Group repeats of the same config (same point minus the seed)."""
    groups = {}
    for r in rows:
        point = dict(r['point'])
        point.pop('seed', None)
        groups.setdefault(point_key(point), []).append(r)
    out = []
    for rs in groups.values():
        scores = [r['score'] for r in rs]
        best = min(rs, key=lambda r: r['score'])
        point = dict(best['point'])
        point.pop('seed', None)
        out.append({
            'point': point,
            'score': sum(scores) / len(scores),
            'spread': max(scores) - min(scores),
            'n': len(rs),
            'best': best,
        })
    out.sort(key=lambda g: g['score'])
    return out


def print_table(rows, set_names, score_req, log=print):
    if not rows:
        log("no completed trials")
        return None

    groups = aggregate(rows)
    width = max(len(describe_point(g['point'])) for g in groups)
    repeated = any(g['n'] > 1 for g in groups)

    log("")
    log(f"ranked by combined val loss at fixed score weights "
        f"{':'.join(f'{w:g}' for w in score_req)}")
    header = f"  {'#':>3}  {'config':<{width}}  {'score':>8}"
    if repeated:
        header += f"  {'spread':>7}  {'n':>2}"
    for name in set_names:
        header += f"  {name.split('.')[0][:16]:>16}"
    header += f"  {'params':>8}  {'secs':>6}"
    log(header)
    log("  " + "-" * (len(header) - 2))

    for i, g in enumerate(groups):
        b = g['best']
        line = f"  {i + 1:>3}  {describe_point(g['point']):<{width}}  {g['score']:>8.4f}"
        if repeated:
            line += f"  {g['spread']:>7.4f}  {g['n']:>2}"
        for name in set_names:
            m = b['metrics'].get(name, {})
            line += f"  {headline(b['kinds'][name], m) if m else '-':>16}"
        line += f"  {b['params']:>8,}  {b['seconds']:>6.0f}"
        log(line)

    return groups[0]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+',
                    default=(gen_selfplay_data.default_midgame_datasets()
                             + [os.path.join(NNUE_DIR, 'data', 'db_positions.npz')]),
                    help="one or more .npz files (wld or eval_cp labelled); "
                         "defaults to every midgame set present plus the "
                         "tablebase set")
    ap.add_argument('--grid', nargs='+', default=None,
                    help="axes as name=v1,v2,... e.g. lr=1e-3,2e-3 blend=1:0.5,1:1")
    ap.add_argument('--out-dir', default=os.path.join(NNUE_DIR, 'models', 'grid'),
                    help="checkpoints and results land here")
    ap.add_argument('--score-weights', nargs='+', type=float, default=None,
                    help="fixed per-set aggregate weights used to RANK trials "
                         "(default 1 each). Not what any trial trains on.")
    ap.add_argument('--score-lambda', type=float, default=1.0,
                    help="fixed lambda used to RANK trials, for the same reason "
                         "as --score-weights: a lam= sweep changes what each "
                         "trial trains toward, and trials scored against their "
                         "own target are not comparable. Default 1.0, i.e. rank "
                         "everything on how well it predicts the search score.")
    ap.add_argument('--repeats', type=int, default=1,
                    help="train each config this many times with different seeds; "
                         "reports the mean and the spread")
    ap.add_argument('--mode', choices=['grid', 'random'], default='grid')
    ap.add_argument('--trials', type=int, default=20,
                    help="how many points to sample when --mode random")
    ap.add_argument('--min-pieces', type=int, default=0,
                    help="drop positions below N pieces, as train.py. Fixed "
                         "across the search rather than an axis: it changes "
                         "which rows exist, and the sets are loaded once")
    ap.add_argument('--limit', type=int, default=0,
                    help="use only the first N positions of each set (smoke tests)")
    ap.add_argument('--data-frac', type=float, default=1.0,
                    help="train on this fraction of every dataset, "
                         "proportionally, so the relative sizes and therefore "
                         "the meaning of a blend weight are preserved. Cuts the "
                         "cost of a search at the price of the schedule axes "
                         "(epochs, lr) no longer transferring to full data")
    ap.add_argument('--seed', type=int, default=0,
                    help="base seed: trial seeds are this plus the repeat index")
    ap.add_argument('--resume', action='store_true',
                    help="skip configs already present in results.jsonl")
    ap.add_argument('--dry-run', action='store_true',
                    help="print the trial list and exit")
    args = ap.parse_args()

    n_sets = len(args.data)

    # ---- grid ----
    try:
        if args.grid:
            specs = args.grid
        else:
            # go through the same parser rather than using DEFAULT_GRID's values
            # directly, so the default grid cannot drift from the --grid syntax
            specs = [f"{k}=" + ",".join(str(v) for v in vals)
                     for k, vals in DEFAULT_GRID.items()
                     # the default blend list is written for two datasets
                     if not (k == 'blend' and n_sets != 2)]
        axes = {}
        for spec in specs:
            name, values = parse_axis(spec, n_sets)
            axes[name] = values
    except ValueError as e:
        print(f"error: {e}")
        return 1

    if args.repeats > 1:
        if 'seed' in axes:
            print("error: --repeats and an explicit seed= axis both set the seed")
            return 1
        axes['seed'] = [args.seed + i for i in range(args.repeats)]

    points = build_trials(axes, args.mode, args.trials, args.seed)

    score_req = args.score_weights or [1.0] * n_sets
    if len(score_req) != n_sets:
        print(f"--score-weights has {len(score_req)} entries for {n_sets} datasets")
        return 1

    print(f"{len(points)} trials over " +
          ", ".join(f"{k}({len(v)})" for k, v in axes.items()))
    if args.dry_run:
        for i, p in enumerate(points):
            print(f"  {i + 1:>3}  {describe_point(p)}")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, 'results.jsonl')

    done = {}
    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[point_key(r['point'])] = r
        print(f"resuming: {len(done)} trials already in {results_path}")

    # ---- load once ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available, training on CPU - this will be slow")
    else:
        print(f"training on {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    sets = []
    for path in args.data:
        ds = load_dataset(path, device, args.limit, args.min_pieces,
                          frac=args.data_frac)
        sets.append(ds)
        print(f"  {ds.name:<28} {len(ds):>9,} positions  labels={ds.kind}")
    print(f"loaded and encoded in {time.time() - t0:.1f}s "
          f"(once, for all {len(points)} trials)")

    set_names = [d.name for d in sets]
    kinds = {d.name: d.kind for d in sets}
    if len(set(set_names)) != len(set_names):
        print("error: two datasets share a basename, so per-set results would "
              "collide - rename one")
        return 1

    # ---- run ----
    rows = list(done.values())
    t_start = time.time()
    n_run = 0
    try:
        for i, point in enumerate(points):
            key = point_key(point)
            if key in done:
                print(f"[{i + 1}/{len(points)}] skip (done)  {describe_point(point)}")
                continue

            cfg = TrainConfig(out=os.path.join(args.out_dir, f'trial_{i + 1:03d}.pt'))
            for k, v in point.items():
                if k == 'blend':
                    cfg.weights = list(v)
                else:
                    setattr(cfg, k, v)

            eta = ''
            if n_run:
                per = (time.time() - t_start) / n_run
                left = sum(1 for p in points[i:] if point_key(p) not in done)
                eta = f"  eta {per * left / 60:.0f}m"
            print(f"\n[{i + 1}/{len(points)}] {describe_point(point)}{eta}")

            res = run_training(sets, cfg, score_req=score_req,
                               score_lam=args.score_lambda, verbose=False)
            if res['stopped']:
                # train.py turns Ctrl-C into a clean early stop, so the trial
                # returned normally - but a trial that trained for three epochs
                # of thirty is not a result, and recording it would poison both
                # the ranking and --resume. Hand the interrupt back.
                raise KeyboardInterrupt
            n_run += 1

            row = {
                'point': point, 'score': res['val_loss'],
                'best_epoch': res['best_epoch'], 'metrics': res['metrics'],
                'kinds': kinds, 'params': res['params'],
                'seconds': res['seconds'], 'ckpt': cfg.out,
            }
            rows.append(row)
            done[key] = row

            # append before anything else can fail, so a kill costs one trial
            with open(results_path, 'a') as f:
                f.write(json.dumps(row) + '\n')

            summary = '  '.join(f"{n.split('.')[0]}: {headline(kinds[n], res['metrics'][n])}"
                                for n in set_names if n in res['metrics'])
            print(f"      score {res['val_loss']:.4f}  best epoch "
                  f"{res['best_epoch']}/{cfg.epochs}  {summary}  "
                  f"({res['seconds']:.0f}s)")

            del res
            if device == 'cuda':
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\ninterrupted - ranking the trials that finished")

    best = print_table(rows, set_names, score_req)
    if best is None:
        return 1

    print(f"\nresults appended to {results_path}")
    print(f"best: {describe_point(best['point'])}")
    print(f"      {best['best']['ckpt']}")

    # the winner's error profile, which is the part a single score hides
    ckpt = torch.load(best['best']['ckpt'], weights_only=False)
    model = CheckersNet(ckpt['l1'], ckpt['l2']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    val_frac = best['point'].get('val_frac', TrainConfig.val_frac)
    val_sets = [split(ds, int(len(ds) * val_frac))[1] for ds in sets]
    piece_breakdown(model, val_sets)

    if any('l1' in r['point'] or 'l2' in r['point'] for r in rows):
        print("\nNOTE: this search varied the network shape, and the score does "
              "not know what a shape costs the engine. Price the winner with "
              "bench_width.py before believing it is an improvement.")
    if any('lam' in r['point'] for r in rows):
        print(f"\nNOTE: this search varied lam, and the ranking above is rigged "
              f"in favour of lam={args.score_lambda:g} - that is the target "
              f"every trial was scored against, so the trial trained on it wins "
              f"by construction. Mixing in the game result is *meant* to fit "
              f"the search score slightly worse. The numbers above size that "
              f"cost; only bot_vs_bot.py can say whether it buys anything.")
    print("\nA lower score is a better fit to the labels, not a stronger engine. "
          "Settle the top candidates with bot_vs_bot.py.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
