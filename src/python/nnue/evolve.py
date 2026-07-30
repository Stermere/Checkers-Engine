"""Evolve training hyperparameters against game results.

`grid_search.py` ranks nets by validation loss. This ranks them by Elo: every
individual is trained, exported, compiled into the engine and played against the
frozen baseline, and only the match decides whether it breeds. That is the whole
point - the repo has already been burned once by a change whose training metrics
were healthy and which lost 81 Elo (see the bucketing post-mortem in STATUS.md),
and validation loss cannot see that class of failure by construction.

    genome -> train -> export -> build -> N games vs baseline -> Elo -> select

Read this before starting a long run, because three of the design choices are
about the objective being noisy rather than about genetic algorithms.

**A match is cheap; a training run is not.** Measured on this machine: build 7 s,
500 games at 0.1 s/move on 14 workers 35 s, and a full-data 30-epoch training run
several minutes. So games are the *cheap* axis, and spending 500 of them per
individual is a false economy: 500 games is +-33 Elo at 95%, which cannot resolve
the 5-20 Elo differences most hyperparameters produce. A GA fed that signal does
not search, it drifts - it will happily crown whichever individual got lucky and
then breed from its noise. `--games` therefore defaults high, and selection is
staged: everyone plays `--games`, and only the top few of each generation play
`--confirm-games` more, with the pooled result used for ranking.

**The scale genes are two genes, not one.** `train_eval_scale` changes what the
net is taught; `export_eval_scale` changes only how its logits become centipawns
for the engine, and the engine is not scale free (EVAL_MAX clamps static evals at
1500, TERMINATE_EARLY_THRESHOLD is 20 absolute centipawns). The second costs a
build and a match with no retraining, so individuals differing only in it reuse
their parent's checkpoint - see `training_key`. This is what makes the export
scale nearly free to search, and it appears to matter: a 400 -> 120 change was
worth well over 100 Elo in a hand test.

**Everything is written down as it happens.** `results.jsonl` gets one line per
individual as it finishes, so a killed run keeps every match it paid for, and
`--resume` rebuilds the population from it. Checkpoints of individuals that were
never the best are deleted unless `--keep-all`, because a 60-individual run at
1.4 MB each is not the problem but a 600-individual one is.

Run:

    # a short exploratory run: 8 per generation, 6 generations, ~50 individuals
    python src/python/nnue/evolve.py --pop 8 --generations 6

    # search only the free axis - no training at all, just export + build + play
    python src/python/nnue/evolve.py --ckpt src/python/nnue/models/net.pt \
        --genes export_eval_scale --pop 6 --generations 4

    # what it would do, without doing it
    python src/python/nnue/evolve.py --dry-run
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_selfplay_data
import match_net
from model import EVAL_SCALE
from train import TrainConfig, load_dataset, run_training

NNUE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# genes
# ---------------------------------------------------------------------------
# kind:
#   'log'   sampled and mutated multiplicatively - anything spanning decades
#   'lin'   sampled and mutated additively
#   'int'   as 'lin', rounded
#   'pick'  an unordered choice; mutation resamples
#
# `sigma` is the mutation step: a factor for 'log', a fraction of the range for
# 'lin'/'int'. Chosen small enough that a child stays near its parent, which is
# the only thing making this a search rather than repeated random sampling.
GENES = {
    'lr':               ('log', 3e-4, 8e-3, 1.5),
    'epochs':           ('int', 8, 60, 0.25),
    'batch':            ('pick', [4096, 8192, 16384, 32768], None, None),
    'weight_decay':     ('log', 1e-8, 1e-4, 3.0),
    'pct_start':        ('lin', 0.03, 0.5, 0.2),
    'lam':              ('lin', 0.4, 1.0, 0.15),
    'piece_balance':    ('lin', 0.0, 0.8, 0.2),
    'piece_cap':        ('lin', 1.5, 8.0, 0.2),
    'l1':               ('pick', [256, 512, 1024], None, None),
    'l2':               ('pick', [8, 16, 32], None, None),
    'train_eval_scale': ('log', 60.0, 800.0, 1.6),
    # the one gene that does not require a training run; see the module docstring
    'export_eval_scale': ('log', 60.0, 800.0, 1.6),
}

#: per-dataset blend weight. Handled apart from GENES because how many there are
#: depends on --data. Log-scaled: the interesting moves are 2x, not +0.1.
BLEND_RANGE = (0.02, 4.0)
BLEND_SIGMA = 1.6

#: genes that change the trained net. Two individuals agreeing on all of these
#: share a checkpoint, which is what makes export_eval_scale cheap to explore.
TRAINING_GENES = set(GENES) - {'export_eval_scale'}


def _log_uniform(lo, hi, rng):
    return float(10 ** rng.uniform(math.log10(lo), math.log10(hi)))


def sample_gene(name, rng):
    kind, lo, hi, _sigma = GENES[name]
    if kind == 'pick':
        return rng.choice(lo)
    if kind == 'log':
        return _log_uniform(lo, hi, rng)
    v = rng.uniform(lo, hi)
    return int(round(v)) if kind == 'int' else float(v)


def mutate_gene(name, value, rng):
    kind, lo, hi, sigma = GENES[name]
    if kind == 'pick':
        return rng.choice(lo)
    if kind == 'log':
        return float(min(hi, max(lo, value * (sigma ** rng.gauss(0, 1)))))
    v = value + rng.gauss(0, 1) * sigma * (hi - lo)
    v = min(hi, max(lo, v))
    return int(round(v)) if kind == 'int' else float(v)


def random_genome(genes, n_sets, rng):
    g = {name: sample_gene(name, rng) for name in genes if name != 'blend'}
    if 'blend' in genes:
        g['blend'] = [_log_uniform(*BLEND_RANGE, rng) for _ in range(n_sets)]
    return g


def mutate(genome, genes, rng, rate):
    """Mutate each gene independently with probability `rate`.

    Per-gene rather than one-gene-per-child: with a dozen axes and an objective
    this noisy, changing one thing at a time means most of the budget is spent
    re-measuring the parent.
    """
    child = copy.deepcopy(genome)
    for name in genes:
        if name == 'blend':
            if rng.random() < rate:
                child['blend'] = [
                    float(min(BLEND_RANGE[1], max(BLEND_RANGE[0],
                          w * (BLEND_SIGMA ** rng.gauss(0, 1)))))
                    for w in child['blend']]
        elif rng.random() < rate:
            child[name] = mutate_gene(name, child[name], rng)
    return child


def crossover(a, b, genes, rng):
    """Uniform crossover. Blend vectors are inherited whole, not per-element:
    the weights are meaningful relative to each other, and splicing two ratios
    componentwise mostly produces a ratio neither parent had and neither
    parent's match said anything about."""
    child = {}
    for name in genes:
        child[name] = copy.deepcopy((a if rng.random() < 0.5 else b)[name])
    return child


def describe(genome):
    bits = []
    for k, v in sorted(genome.items()):
        if k == 'blend':
            bits.append('blend=' + ':'.join(f'{x:.3g}' for x in v))
        elif isinstance(v, float):
            bits.append(f'{k}={v:.4g}')
        else:
            bits.append(f'{k}={v}')
    return ' '.join(bits)


def training_key(genome):
    """Identity of the *trained net*, ignoring export-only genes."""
    return json.dumps({k: v for k, v in sorted(genome.items())
                       if k != 'export_eval_scale'}, default=list)


def genome_key(genome):
    return json.dumps(genome, sort_keys=True, default=list)


# ---------------------------------------------------------------------------
# fitness
# ---------------------------------------------------------------------------
def elo_of(wins, losses, draws):
    n = wins + losses + draws
    if n == 0:
        return 0.0
    s = (wins + 0.5 * draws) / n
    s = min(max(s, 0.5 / n), 1.0 - 0.5 / n)     # keep it finite at 0% / 100%
    return -400.0 * math.log10(1.0 / s - 1.0)


class Evaluator:
    """Trains and plays one genome. Owns the dataset tensors and the net cache."""

    def __init__(self, sets, args):
        self.sets = sets
        self.args = args
        self.ckpt_cache = {}     # training_key -> checkpoint path
        self.train_seconds = 0.0
        self.match_seconds = 0.0

    def checkpoint_for(self, genome, tag):
        """Train this genome's net, or reuse one trained for the same genes."""
        if self.args.ckpt:                    # export-only mode
            return self.args.ckpt, 0.0
        key = training_key(genome)
        hit = self.ckpt_cache.get(key)
        if hit and os.path.exists(hit):
            return hit, 0.0

        cfg = TrainConfig(out=os.path.join(self.args.out_dir, f'{tag}.pt'),
                          seed=self.args.seed)
        for name, value in genome.items():
            if name == 'blend':
                cfg.weights = list(value)
            elif name == 'train_eval_scale':
                cfg.eval_scale = value
            elif name == 'export_eval_scale':
                continue
            else:
                setattr(cfg, name, value)

        t0 = time.time()
        res = run_training(self.sets, cfg, verbose=False)
        # run_training hands back the live model and the validation splits; a
        # run of this length would otherwise accumulate one of each per
        # individual on the GPU until it runs out
        del res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        secs = time.time() - t0
        self.train_seconds += secs
        self.ckpt_cache[key] = cfg.out
        return cfg.out, secs

    def play(self, ckpt, genome, games, quiet=True):
        t0 = time.time()
        res = match_net.evaluate_checkpoint(
            ckpt, module=self.args.module, games=games,
            seconds=self.args.seconds, workers=self.args.workers,
            depth=self.args.depth, baseline=self.args.baseline,
            eval_scale=genome.get('export_eval_scale'),
            work_dir=self.args.out_dir, quiet=quiet)
        self.match_seconds += time.time() - t0
        return res


def evaluate(ev, genome, tag, games, log):
    """One individual, end to end. Returns a record or None if it was refused."""
    ckpt, train_secs = ev.checkpoint_for(genome, tag)

    try:
        res = ev.play(ckpt, genome, games)
    except SystemExit as e:
        # the clamp guard in match_net refuses nets whose evals saturate at
        # EVAL_MAX. That is a real result about the genome, not an error: it
        # gets the worst possible fitness and stays out of the breeding pool.
        log(f"      refused: {str(e).splitlines()[0]}")
        return {'genome': genome, 'tag': tag, 'ckpt': ckpt, 'refused': True,
                'wins': 0, 'losses': games, 'draws': 0, 'games': games,
                'elo': -999.0, 'train_seconds': train_secs}

    return {'genome': genome, 'tag': tag, 'ckpt': ckpt, 'refused': False,
            'wins': res['wins'], 'losses': res['losses'], 'draws': res['draws'],
            'games': res['games'], 'elo': elo_of(res['wins'], res['losses'],
                                                 res['draws']),
            'eval_range': res.get('eval_range'),
            'train_seconds': train_secs}


def confirm(ev, rec, games, log):
    """Play more games and pool them into the record.

    Selection on a short match alone is selection on its noise: the individual
    that looks best out of eight after 500 games is, for realistic effect sizes,
    usually the luckiest rather than the best. Pooling a second, longer match
    into the same record is the cheapest correction available, and it is cheap
    precisely because games cost far less than the training run already spent.
    """
    if rec['refused']:
        return rec
    res = ev.play(rec['ckpt'], rec['genome'], games)
    rec['wins'] += res['wins']
    rec['losses'] += res['losses']
    rec['draws'] += res['draws']
    rec['games'] += res['games']
    rec['elo'] = elo_of(rec['wins'], rec['losses'], rec['draws'])
    rec['confirmed'] = True
    log(f"      confirmed over {rec['games']} games: {rec['elo']:+.0f} Elo")
    return rec


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+',
                    default=(gen_selfplay_data.default_midgame_datasets()
                             + [os.path.join(NNUE_DIR, 'data',
                                             'db_positions.npz')]))
    ap.add_argument('--out-dir', default=os.path.join(NNUE_DIR, 'models', 'evolve'))
    ap.add_argument('--genes', nargs='+', default=None,
                    help="which axes to search (default: all of them plus "
                         "blend). Naming a subset pins the rest at their "
                         "TrainConfig defaults")

    ap.add_argument('--pop', type=int, default=8, help="individuals per generation")
    ap.add_argument('--generations', type=int, default=6)
    ap.add_argument('--elite', type=int, default=2,
                    help="best N carried into the next generation unchanged")
    ap.add_argument('--mutation-rate', type=float, default=0.3,
                    help="probability each gene of a child is mutated")
    ap.add_argument('--random-frac', type=float, default=0.15,
                    help="fraction of each generation drawn fresh at random, so "
                         "the population cannot collapse onto one lineage")

    ap.add_argument('--games', type=int, default=1000,
                    help="games every individual plays. 500 is +-33 Elo and too "
                         "noisy to select on; 1000 is +-23, 4000 is +-12")
    ap.add_argument('--confirm-games', type=int, default=2000,
                    help="extra games for the top --confirm-top of a "
                         "generation, pooled into their record")
    ap.add_argument('--confirm-top', type=int, default=3)
    ap.add_argument('--seconds', type=float, default=0.1)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--baseline', default='search_engine_old')
    ap.add_argument('--module', default='search_engine_evolve')

    ap.add_argument('--ckpt', default=None,
                    help="skip training entirely and search only the export "
                         "side against this fixed checkpoint. With --genes "
                         "export_eval_scale this is a scale sweep that costs a "
                         "build and a match per point")
    ap.add_argument('--min-pieces', type=int, default=0,
                    help="fixed across the run; the sets are loaded once")
    ap.add_argument('--limit', type=int, default=0,
                    help="use only the first N rows of EACH set. Prefer "
                         "--data-frac, which does not distort the mix")
    ap.add_argument('--data-frac', type=float, default=1.0,
                    help="train each individual on this fraction of every "
                         "dataset (0.25 = a quarter, proportional so the "
                         "relative sizes are preserved). Training dominates the "
                         "cost of a run, so this is the dial that decides "
                         "whether a search finishes overnight or over a week. "
                         "See the warning it prints: the schedule genes do not "
                         "transfer back to full data unchanged")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep-all', action='store_true',
                    help="keep every checkpoint, not just the best so far")
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_sets = len(args.data)

    genes = args.genes
    if genes is None:
        genes = sorted(GENES) + (['blend'] if not args.ckpt else [])
    if args.ckpt:
        bad = [g for g in genes if g != 'export_eval_scale']
        if bad:
            print(f"--ckpt fixes the trained net, so these genes cannot do "
                  f"anything: {', '.join(bad)}")
            return 1
    unknown = [g for g in genes if g not in GENES and g != 'blend']
    if unknown:
        print(f"unknown genes: {', '.join(unknown)}\n"
              f"known: blend, {', '.join(sorted(GENES))}")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, 'results.jsonl')

    done = {}
    if args.resume and os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[genome_key(r['genome'])] = r
        print(f"resuming: {len(done)} individuals already in {results_path}")

    total = args.pop * args.generations
    print(f"{args.generations} generations x {args.pop} = {total} individuals")
    print(f"genes: {', '.join(genes)}")
    print(f"fitness: {args.games} games vs {args.baseline} at "
          f"{args.seconds}s/move, top {args.confirm_top} confirmed over "
          f"{args.confirm_games} more")

    if args.dry_run:
        for i in range(args.pop):
            print(f"  gen 1 #{i + 1}  {describe(random_genome(genes, n_sets, rng))}")
        print(f"\n(and {args.generations - 1} more generations bred from "
              f"whatever wins)")
        return 0

    # ---- data, loaded once for the whole run ----
    sets = []
    if not args.ckpt:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("WARNING: CUDA not available - a run this size will not "
                  "finish on CPU")
        else:
            print(f"training on {torch.cuda.get_device_name(0)}")
        t0 = time.time()
        for path in args.data:
            ds = load_dataset(path, device, args.limit, args.min_pieces,
                              frac=args.data_frac)
            sets.append(ds)
            print(f"  {ds.name:<44} {len(ds):>10,}  {ds.kind}")
        print(f"loaded and encoded in {time.time() - t0:.0f}s "
              f"(once, for all {total} individuals)")

        if args.data_frac < 1.0:
            # Worth saying out loud every run, because the failure it causes
            # looks like a successful search. Fewer rows means fewer optimizer
            # steps per epoch, so the epoch count and learning rate that win
            # here are the right ones *for this much data* and are not the right
            # ones for the full set - typically the winner wants fewer epochs
            # and a smaller lr once the data is 4x bigger. The blend, lambda and
            # scale genes are far more portable, because they describe the
            # target rather than the schedule.
            print(f"\n  NOTE: searching on {args.data_frac:.0%} of the data. "
                  f"Treat the schedule genes (epochs, lr, batch, pct_start) as "
                  f"provisional - retrain the winners on the full set with a "
                  f"small sweep around them before believing those values. "
                  f"The target genes (blend, lam, *_eval_scale) transfer far "
                  f"better.")

    ev = Evaluator(sets, args)
    history = list(done.values())
    population = []
    best = max(history, key=lambda r: r['elo']) if history else None
    t_start = time.time()
    n_evaluated = 0

    try:
        for gen in range(args.generations):
            # ---- build this generation ----
            if not population:
                candidates = [random_genome(genes, n_sets, rng)
                              for _ in range(args.pop)]
            else:
                ranked = sorted(population, key=lambda r: -r['elo'])
                candidates = [r['genome'] for r in ranked[:args.elite]]
                n_random = max(0, int(round(args.random_frac * args.pop)))
                while len(candidates) < args.pop - n_random:
                    # tournament of 3, which keeps selection pressure mild -
                    # appropriate when the fitness has a +-20 Elo error bar
                    def pick():
                        return max(rng.sample(ranked, min(3, len(ranked))),
                                   key=lambda r: r['elo'])['genome']
                    child = crossover(pick(), pick(), genes, rng)
                    candidates.append(mutate(child, genes, rng,
                                             args.mutation_rate))
                while len(candidates) < args.pop:
                    candidates.append(random_genome(genes, n_sets, rng))

            print(f"\n{'=' * 70}\ngeneration {gen + 1}/{args.generations}")
            population = []
            for i, genome in enumerate(candidates):
                tag = f'g{gen + 1:02d}_i{i + 1:02d}'
                key = genome_key(genome)
                if key in done:
                    print(f"  [{tag}] skip (done)  elo {done[key]['elo']:+.0f}")
                    population.append(done[key])
                    continue

                eta = ''
                if n_evaluated:
                    per = (time.time() - t_start) / n_evaluated
                    left = total - (gen * args.pop + i)
                    eta = f"  eta {per * left / 3600:.1f}h"
                print(f"  [{tag}]{eta}\n      {describe(genome)}")

                rec = evaluate(ev, genome, tag, args.games, log=print)
                if not rec['refused']:
                    print(f"      {rec['wins']}W {rec['losses']}L "
                          f"{rec['draws']}D  ->  {rec['elo']:+.0f} Elo "
                          f"({rec['train_seconds']:.0f}s training)")
                population.append(rec)
                history.append(rec)
                done[key] = rec
                n_evaluated += 1
                with open(results_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')

            # ---- confirm the top few before breeding from them ----
            top = sorted([r for r in population if not r.get('confirmed')],
                         key=lambda r: -r['elo'])[:args.confirm_top]
            for rec in top:
                print(f"  [{rec['tag']}] confirming {rec['elo']:+.0f} Elo")
                confirm(ev, rec, args.confirm_games, log=print)
                with open(results_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')

            gen_best = max(population, key=lambda r: r['elo'])
            if best is None or gen_best['elo'] > best['elo']:
                best = gen_best
            print(f"\n  generation best {gen_best['elo']:+.0f} Elo "
                  f"({gen_best['tag']}), run best {best['elo']:+.0f} Elo "
                  f"({best['tag']})")

            if not args.keep_all and not args.ckpt:
                # only ever delete inside --out-dir, and never the incumbent or
                # anything the live population still points at
                out_dir = os.path.abspath(args.out_dir)
                keep = {r['ckpt'] for r in population} | {best['ckpt']}
                for rec in history:
                    p = rec.get('ckpt')
                    if (p and p not in keep and os.path.exists(p)
                            and os.path.dirname(os.path.abspath(p)) == out_dir):
                        os.remove(p)

    except KeyboardInterrupt:
        print("\ninterrupted - ranking what finished")

    # ---- report ----
    if not history:
        print("nothing was evaluated")
        return 1

    ranked = sorted(history, key=lambda r: -r['elo'])
    print(f"\n{'=' * 70}\ntop 10 of {len(history)} individuals, by Elo vs "
          f"{args.baseline}")
    for i, r in enumerate(ranked[:10]):
        mark = ' (confirmed)' if r.get('confirmed') else ''
        print(f"  {i + 1:>2}  {r['elo']:+7.0f}  {r['games']:>5} games{mark:<12} "
              f"{describe(r['genome'])}")

    print(f"\ntraining {ev.train_seconds / 3600:.1f}h, "
          f"matches {ev.match_seconds / 3600:.1f}h, "
          f"wall {(time.time() - t_start) / 3600:.1f}h")
    print(f"results: {results_path}")

    winner = ranked[0]
    print(f"\nbest: {winner['ckpt']}  at {winner['elo']:+.0f} Elo over "
          f"{winner['games']} games")
    print(f"  {describe(winner['genome'])}")
    if winner['elo'] > 0:
        print(f"\nDeploy it with:\n  python src/python/nnue/match_net.py "
              f"--ckpt {winner['ckpt']} --games 4000 --deploy")
        print("The extra games are the point: this one was selected for having "
              "the highest sample out of many, so its recorded Elo is biased "
              "upward and needs an independent match to be believed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
