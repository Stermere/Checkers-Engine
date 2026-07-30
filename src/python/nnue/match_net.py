"""Play a trained checkpoint against the frozen baseline and report Elo.

This is the fitness function. `train.py` and `grid_search.py` rank nets by
validation loss, which STATUS.md and the bucketing post-mortem both record as a
proxy that has already been wrong by 81 Elo while every training metric looked
healthy. This script closes that loop: checkpoint in, Elo out, no human in the
middle.

    ckpt -> quantize -> nnue_weights_candidate.h -> build -> bot_vs_bot -> Elo

Four things about it are deliberate.

**The committed header is never touched.** `src/engine/nnue.c` includes
`nnue_weights_candidate.h` instead when built with `NNUE_WEIGHTS_CANDIDATE=1`,
and that file is gitignored. A search that plays two hundred nets therefore
leaves the working tree exactly as it found it, and a run killed halfway leaves
nothing to restore. Promoting a winner is `--deploy`, an explicit act.

**The build is forced.** setuptools decides whether to recompile by comparing
timestamps on the *sources* it was given, and the weights arrive through a
header. So a changed net looks like no change at all, `build_ext` skips, and the
match is played by whatever net happened to be linked last - silently, and
looking exactly like a real result. `touch`ing board_search.c is what prevents
that, and it is not optional.

**Each candidate gets its own module name.** `search_engine.pyd` cannot be
relinked while anything holds it open - a data generation run, or Dropbox
mid-sync - and a match that had to relink it would be unable to run alongside
anything else. Building as `search_engine_cand` (or `--module X`) sidesteps that
entirely, and lets two candidates exist at once.

**The static eval range is measured before the match, not after.** The score
band invariant is that heuristic evals stay inside EVAL_MAX=1500 while proven
results live above WIN_MIN=2000; board_search.c clamps to enforce it, so a net
whose evals run past 1500 does not corrupt the search, it just goes flat in
exactly the won positions where it matters, and plays a match that looks
ordinary and means nothing. `--eval-scale` makes this reachable in one step, so
the clamp rate is reported every time and a bad one is refused up front.

Run:

    # measure the current checkpoint against the frozen baseline
    python src/python/nnue/match_net.py --ckpt src/python/nnue/models/net.pt

    # more games for a tighter interval; 500 games is +-33 Elo, 5000 is +-10
    python src/python/nnue/match_net.py --ckpt ... --games 5000

    # a candidate that beat the baseline, promoted into the committed header
    python src/python/nnue/match_net.py --ckpt ... --deploy
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import export
from model import CheckersNet, EVAL_SCALE

NNUE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(NNUE_DIR, '..', '..', '..'))
ENGINE_DIR = os.path.join(REPO, 'src', 'engine')

CANDIDATE_HEADER = os.path.join(ENGINE_DIR, 'nnue_weights_candidate.h')
COMMITTED_HEADER = os.path.join(ENGINE_DIR, 'nnue_weights.h')
BOARD_SEARCH_C = os.path.join(ENGINE_DIR, 'board_search.c')

PACKAGE_ENGINE = os.path.join(REPO, 'src', 'python', 'Package_engine.py')
BOT_VS_BOT = os.path.join(REPO, 'src', 'python', 'bot_vs_bot.py')

#: board_search.c clamps every static evaluation into +-EVAL_MAX. Kept here so
#: the clamp rate can be reported; if it moves there, it has to move here.
EVAL_MAX = 1500

#: the module a candidate is built as by default. Not `search_engine`, so that a
#: match never needs to relink the engine another process may be using.
DEFAULT_MODULE = 'search_engine_cand'


def log(msg=''):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def export_candidate(ckpt_path, out_path, eval_scale=None):
    """Quantize a checkpoint into a C header. Returns (header info, quantized).

    `eval_scale` overrides what the checkpoint recorded. It is a pure output
    calibration - the net's logits do not change - so overriding it here is a
    legitimate way to sweep the scale without retraining, and is exactly what
    `evolve.py` does when it treats the scale as a searchable axis.
    """
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model = CheckersNet(ckpt['l1'], ckpt['l2'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    scale = float(eval_scale if eval_scale is not None
                  else ckpt.get('eval_scale', EVAL_SCALE))

    q = export.quantize(model)
    export.write_header(q, out_path, scale)
    return {
        'ckpt': ckpt_path,
        'l1': ckpt['l1'], 'l2': ckpt['l2'],
        'eval_scale': scale,
        'ckpt_eval_scale': float(ckpt.get('eval_scale', EVAL_SCALE)),
        'val_loss': ckpt.get('val_loss'),
        'datasets': ckpt.get('datasets'),
        'acc_abs_max': q['acc_abs_max'],
    }, q


def eval_range(q, data_path, scale, n=200_000):
    """What the quantized net actually outputs, and how much of it survives.

    Runs the integer forward pass - the one the engine runs, not the float one -
    over a sample of real positions and reports the spread against EVAL_MAX.
    `db_positions.npz` is the right sample to use because it is every position
    at <=5 pieces, so it is dense in the lopsided king endings where a net's
    output is largest and where clamping therefore bites first.
    """
    if not os.path.exists(data_path):
        return None
    idx, _label, _kind = export.load_check_set(data_path, n)
    cp = export.int_forward(q, idx, scale)
    clipped = float(np.mean(np.abs(cp) >= EVAL_MAX))
    return {'n': int(len(cp)), 'min': int(cp.min()), 'max': int(cp.max()),
            'p99_abs': float(np.percentile(np.abs(cp), 99)),
            'clamp_rate': clipped}


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_engine(module, python=None, avx2=True, extra_defines=(), quiet=True):
    """Compile the engine against the candidate header, as `module`.

    Package_engine.py is responsible for noticing that the header changed -
    it lists every file in src/engine as a dependency and deletes build outputs
    older than the newest of them. That has to hold, because setuptools on its
    own stats only `board_search.c`: a new net would otherwise compile to the
    previous net's binary and this function would return a match result for a
    net that never played.
    """
    cmd = [python or sys.executable, PACKAGE_ENGINE, 'build',
           '--name', module, '--define', 'NNUE_WEIGHTS_CANDIDATE=1']
    for d in extra_defines:
        cmd += ['--define', d]
    if not avx2:
        cmd.append('--no-avx2')

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"build failed ({' '.join(cmd)}):\n"
                         f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    if not quiet:
        log(proc.stdout)
    return time.time() - t0


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------
def play_match(module, games, seconds, workers, depth, baseline, json_path,
               opponent='old', move_cap=200, python=None, quiet=False,
               extra_args=()):
    """Run bot_vs_bot and read the result back out of its JSON dump."""
    cmd = [python or sys.executable, BOT_VS_BOT, str(games), str(seconds),
           str(workers), '--depth', str(depth), '--engine', module,
           '--baseline', baseline, '--opponent', opponent,
           '--move-cap', str(move_cap), '--json', json_path]
    cmd += list(extra_args)

    # `quiet` captures rather than suppresses: bot_vs_bot's progress display is
    # worth watching interactively, but a search running hundreds of matches
    # wants it out of the way - and still wants it on the failure path.
    pipe = subprocess.PIPE if quiet else None
    proc = subprocess.run(cmd, cwd=REPO, text=True, stdout=pipe, stderr=pipe)
    if proc.returncode != 0:
        out = (proc.stdout or '') + (proc.stderr or '')
        raise SystemExit(f"match failed ({' '.join(cmd)}):\n{out[-4000:]}")

    with open(json_path) as f:
        return json.load(f)


def sprt_llr(wins, losses, draws, elo0=0.0, elo1=10.0):
    """Log likelihood ratio for "the candidate is elo1 better" vs "elo0".

    Reported alongside the Elo because a fixed-length match answers the wrong
    question: 500 games at +-33 Elo cannot distinguish +5 from -5, and a search
    that selects on that interval is selecting on its own noise. The LLR says
    how much evidence there actually is, under the pentanomial-free
    trinomial model, and the usual bounds are +-2.94 for 5%/5%.
    """
    n = wins + losses + draws
    if n == 0:
        return 0.0
    # probabilities under each hypothesis, from the Elo it implies
    def probs(elo):
        s = 1.0 / (1.0 + 10 ** (-elo / 400.0))
        d = draws / n                      # draw rate is taken as observed
        w = max(min(s - d / 2.0, 1.0 - d), 1e-9)
        return w, max(1.0 - d - w, 1e-9), d
    w0, l0, d0 = probs(elo0)
    w1, l1, d1 = probs(elo1)
    return (wins * math.log(w1 / w0) + losses * math.log(l1 / l0)
            + draws * math.log(max(d1, 1e-9) / max(d0, 1e-9)))


# ---------------------------------------------------------------------------
def evaluate_checkpoint(ckpt, module=DEFAULT_MODULE, games=500, seconds=0.1,
                        workers=14, depth=8, baseline='search_engine_old',
                        eval_scale=None, data=None, work_dir=None,
                        quiet=False, opponent='old', skip_build=False,
                        max_clamp=0.02):
    """The whole pipeline, as one call. This is what evolve.py uses.

    Raises SystemExit if the net saturates the engine's +-EVAL_MAX clamp on more
    than `max_clamp` of sampled positions - the same guard the CLI applies, and
    it has to be here too or a search would happily breed from nets whose match
    result measured the clamp rather than the network.
    """
    work_dir = work_dir or os.path.join(NNUE_DIR, 'models', 'match')
    os.makedirs(work_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(ckpt))[0]
    result_json = os.path.join(work_dir, f'{tag}.match.json')

    t0 = time.time()
    info, q = export_candidate(ckpt, CANDIDATE_HEADER, eval_scale)
    rng = eval_range(q, data or os.path.join(NNUE_DIR, 'data',
                                             'db_positions.npz'),
                     info['eval_scale'])
    if rng is not None and rng['clamp_rate'] > max_clamp:
        raise SystemExit(
            f"{rng['clamp_rate']:.2%} of evals clamped at +-{EVAL_MAX} "
            f"(limit {max_clamp:.2%}) at EVAL_SCALE {info['eval_scale']:g}")
    build_seconds = 0.0 if skip_build else build_engine(module)
    res = play_match(module, games, seconds, workers, depth, baseline,
                     result_json, opponent=opponent, quiet=quiet)

    res.update({'net': info, 'eval_range': rng,
                'build_seconds': build_seconds,
                'total_seconds': time.time() - t0,
                'llr_0_10': sprt_llr(res['wins'], res['losses'], res['draws'])})
    with open(result_json, 'w') as f:
        json.dump(res, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser(
        description="Export a checkpoint, build the engine against it, and "
                    "play it against the frozen baseline.")
    ap.add_argument('--ckpt', default=os.path.join(NNUE_DIR, 'models', 'net.pt'),
                    help="checkpoint to measure")
    ap.add_argument('--module', default=DEFAULT_MODULE,
                    help=f"module name to build as (default {DEFAULT_MODULE}). "
                         f"Deliberately not 'search_engine': relinking that one "
                         f"fails while any process holds it open")
    ap.add_argument('--baseline', default='search_engine_old',
                    help="the opponent module, i.e. the champion to beat")
    ap.add_argument('--opponent', choices=('old', 'kingsrow'), default='old',
                    help="'old' plays --baseline; 'kingsrow' plays Kingsrow, an "
                         "absolute yardstick whose numbers are not comparable "
                         "to old-mode ones")

    ap.add_argument('--games', type=int, default=500)
    ap.add_argument('--seconds', type=float, default=0.1,
                    help="seconds per move")
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--depth', type=int, default=8)

    ap.add_argument('--eval-scale', type=float, default=None,
                    help="override the checkpoint's EVAL_SCALE at export time. "
                         "The logits do not change - this is the centipawn "
                         "calibration only - so a scale sweep costs a build and "
                         "a match, not a training run")
    ap.add_argument('--data', default=os.path.join(NNUE_DIR, 'data',
                                                   'db_positions.npz'),
                    help="positions used to measure the static eval range")
    ap.add_argument('--max-clamp', type=float, default=0.02,
                    help="refuse to play if more than this fraction of sampled "
                         "evals hit the +-1500 clamp; past that the net is flat "
                         "exactly where games are decided and the match means "
                         "nothing. Pass 1.0 to measure anyway")

    ap.add_argument('--deploy', action='store_true',
                    help="on success, copy the candidate header over the "
                         "committed src/engine/nnue_weights.h. This is the only "
                         "thing here that writes a tracked file")
    ap.add_argument('--skip-build', action='store_true',
                    help="reuse the existing module binary (for re-running a "
                         "match on a net already built)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"no such checkpoint: {args.ckpt}")

    log(f"[1/4] exporting {args.ckpt}")
    info, q = export_candidate(args.ckpt, CANDIDATE_HEADER, args.eval_scale)
    log(f"      {info['l1']}x{info['l2']}  EVAL_SCALE {info['eval_scale']:g}"
        + (f"  (checkpoint said {info['ckpt_eval_scale']:g})"
           if info['eval_scale'] != info['ckpt_eval_scale'] else '')
        + (f"  val_loss {info['val_loss']:.4f}" if info['val_loss'] else ''))
    if info['datasets']:
        log(f"      trained on {', '.join(info['datasets'])}")

    log(f"[2/4] static eval range over {os.path.basename(args.data)}")
    rng = eval_range(q, args.data, info['eval_scale'])
    if rng is None:
        log(f"      skipped - {args.data} not found")
    else:
        log(f"      {rng['min']} .. {rng['max']} cp  (p99 |eval| "
            f"{rng['p99_abs']:.0f}, EVAL_MAX {EVAL_MAX})")
        log(f"      {rng['clamp_rate']:.3%} of {rng['n']:,} sampled evals hit "
            f"the clamp")
        if rng['clamp_rate'] > args.max_clamp:
            raise SystemExit(
                f"\nrefusing to play: {rng['clamp_rate']:.2%} of evals are "
                f"clamped at +-{EVAL_MAX}, over the {args.max_clamp:.2%} limit."
                f"\nThe net is saturated in exactly the won positions that "
                f"decide games, so the match would measure the clamp, not the "
                f"net. Lower --eval-scale, or pass --max-clamp 1.0 to measure "
                f"it anyway.")

    if args.skip_build:
        log(f"[3/4] skipping build, reusing {args.module}")
    else:
        log(f"[3/4] building {args.module}")
        secs = build_engine(args.module)
        log(f"      built in {secs:.1f}s")

    log(f"[4/4] {args.games} games vs {args.baseline} at {args.seconds}s/move\n")
    work_dir = os.path.join(NNUE_DIR, 'models', 'match')
    os.makedirs(work_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    result_json = os.path.join(work_dir, f'{tag}.match.json')

    res = play_match(args.module, args.games, args.seconds, args.workers,
                     args.depth, args.baseline, result_json,
                     opponent=args.opponent)

    res.update({'net': info, 'eval_range': rng,
                'llr_0_10': sprt_llr(res['wins'], res['losses'], res['draws'])})
    with open(result_json, 'w') as f:
        json.dump(res, f, indent=2)

    elo = res['elo']
    num = lambda v: f"{v:+.0f}" if isinstance(v, (int, float)) else "n/a"
    log(f"  {os.path.basename(args.ckpt)}: "
        f"{res['wins']}W {res['losses']}L {res['draws']}D  "
        f"Elo {num(elo)} ({num(res['elo_lo'])}..{num(res['elo_hi'])})  "
        f"LLR {res['llr_0_10']:+.2f} for +10 Elo")
    log(f"  result written to {result_json}")

    if args.deploy:
        if elo is not None and elo < 0:
            log(f"\n  NOT deploying: this net measured {elo:+.0f} Elo against "
                f"{args.baseline}.")
            log(f"  Copy {CANDIDATE_HEADER} over {COMMITTED_HEADER} by hand if "
                f"that is really what you want.")
            return 1
        shutil.copyfile(CANDIDATE_HEADER, COMMITTED_HEADER)
        log(f"\n  deployed -> {COMMITTED_HEADER}")
        log(f"  rebuild the engine to pick it up: "
            f"python src/python/Package_engine.py build")
    return 0


if __name__ == '__main__':
    sys.exit(main())
