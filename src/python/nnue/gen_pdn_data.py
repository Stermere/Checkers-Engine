"""Training data from human games: label the positions of `CheckersGames.pdn`.

This is the diversity half of the dataset. `gen_selfplay_data.py` supplies
volume, but every position it produces descends from the same few hundred
11-man ballots, and `STATUS.md` records what that costs: the duplicate rate
climbed from 5.7% at 56 games to 20% at 17,000, because more games drawn from
the same narrow mouth of the tree keep landing on each other. 22,621 human
games reach parts of the tree self-play from a fixed book never visits.

The labels come from a search - this is distillation, same as self-play. The
human moves only decide *which* positions get labelled, never what the label
is, so a weak human move is not a wrong label, it is just an unusual position.
That is the point.

`--teacher` picks whose search. Distillation is capped by the teacher, and ours
is a depth-9 search of the engine we are trying to improve, so it can only
teach the net an eval the engine already has. `--teacher kingsrow` labels with
Kingsrow(x64) 1.19e instead, which is far stronger and, at 0.05s per position,
no slower than our own depth 9. Because this script labels a *fixed* pool, the
same positions can be labelled by both and the two nets compared directly -
which is the only way to find out whether the better teacher is worth anything
in Elo. See `kingsrow.py` for the conventions and what they cost to establish.

Two things make this cheaper per position than self-play:

  * **Deduplication happens before the searches, not after.** The whole pool is
    built and uniqued in the parent, so an opening line shared by 400 games is
    searched once. Self-play cannot do this - it does not know what a game will
    contain until it has played it, by which time the searches are spent.
  * **The filters also run before the searches.** A position with a capture
    available, or one the tablebase already owns, is dropped without ever being
    handed to a worker.

The pool is cached (`data/pdn_pool.npz`) because building it means replaying
1.16M plies in Python, which takes about a minute. It is stored unfiltered so
that changing `--min-pieces` or `--skip-plies` does not mean rebuilding it.

Output format matches `gen_selfplay_data.py` exactly - an .npz of
p1/p2/p1k/p2k/stm plus `eval_cp`, `result` and `result_n` - so `train.py --data`
takes all three sets together.

The outcome label
-----------------
`result` is how the games that reached a position actually finished, as a win
probability in [0, 1] for the side to move, and `result_n` is how many games
that is averaged over. `train.py` blends it with the search score.

Averaging is the point, and it is why this set carries a *probability* rather
than a win/draw/loss class. The pool is deduplicated across games, so a position
is one row no matter how many games reached it - and a popular opening position
is reached by hundreds of games with different results. The empirical score over
all of them is a far better label than any single game's result, and `result_n`
says how much to trust it: an opening position backed by 400 games is a real
measurement, while a ply-40 position backed by 1 is one player's bad day.

Positions from games with no readable `[Result]` get `result_n = 0`, which
`train.py` reads as "no outcome known" and falls back to the search score alone
rather than inventing a draw.

Run:
    python src/python/nnue/gen_pdn_data.py --workers 14 --depth 9
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

NNUE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NNUE_DIR)
sys.path.insert(1, os.path.abspath(os.path.join(NNUE_DIR, '..')))

import bitboard_converter as bc         # noqa: E402
import pdn                              # noqa: E402
from gen_selfplay_data import MATE_BAND, _key_view   # noqa: E402

DEFAULT_POOL = os.path.join(NNUE_DIR, 'data', 'pdn_pool.npz')

# Pool columns. The outcome is carried as integer half-points (loss 0, draw 1,
# win 2) summed over games, and a count, so the whole pool stays one int64 array
# and every existing mask over it keeps working. The probability the trainer
# wants is res_sum / (2 * res_n).
COL_PLY, COL_RES_SUM, COL_RES_N = 5, 6, 7
POOL_COLS = 8


# ---------------------------------------------------------------------------
# the position pool
# ---------------------------------------------------------------------------
def build_pool(pdn_path, verbose=True):
    """Every distinct position reached in the file, with the ply it appeared at
    and the pooled outcome of every game that reached it.

    Unfiltered on purpose; `filter_pool` applies the training filters, so the
    expensive part (replaying the games) is done once and reused.
    """
    t0 = time.time()
    games, stats = pdn.load_games(pdn_path)
    if verbose:
        print(f"  {stats['ok']:,}/{stats['total']:,} games replayed "
              f"({stats['rejected']:,} rejected, "
              f"{stats['no_result']:,} with no readable result)")

    rows = []
    for game in games:
        for ply_no, (board, player, _hops) in enumerate(game.plies):
            p1, p2, p1k, p2k = bc.convert_to_bitboard(board)
            # the label is side-to-move relative, like everything else here, so
            # the same final position scores 2 for the winner and 0 for the
            # loser depending on whose turn it is
            if game.result == pdn.RES_UNKNOWN:
                half, known = 0, 0
            elif game.result == pdn.RES_DRAW:
                half, known = 1, 1
            else:
                half, known = (2 if game.result == player else 0), 1
            rows.append((p1, p2, p1k, p2k, player, ply_no, half, known))

    arr = np.array(rows, dtype=np.int64)
    raw = len(arr)
    arr = pool_aggregate(arr)
    if verbose:
        n_lab = int((arr[:, COL_RES_N] > 0).sum())
        print(f"  {raw:,} plies -> {len(arr):,} unique positions "
              f"({(raw - len(arr)) / raw:.1%} duplicates) in {time.time() - t0:.0f}s")
        print(f"  {n_lab:,} of them ({n_lab / len(arr):.1%}) carry a game "
              f"outcome, averaging {arr[:, COL_RES_N].mean():.1f} games each "
              f"(max {arr[:, COL_RES_N].max():,})")
    return arr


def pool_aggregate(arr):
    """Collapse duplicate positions, *summing* their outcomes.

    Deliberately not `gen_selfplay_data.deduplicate`, which keeps the first row
    it sees and throws the rest away. That is right when the duplicate rows are
    identical, and wrong here: the whole value of the human set is that hundreds
    of games pass through the same opening position, and their results are what
    makes the outcome label worth having. Keeping the first would reduce a
    400-game consensus to one game's result.

    Everything except the outcome comes from the position's first appearance,
    which for `ply` means the earliest ply it was ever reached at.
    """
    _, first, inverse = np.unique(_key_view(arr), return_index=True,
                                  return_inverse=True)
    res_sum = np.bincount(inverse, weights=arr[:, COL_RES_SUM], minlength=len(first))
    res_n = np.bincount(inverse, weights=arr[:, COL_RES_N], minlength=len(first))

    order = np.argsort(first)            # back into first-appearance order
    out = arr[first[order]]
    out[:, COL_RES_SUM] = res_sum[order]
    out[:, COL_RES_N] = res_n[order]
    return out


def load_pool(pdn_path, cache, rebuild=False, verbose=True):
    if cache and os.path.exists(cache) and not rebuild:
        d = np.load(cache)
        if 'res_sum' in d.files:
            arr = np.stack([d['p1'], d['p2'], d['p1k'], d['p2k'], d['stm'],
                            d['ply'], d['res_sum'], d['res_n']],
                           axis=1).astype(np.int64)
            if verbose:
                print(f"pool: {len(arr):,} positions from cache {cache}")
            return arr
        # a cache from before outcomes were recorded. Rebuilding is a minute,
        # and the alternative is a pool that silently has no outcome labels in
        # it, which looks exactly like a pool that has them and lost them.
        if verbose:
            print(f"pool: cache {cache} predates outcome labels, rebuilding")

    if verbose:
        print(f"pool: building from {pdn_path}")
    arr = build_pool(pdn_path, verbose=verbose)
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.savez_compressed(
            cache,
            p1=arr[:, 0].astype(np.uint64), p2=arr[:, 1].astype(np.uint64),
            p1k=arr[:, 2].astype(np.uint64), p2k=arr[:, 3].astype(np.uint64),
            stm=arr[:, 4].astype(np.uint8), ply=arr[:, COL_PLY].astype(np.int16),
            res_sum=arr[:, COL_RES_SUM].astype(np.uint32),
            res_n=arr[:, COL_RES_N].astype(np.uint32))
        if verbose:
            print(f"  cached to {cache} "
                  f"({os.path.getsize(cache) / 1e6:.1f} MB)")
    return arr


def filter_pool(arr, skip_plies, min_pieces, verbose=True):
    """Apply the training filters. No search is involved, so this is free.

    Same three filters `gen_selfplay_data.py` applies, and for the same reasons:
    opening plies are near-identical across games, positions below `min_pieces`
    belong to the tablebase probe that runs ahead of the net, and the static
    eval of a position with a capture pending describes the end of the forced
    sequence rather than the board in front of you.
    """
    import engine_iface as ei

    n0 = len(arr)
    arr = arr[arr[:, 5] >= skip_plies]
    n1 = len(arr)

    pieces = np.fromiter(
        (ei.popcount(*(int(v) for v in r[:4])) for r in arr),
        dtype=np.int32, count=len(arr))
    arr = arr[pieces >= min_pieces]
    n2 = len(arr)

    keep = np.fromiter(
        (not ei.has_any_jump(int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]))
         for r in arr), dtype=bool, count=len(arr))
    arr = arr[keep]

    if verbose:
        print(f"filters: {n0:,} -> {n1:,} (ply >= {skip_plies}) "
              f"-> {n2:,} (pieces >= {min_pieces}) "
              f"-> {len(arr):,} (no capture available)")
    return arr


# ---------------------------------------------------------------------------
# labelling, in workers
# ---------------------------------------------------------------------------
#: one Kingsrow per worker process. The DLL keeps global state - hash table,
#: book, search threads - so two instances in one process would share it, and
#: constructing one is expensive enough that it must not happen per batch.
_KINGSROW = None


def _kingsrow(seconds):
    global _KINGSROW
    if _KINGSROW is None:
        import kingsrow
        _KINGSROW = kingsrow.Kingsrow(seconds=seconds)
    return _KINGSROW


def label_batch(args):
    """Label a batch of positions.

    Takes (p1,p2,p1k,p2k,stm,res_sum,res_n) rows and returns
    (p1,p2,p1k,p2k,stm,eval_cp,res_sum,res_n). The outcome columns ride along
    untouched rather than being joined back on afterwards: positions are dropped
    here when the search returns a mate score, so the output is not row-aligned
    with the input and a positional join would silently pair a label with the
    wrong position's outcome.
    """
    positions, seconds, depth, teacher = args

    if teacher == 'kingsrow':
        kr = _kingsrow(seconds)
        out = []
        for p1, p2, p1k, p2k, stm, res_sum, res_n in positions:
            board = bc.convert_to_matrix(int(p1), int(p2), int(p1k), int(p2k))
            cp = kr.eval(board, int(stm))
            # None means the engine returned no value at all, which happens on
            # positions it answers from the book or refuses to search
            if cp is not None and abs(cp) < MATE_BAND:
                out.append((int(p1), int(p2), int(p1k), int(p2k), int(stm),
                            int(cp), int(res_sum), int(res_n)))
        return out

    import engine_iface as ei

    out = []
    for p1, p2, p1k, p2k, stm, res_sum, res_n in positions:
        res = ei.search(int(p1), int(p2), int(p1k), int(p2k), int(stm),
                        seconds, depth)
        if abs(res.eval) < MATE_BAND:
            out.append((int(p1), int(p2), int(p1k), int(p2k), int(stm),
                        int(res.eval), int(res_sum), int(res_n)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pdn', default=pdn.DEFAULT_PDN)
    ap.add_argument('--pool-cache', default=DEFAULT_POOL)
    ap.add_argument('--rebuild-pool', action='store_true')
    ap.add_argument('--pool-only', action='store_true',
                    help="build and cache the pool, then stop without searching")
    ap.add_argument('--positions', type=int, default=0,
                    help="how many to label; 0 means all of them")
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--teacher', choices=('self', 'kingsrow'), default='self',
                    help="who produces the label: our own search, or Kingsrow "
                         "(much stronger, see kingsrow.py). Both report in the "
                         "same units and the same side-to-move-relative "
                         "convention, but they are still different oracles - "
                         "do not mix them within one training set")
    ap.add_argument('--depth', type=int, default=9,
                    help="must match the other datasets: a net cannot be "
                         "trained on two different oracles at once. Ignored "
                         "for --teacher kingsrow, which has no depth control")
    ap.add_argument('--seconds', type=float, default=5.0,
                    help="per-position time cap; keep it above what --depth "
                         "needs, or the file silently mixes search depths. "
                         "For --teacher kingsrow this is the whole budget "
                         "(0.05 reaches depth 15-19)")
    ap.add_argument('--skip-plies', type=int, default=6)
    ap.add_argument('--min-pieces', type=int, default=6)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(NNUE_DIR, 'data',
                                                  'pdn_positions.npz'))
    args = ap.parse_args()

    arr = load_pool(args.pdn, args.pool_cache, args.rebuild_pool)
    if args.pool_only:
        return 0

    arr = filter_pool(arr, args.skip_plies, args.min_pieces)
    if not len(arr):
        print("nothing left after filtering")
        return 1

    rng = np.random.default_rng(args.seed)
    arr = arr[rng.permutation(len(arr))]
    if args.positions and args.positions < len(arr):
        arr = arr[:args.positions]

    oracle = (f"kingsrow at {args.seconds}s" if args.teacher == 'kingsrow'
              else f"our own search at depth {args.depth}")
    print(f"\nlabelling {len(arr):,} positions with {oracle} "
          f"on {args.workers} workers\n")

    positions = [(int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]),
                  int(r[COL_RES_SUM]), int(r[COL_RES_N])) for r in arr]
    jobs = [(positions[i:i + args.batch], args.seconds, args.depth,
             args.teacher)
            for i in range(0, len(positions), args.batch)]

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, batch in enumerate(pool.map(label_batch, jobs), start=1):
            rows.extend(batch)
            if i % 20 == 0 or i == len(jobs):
                el = time.time() - t0
                frac = i / len(jobs)
                eta = el / frac - el if frac else 0
                print(f"  {i}/{len(jobs)} batches  {len(rows):,} labelled  "
                      f"{len(rows) / el:.0f} pos/s  eta {eta / 60:.0f}m")

    if not rows:
        print("no positions produced")
        return 1

    out = np.array(rows, dtype=np.int64)
    dropped = len(arr) - len(out)
    print(f"\n{len(out):,} labelled ({dropped:,} dropped as "
          f"|eval| >= {MATE_BAND} mate/tablebase scores)")

    out = out[rng.permutation(len(out))]
    eval_cp = out[:, 5].astype(np.int32)
    # half-points over 2*games -> a win probability for the side to move.
    # result_n == 0 rows keep result 0.5, which is never read: train.py gates on
    # result_n, so an unlabelled row falls back to the search score rather than
    # being taught that it is a draw.
    # eval_cp *replaces* ply at column 5, so the outcome columns keep their
    # pool indices
    res_n = out[:, COL_RES_N]
    res_sum = out[:, COL_RES_SUM]
    result = np.where(res_n > 0, res_sum / (2.0 * np.maximum(res_n, 1)), 0.5)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        p1=out[:, 0].astype(np.uint64), p2=out[:, 1].astype(np.uint64),
        p1k=out[:, 2].astype(np.uint64), p2k=out[:, 3].astype(np.uint64),
        stm=out[:, 4].astype(np.uint8), eval_cp=eval_cp,
        result=result.astype(np.float32),
        result_n=np.minimum(res_n, 65535).astype(np.uint16))

    print("\neval distribution (centipawns, side-to-move relative):")
    for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
        print(f"  p{q:<3} {np.percentile(eval_cp, q):>8.0f}")
    print(f"  mean {eval_cp.mean():.1f}  std {eval_cp.std():.1f}")
    print(f"  |eval| <= 25: {(np.abs(eval_cp) <= 25).mean():.1%}")

    have = res_n > 0
    print(f"\ngame outcome: {have.sum():,}/{len(out):,} positions "
          f"({have.mean():.1%}) carry one")
    if have.any():
        r = result[have]
        print(f"  mean score for the side to move {r.mean():.3f}  "
              f"(0.5 = the human games broke even from here)")
        print(f"  backed by {res_n[have].mean():.1f} games on average, "
              f"max {res_n[have].max():,}")
        for lo, hi in ((1, 1), (2, 4), (5, 19), (20, 10 ** 9)):
            m = have & (res_n >= lo) & (res_n <= hi)
            if m.any():
                label = f"{lo}" if lo == hi else f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
                print(f"    {label:>5} games: {m.sum():>7,} positions, "
                      f"mean score {result[m].mean():.3f}")
    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print(f"total time {(time.time() - t0) / 60:.1f}m")
    return 0


if __name__ == '__main__':
    sys.exit(main())
