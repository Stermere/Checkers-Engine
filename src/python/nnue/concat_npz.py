"""Join several .npz datasets into one, keeping each position once.

Data arrives in runs, not in datasets. A night of self-play, a re-run under new
openings, a salvaged shard directory - each lands as its own file, and after a
few of them `--data a.npz b.npz c.npz` is doing three things you probably did
not ask for:

  * **Positions repeat across files.** `gen_selfplay_data.py` deduplicates
    within a run and `--exclude` can subtract a file it already knows about,
    but nothing deduplicates two finished files against each other. train.py
    splits *each file* into its own train and validation halves, so a position
    in two files can sit in one file's training set and another's validation
    set - the exact leak deduplication exists to prevent, just spread across
    files. This is not hypothetical: runs seeded from the same PDN pool share
    their openings.
  * **Each file gets its own weight and its own validation split.** Fine when
    the files really are different kinds of data (tablebase vs midgame), noise
    when they are three helpings of the same thing.
  * **Order stops being random.** Concatenation is not shuffling, so this tool
    reshuffles - see `--no-shuffle`.

What comes out is a normal dataset with the same column names and dtypes
train.py already reads, so it is a drop-in replacement for the inputs.

Merging is refused when it would quietly change what a label means: `wld`
(exact tablebase win/draw/loss) and `eval_cp` (a search score) train through
different heads in train.py, so those two cannot go in one file. Mixing search
scores from different teachers or depths *is* allowed - only you know whether
that is a blend or a mistake - and `--keep` decides which copy of a repeated
position survives.

Everything is held in memory at once, which is a few hundred MB for the ~10M
row sets here and the reason this is 200 lines instead of a streaming merge.

Run:
    python src/python/nnue/concat_npz.py data/a.npz data/b.npz --out data/all.npz
    python src/python/nnue/concat_npz.py data/*.npz --out data/all.npz --dry-run
    python src/python/nnue/concat_npz.py data/new.npz data/all.npz \
        --out data/all.npz --replace --keep first
"""

import argparse
import os
import sys
import time

import numpy as np

NNUE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NNUE_DIR)
sys.path.insert(1, os.path.abspath(os.path.join(NNUE_DIR, '..')))

# the one definition of "the same position" in this tree. Reimplementing it
# here would mean a dataset could be deduplicated one way by the generator and
# another way by this tool, which is worse than the import.
from gen_selfplay_data import _key_view                    # noqa: E402

#: the board and side to move: what a duplicate is keyed on.
KEY_COLS = ('p1', 'p2', 'p1k', 'p2k', 'stm')

#: the label. Exactly one of these per file, and it must be the same one in
#: every input - train.py picks its loss off whichever it finds.
LABEL_COLS = ('wld', 'eval_cp')

#: how the games through a position finished. Optional: a file without them
#: gets result_n = 0, which is train.py's "no outcome known" and is not the
#: same as a draw.
OUTCOME_COLS = ('result', 'result_n')

#: what the generators write, and therefore what this must write back
DTYPE = {'p1': np.uint64, 'p2': np.uint64, 'p1k': np.uint64, 'p2k': np.uint64,
         'stm': np.uint8, 'wld': np.uint8, 'eval_cp': np.int32,
         'result': np.float32, 'result_n': np.uint16}


def read_dataset(path):
    """One file as {column: array}, with its label kind and dropped columns.

    Columns this tool does not know about are reported and left behind rather
    than carried: `pdn_pool.npz` keeps a `ply` and a games-played count that
    only mean something inside the pool that built them, and inventing values
    for them in the other inputs would be making data up.
    """
    if not os.path.exists(path):
        raise SystemExit(f"no such file: {path}")
    try:
        d = np.load(path)
    except (ValueError, OSError, EOFError) as e:
        raise SystemExit(f"{path}: not a readable .npz ({e})")

    with d:
        files = set(d.files)
        missing = [c for c in KEY_COLS if c not in files]
        if missing:
            raise SystemExit(f"{path}: not a position dataset - no "
                             f"{', '.join(missing)} (found {sorted(files)})")

        kinds = [c for c in LABEL_COLS if c in files]
        if not kinds:
            raise SystemExit(f"{path}: carries no label - neither 'wld' nor "
                             f"'eval_cp' (found {sorted(files)})")
        if len(kinds) > 1:
            raise SystemExit(f"{path}: carries both {' and '.join(kinds)}, so "
                             f"which label it means is ambiguous")

        wanted = KEY_COLS + (kinds[0],) + OUTCOME_COLS
        cols = {c: d[c] for c in wanted if c in files}
        extra = sorted(files - set(wanted))

    n = len(cols['p1'])
    ragged = [c for c, a in cols.items() if len(a) != n]
    if ragged:
        raise SystemExit(f"{path}: {', '.join(ragged)} disagree(s) with p1 on "
                         f"row count; the file is damaged")

    # result without result_n (or the reverse) cannot be read: the count is
    # what says whether the value is an outcome or the 0.5 placeholder
    have = [c for c in OUTCOME_COLS if c in cols]
    if len(have) == 1:
        print(f"  ! {os.path.basename(path)} has {have[0]} but not "
              f"{OUTCOME_COLS[1 - OUTCOME_COLS.index(have[0])]}; its outcomes "
              f"are unreadable and are dropped")
        cols.pop(have[0])

    return cols, kinds[0], extra


def stack(datasets, kind, with_outcomes):
    """The inputs end to end, as one column dict in the output dtypes."""
    out = {}
    for c in KEY_COLS + (kind,):
        out[c] = np.concatenate([cols[c].astype(DTYPE[c], copy=False)
                                 for cols in datasets])
    if with_outcomes:
        res, res_n = [], []
        for cols in datasets:
            n = len(cols['p1'])
            if 'result' in cols:
                res.append(cols['result'].astype(np.float32, copy=False))
                res_n.append(cols['result_n'].astype(np.uint16, copy=False))
            else:
                # 0.5 is the placeholder train.py never reads, because the 0
                # count beside it says there is no outcome here
                res.append(np.full(n, 0.5, np.float32))
                res_n.append(np.zeros(n, np.uint16))
        out['result'] = np.concatenate(res)
        out['result_n'] = np.concatenate(res_n)
    return out


def unique_index(cols, keep):
    """Row indices of the distinct positions, in input order.

    `keep` decides which copy of a repeated position wins, which is a decision
    about labels: 'first' keeps the earliest file's, 'last' the latest'. Put
    the file you trust in that position on the command line.
    """
    keys = _key_view(np.stack([cols[c].astype(np.int64) for c in KEY_COLS],
                              axis=1))
    order = np.arange(len(keys))
    if keep == 'last':
        keys, order = keys[::-1], order[::-1]
    # np.unique sorts stably when return_index is set, so this is the first
    # occurrence of each key in whichever direction we are reading
    _, idx = np.unique(keys, return_index=True)
    return np.sort(order[idx])


def describe(cols, kind):
    """The same summary the generators print, so a merge can be eyeballed."""
    n = len(cols['p1'])
    if kind == 'eval_cp':
        cp = cols['eval_cp']
        print("\neval distribution (centipawns, side-to-move relative):")
        for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
            print(f"  p{q:<3} {np.percentile(cp, q):>8.0f}")
        print(f"  mean {cp.mean():.1f}  std {cp.std():.1f}")
        print(f"  |eval| <= 25: {(np.abs(cp) <= 25).mean():.1%}")
    else:
        print("\nwin/draw/loss labels:")
        for name, v in (('draw', 0), ('win', 1), ('loss', 2)):
            m = cols['wld'] == v
            print(f"  {name:<5} {m.sum():>10,} ({m.mean():.1%})")

    if 'result_n' in cols:
        known = cols['result_n'] > 0
        print(f"\ngame outcome: {known.sum():,}/{n:,} positions "
              f"({known.mean():.1%}) carry one")
        if known.any():
            print(f"  mean score for the side to move "
                  f"{cols['result'][known].mean():.3f}, backed by "
                  f"{cols['result_n'][known].mean():.1f} games on average")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('inputs', nargs='+',
                    help=".npz datasets to join, in priority order (see --keep)")
    ap.add_argument('--out', required=True,
                    help="output .npz; may be one of the inputs with --replace")
    ap.add_argument('--keep', choices=('first', 'last'), default='first',
                    help="which copy of a repeated position survives: the one "
                         "from the earliest file listed, or the latest "
                         "(default: first)")
    ap.add_argument('--no-shuffle', action='store_true',
                    help="keep input order. train.py splits off validation as "
                         "a prefix and relies on the file being shuffled, so "
                         "this leaves a set whose validation half is the first "
                         "input and nothing else")
    ap.add_argument('--seed', type=int, default=0, help="shuffle seed")
    ap.add_argument('--replace', action='store_true',
                    help="allow writing over an existing output; the old file "
                         "is renamed aside, never deleted")
    ap.add_argument('--dry-run', action='store_true',
                    help="report what the merge would produce and stop")
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    if not out_path.endswith('.npz'):
        # np.savez_compressed appends .npz itself, which would put the file
        # somewhere other than where --out said
        out_path += '.npz'

    # read everything before touching the output: --out may be an input
    print(f"{len(args.inputs)} input(s):")
    datasets, kinds = [], set()
    for path in args.inputs:
        cols, kind, extra = read_dataset(os.path.abspath(path))
        datasets.append(cols)
        kinds.add(kind)
        note = "" if 'result_n' in cols else ", no outcomes"
        print(f"  {os.path.basename(path):<40} {len(cols['p1']):>10,} rows  "
              f"{kind}{note}")
        for c in extra:
            print(f"  ! {c} is not a column this tool carries; dropped")

    if len(kinds) > 1:
        raise SystemExit(
            f"\nrefusing to merge {' and '.join(sorted(kinds))} labelled sets: "
            f"train.py reads exact tablebase results and search scores through "
            f"different losses, and one file can only say which it holds. "
            f"Merge each kind separately and pass both to --data.")
    kind = kinds.pop()

    if len(datasets) < 2:
        print("\nonly one input: this will still dedupe it"
              + ("" if args.no_shuffle else " and reshuffle it"))

    with_outcomes = any('result_n' in c for c in datasets)
    if with_outcomes and not all('result_n' in c for c in datasets):
        print("  ! some inputs carry no outcomes; their rows are written as "
              "'no outcome known', which train.py falls back from")

    # which input each row came from, so the merge can report how much of each
    # file survived - the interesting number is not how big an input was but
    # how much of it was new
    source = np.concatenate([np.full(len(c['p1']), i, np.int32)
                             for i, c in enumerate(datasets)])

    cols = stack(datasets, kind, with_outcomes)
    del datasets

    raw_n = len(cols['p1'])
    idx = unique_index(cols, args.keep)
    dropped = raw_n - len(idx)
    print(f"\n{raw_n:,} positions, {len(idx):,} unique "
          f"({dropped:,} duplicates dropped, {dropped / raw_n:.1%}, "
          f"keeping the {args.keep} copy)")

    if len(args.inputs) > 1:
        kept = np.bincount(source[idx], minlength=len(args.inputs))
        total = np.bincount(source, minlength=len(args.inputs))
        for path, k, t in zip(args.inputs, kept, total):
            print(f"  {os.path.basename(path):<40} {k:>10,} kept of {t:,} "
                  f"({k / t:.1%})")
    del source

    if not args.no_shuffle:
        rng = np.random.default_rng(args.seed)
        idx = idx[rng.permutation(len(idx))]
    cols = {c: a[idx] for c, a in cols.items()}

    describe(cols, kind)

    print(f"\n-> {out_path}")
    if args.dry_run:
        print("--dry-run, nothing written")
        return 0

    if os.path.exists(out_path):
        if not args.replace:
            raise SystemExit(f"{out_path} already exists. Pass --replace to "
                             f"move it aside and write a new one, or pick "
                             f"another --out.")
        # renamed, never unlinked: a merge is cheap to redo, but the file it
        # would overwrite may be the only copy of a run that was not
        aside = f"{out_path}.replaced-{time.strftime('%Y%m%d-%H%M%S')}"
        os.rename(out_path, aside)
        print(f"--replace: previous file moved to {os.path.basename(aside)}\n"
              f"  delete it yourself once you are happy with the merge")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **cols)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB, "
          f"{len(idx):,} positions)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
