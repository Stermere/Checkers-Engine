"""Turn a shard directory into a dataset, without finishing the run.

`gen_selfplay_data.py` writes shards as it goes and merges them at the end, so
a run that was killed leaves a directory of `shard_*.npy` that is a perfectly
good dataset which simply has not had the last step done to it. Re-running the
generator resumes and eventually produces the .npz, but that means paying for
every remaining game. Sometimes what you want is the data that already exists:

  * the run was going to be stopped early anyway, and 1.7M positions is enough
  * the games left would take another six hours and the point was a smoke test
  * the run finished the games but died in the final assembly, so the shards
    are the only copy
  * the shards were recovered from a backup and the run they came from is gone

This reads them, applies exactly the same dedupe / exclude / shuffle that ends
a normal run - via `gen_selfplay_data.write_dataset`, not a second copy of it -
and writes the .npz.

**It never modifies the shard directory.** No shard is deleted, no manifest is
rewritten, nothing is renamed. The generator's rule is that shards are unlinked
only by the cleanup that runs after a successful write, and a salvage tool has
even less business breaking it: you may well want to run this, look at what
came out, and then go back and resume the original run. Delete the directory
yourself when you are done with it.

Run:
    python src/python/nnue/shards_to_npz.py data/selfplay_3m.npz.shards
    python src/python/nnue/shards_to_npz.py <dir> --out data/partial.npz \
        --exclude data/pdn_positions.npz
    python src/python/nnue/shards_to_npz.py <dir> --dry-run
"""

import argparse
import os
import sys

import numpy as np

NNUE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NNUE_DIR)
sys.path.insert(1, os.path.abspath(os.path.join(NNUE_DIR, '..')))

from gen_selfplay_data import (MANIFEST, _read_manifest, _shard_rows,   # noqa: E402
                               load_shard, write_dataset)


def find_shards(shard_dir):
    """Every readable shard in the directory, with what the manifest says.

    Returns (shards, games, notes) where `shards` is a list of
    (path, rows) in filename order and `games` is the set of game indices the
    manifest accounts for - which may be empty, since shards predating the
    manifest are still perfectly good data.
    """
    if not os.path.isdir(shard_dir):
        raise SystemExit(f"no such directory: {shard_dir}")

    header, records = _read_manifest(shard_dir)
    by_name = {r['shard']: r for r in records}

    shards, games, notes = [], set(), []
    for name in sorted(os.listdir(shard_dir)):
        if not (name.startswith('shard_') and name.endswith('.npy')):
            continue
        path = os.path.join(shard_dir, name)
        rows = _shard_rows(path)
        if rows is None:
            # left aside rather than renamed: this tool does not touch the
            # directory, it only declines to read the file
            notes.append(f"{name} is unreadable and was skipped")
            continue
        shards.append((path, rows))
        rec = by_name.get(name)
        if rec is None:
            notes.append(f"{name} has no manifest entry; its rows are still "
                         f"included, only the game accounting is missing")
        else:
            games.update(rec.get('games', ()))

    listed = set(by_name) - {os.path.basename(p) for p, _ in shards}
    for name in sorted(listed):
        notes.append(f"{name} is in the manifest but not on disk")

    return shards, games, notes, header


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('shard_dir',
                    help="the .shards directory left by a run")
    ap.add_argument('--out', default=None,
                    help="output .npz; defaults to the path the shard "
                         "directory is named after, with .partial before the "
                         "extension so a resumed run cannot be shadowed")
    ap.add_argument('--exclude', nargs='*', default=[],
                    help=".npz datasets whose positions must not appear here")
    ap.add_argument('--seed', type=int, default=0,
                    help="shuffle seed; match the run's --seed to get the "
                         "ordering that run would have produced")
    ap.add_argument('--dry-run', action='store_true',
                    help="report what is there and stop")
    args = ap.parse_args()

    shard_dir = os.path.abspath(args.shard_dir)
    shards, games, notes, header = find_shards(shard_dir)

    print(f"{shard_dir}")
    if header and header.get('config'):
        cfg = header['config']
        bits = [f"{k}={cfg[k]!r}" for k in ('teacher', 'depth', 'seconds',
                                            'seeds', 'games', 'seed')
                if k in cfg]
        print(f"  run settings: {', '.join(bits)}")
    elif not os.path.exists(os.path.join(shard_dir, MANIFEST)):
        print("  no manifest; these shards predate it or came from a backup")

    total = sum(rows for _, rows in shards)
    print(f"  {len(shards)} shard(s), {total:,} rows"
          + (f", {len(games):,} games accounted for" if games else ""))
    for n in notes:
        print(f"  ! {n}")

    if not shards:
        print("\nnothing to assemble")
        return 1

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        # "<something>.npz.shards" -> "<something>.partial.npz"
        base = shard_dir[:-len('.shards')] if shard_dir.endswith('.shards') \
            else shard_dir
        root, ext = os.path.splitext(base)
        out_path = f"{root}.partial{ext or '.npz'}"
    print(f"  -> {out_path}")

    if args.dry_run:
        print("\n--dry-run, stopping")
        return 0

    # load_shard, not np.load: shards written before outcomes were recorded are
    # 6 columns wide and get widened rather than refused
    arr = np.concatenate([load_shard(p) for p, _ in shards])
    if write_dataset(arr, out_path, args.exclude, args.seed) is None:
        return 1

    print(f"\n{shard_dir} was not modified; the run can still be resumed, "
          f"and the shards are yours to delete when you are done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
