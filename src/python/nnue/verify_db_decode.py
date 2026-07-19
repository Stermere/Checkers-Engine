"""Cross-check the numpy tablebase decoder against the C engine.

gen_db_data.py inverts the tablebase index in numpy. Its internal
decode/re-encode check only proves it is self-consistent - if the decoder and
the re-encoder shared the same misunderstanding of the format, both would
agree and both would be wrong.

This script closes that loop by handing the decoded positions to the engine and
comparing the engine's search result (which probes the same tablebase through
the C encoder) against the label we think the position has. A mismatch means
the Python decoder and the C encoder disagree about what position an index
names.

Run:  python src/python/nnue/verify_db_decode.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'build', 'lib.win-amd64-cpython-314'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import search_engine as se  # noqa: E402

import gen_db_data as G  # noqa: E402

DB_DRAW, DB_WIN, DB_LOSS = 0, 1, 2
NAMES = {DB_DRAW: 'draw', DB_WIN: 'win', DB_LOSS: 'loss'}


def engine_verdict(p1, p2, p1k, p2k, stm, search_time=0.15, depth=30):
    """Search a position and bucket the eval into win / draw / loss."""
    r = se.search_position(int(p1), int(p2), int(p1k), int(p2k), int(stm), search_time, depth)
    ev = r[1][4]
    if ev > 500:
        return DB_WIN, ev
    if ev < -500:
        return DB_LOSS, ev
    return DB_DRAW, ev


def main():
    per_slice = 6
    rng = np.random.default_rng(12345)

    slices = G.discover_slices(os.path.join(REPO_ROOT, 'db'))
    print(f"checking {len(slices)} slices, {per_slice} positions each\n")

    agree = 0
    total = 0
    failures = []

    for (m1, k1, m2, k2, size, path) in slices:
        got = G.sample_slice(path, m1, k1, m2, k2, size, per_slice, rng, verify=True)
        if got is None:
            continue
        p1, p2, p1k, p2k, stm, wld = got

        for i in range(len(p1)):
            verdict, ev = engine_verdict(p1[i], p2[i], p1k[i], p2k[i], stm[i])
            total += 1
            if verdict == int(wld[i]):
                agree += 1
            else:
                failures.append((f"{m1}{k1}{m2}{k2}", int(p1[i]), int(p2[i]),
                                 int(p1k[i]), int(p2k[i]), int(stm[i]),
                                 NAMES[int(wld[i])], NAMES[verdict], ev))

    print(f"engine agreed with the decoded label on {agree}/{total} "
          f"({agree / total:.1%})")

    if failures:
        print(f"\n{len(failures)} disagreements (first 15):")
        for f in failures[:15]:
            print(f"  slice {f[0]}  stm={f[5]}  db={f[6]:5s} engine={f[7]:5s} "
                  f"eval={f[8]}  p1={f[1]} p2={f[2]} p1k={f[3]} p2k={f[4]}")
        print("\nNOTE: a small number of disagreements can be legitimate - the")
        print("engine may not resolve a deep win inside the time limit, and it")
        print("scores some drawn-by-repetition lines differently. A large")
        print("fraction means the decoder is wrong.")
    return 0 if agree / total > 0.95 else 1


if __name__ == '__main__':
    sys.exit(main())
