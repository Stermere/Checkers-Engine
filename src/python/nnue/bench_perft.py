"""Move generation: leaf counts and raw speed, for one build or between builds.

`perft` walks the move generator and nothing else - no evaluation, no
transposition table, no pruning, no ordering. That makes it the gate for a
change to generation itself:

  * **Correctness.** Two builds agree on every perft number if and only if they
    generate the same moves from the same positions. `bench_eval_speed.py`'s
    "same tree" check is weaker here - equal search node counts prove the two
    builds agree on moves *and* on ordering *and* on every pruning decision, so
    when it says DIFFERENT it does not say which of those moved. perft isolates
    the generator.

  * **Speed, with the eval and the table taken out.** The search spends only
    part of a node in generation, so a generator change that looks small in
    nodes/s is much larger here, where it is the whole measurement.

    python src/python/nnue/bench_perft.py
    python src/python/nnue/bench_perft.py --modules search_engine search_engine_spd

With two or more modules the first is the reference: every other build's counts
are compared against it position by position, and any disagreement is printed
with the position that produced it and exits non-zero.
"""

import argparse
import importlib
import sys
import time

from engine_iface import suppress_engine_output
import openings


def perft(module, board, depth):
    """Leaf count and elapsed seconds for one position at one depth."""
    p1, p2, p1k, p2k = board
    with suppress_engine_output(True):
        start = time.perf_counter()
        n = module.perft(int(p1), int(p2), int(p1k), int(p2k), 1, depth)
        elapsed = time.perf_counter() - start
    return n, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--positions", type=int, default=6)
    ap.add_argument("--modules", nargs='+', default=["search_engine"])
    args = ap.parse_args()

    modules = []
    for name in args.modules:
        try:
            modules.append((name, importlib.import_module(name)))
        except ImportError:
            print(f"  {name} is not built - skipping "
                  f"(python src/python/Package_engine.py build --force --name {name})")
    if not modules:
        return 1

    boards = openings.get_11man_bitboards()[:args.positions]
    print(f"{len(boards)} openings, depth {args.depth}\n")

    counts = {name: [] for name, _ in modules}
    totals = {name: [0, 0.0] for name, _ in modules}

    # interleaved per board for the same reason bench_eval_speed.py interleaves:
    # a patch of background load then lands on every build equally
    for i, board in enumerate(boards):
        for name, mod in modules:
            n, t = perft(mod, board, args.depth)
            counts[name].append(n)
            totals[name][0] += n
            totals[name][1] += t
        print(f"  position {i + 1}/{len(boards)}", end='\r', flush=True)
    print(' ' * 30, end='\r')

    for name, _ in modules:
        n, t = totals[name]
        print(f"  {name:<24} {n:>14,} leaves  {t:>7.3f}s  {n / t / 1e6:>6.2f}M leaves/s")

    ref_name = modules[0][0]
    ok = True
    if len(modules) > 1:
        print()
        ref_rate = totals[ref_name][0] / totals[ref_name][1]
        for name, _ in modules[1:]:
            rate = totals[name][0] / totals[name][1]
            bad = [(i, a, b) for i, (a, b) in
                   enumerate(zip(counts[ref_name], counts[name])) if a != b]
            if bad:
                ok = False
                print(f"  {name}: MOVE GENERATION DIFFERS from {ref_name} "
                      f"on {len(bad)}/{len(boards)} positions")
                for i, a, b in bad[:5]:
                    print(f"      position {i}: {ref_name}={a:,}  {name}={b:,}  "
                          f"board={tuple(int(x) for x in boards[i])}")
            else:
                print(f"  {name} generates at {rate / ref_rate:.2f}x {ref_name} "
                      f"(identical move generation)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
