"""Search speed of one build against another, at a fixed depth.

By default this is the question the network has to answer: `search_engine`
evaluates with the network (src/engine/nnue.c), `search_engine_old`
(build_old.bat) evaluates by hand, and the difference in nodes per second is
what the net has to pay back in accuracy before the engine is stronger.

    python src/python/nnue/bench_eval_speed.py [--depth N] [--positions N]

`--modules A B ...` benchmarks any set of builds made with
`Package_engine.py --name`, which is how a change to the evaluation is measured
against the engine it replaces without having to overwrite it first:

    python src/python/Package_engine.py build --force --name search_engine_dev
    python src/python/nnue/bench_eval_speed.py --modules search_engine search_engine_dev

Two rules for reading the output:

  * Node counts are comparable only between builds that visit the same tree.
    Two builds of the network that agree bit-for-bit do (and an unequal node
    count between them is a bug worth chasing, which is why it is reported).
    The OLD_ENGINE build does not: it also uses the old reduction rules.

  * Strength is measured by playing the builds against each other in
    src/python/bot_vs_bot.py, never here.
"""

import argparse
import importlib
import time

from engine_iface import suppress_engine_output
import openings


def timed_search(module, board, depth):
    """Search one position at a fixed depth, returning (nodes, seconds, move, eval)."""
    p1, p2, p1k, p2k = board
    # the engine prints its search report with C printf, so it has to be silenced
    # at the file descriptor level, not with redirect_stdout
    with suppress_engine_output(True):
        start = time.perf_counter()
        # a large time budget so the depth, not the clock, ends the search
        raw = module.search_position(int(p1), int(p2), int(p1k), int(p2k),
                                     1, 3600.0, depth, -1)
        elapsed = time.perf_counter() - start
    return raw[1][2], elapsed, tuple(raw[0]), raw[1][4]


def bench(modules, boards, depth):
    """Time every build on every board, interleaved per board.

    Interleaving matters more than it looks. The transposition table is
    persistent inside each extension, so searching the same position twice in
    one process is not a repeat measurement - the second search returns from the
    table almost immediately. That rules out the usual "run it N times and take
    the best" loop, and leaves breadth as the only way to average out a machine
    that is busy: many distinct positions, with the builds taking turns on each
    one so a slow patch of background load lands on all of them equally.
    """
    totals = {name: [0, 0.0] for name, _ in modules}
    fingerprints = {name: [] for name, _ in modules}
    per_board = {name: [] for name, _ in modules}

    for i, board in enumerate(boards):
        for name, mod in modules:
            n, t, move, ev = timed_search(mod, board, depth)
            totals[name][0] += n
            totals[name][1] += t
            per_board[name].append(t)
            fingerprints[name].append((n, move, ev))
        print(f"  position {i + 1}/{len(boards)}", end='\r', flush=True)
    print(' ' * 30, end='\r')
    return totals, per_board, fingerprints


def describe(module):
    """One line of what this build actually is, so a mislabeled run is obvious."""
    try:
        info = module.nnue_info()
    except AttributeError:
        return "(no nnue hooks)"
    simd = {0: "scalar", 1: "sse2", 2: "avx2"}.get(info.get('simd'), '?')
    what = "network" if info['use_nnue'] else "handcrafted eval"
    extra = "  SELF-CHECKING (slow)" if info.get('verify_incremental') else ""
    return f"{what}, L1={info['l1']} L2={info['l2']}, {simd}{extra}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=11)
    ap.add_argument("--positions", type=int, default=24,
                    help="distinct opening positions; more is a longer but "
                         "quieter measurement")
    ap.add_argument("--modules", nargs='+', default=None,
                    help="builds to compare (default: search_engine vs search_engine_old)")
    args = ap.parse_args()

    names = args.modules or ["search_engine", "search_engine_old"]
    modules = []
    for name in names:
        try:
            modules.append((name, importlib.import_module(name)))
        except ImportError:
            print(f"  {name} is not built - skipping "
                  f"(python src/python/Package_engine.py build --force --name {name})")

    if not modules:
        return

    boards = openings.get_11man_bitboards()[:args.positions]
    print(f"{len(boards)} openings, fixed depth {args.depth}\n")
    for name, mod in modules:
        print(f"  {name:<22} {describe(mod)}")
    print()

    totals, per_board, fingerprints = bench(modules, boards, args.depth)

    for name, _ in modules:
        nodes, seconds = totals[name]
        print(f"  {name:<24} {nodes:>12,} nodes  {seconds:>7.2f}s  "
              f"{nodes / seconds / 1e6:>6.2f}M nodes/s")

    base_name = modules[0][0]
    base_nodes, base_seconds = totals[base_name]
    base_nps = base_nodes / base_seconds
    if len(modules) > 1:
        print()
        for name, _ in modules[1:]:
            nodes, seconds = totals[name]
            ratio = (nodes / seconds) / base_nps
            # the per-position median is the robust version of the same number:
            # one position that happened to collide with a background spike can
            # move the total, but not the median
            pairs = sorted(bt / t for bt, t in zip(per_board[base_name], per_board[name]) if t > 0)
            median = pairs[len(pairs) // 2] if pairs else 0.0
            same = ("same tree" if fingerprints[name] == fingerprints[base_name]
                    else "DIFFERENT tree")
            print(f"  {name} searches at {ratio:.2f}x {base_name} "
                  f"(median per position {median:.2f}x, {same})")


if __name__ == "__main__":
    main()
