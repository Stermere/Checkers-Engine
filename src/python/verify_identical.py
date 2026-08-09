"""Prove that two builds of the engine are the same engine.

Usage:

    python src/python/verify_identical.py                 # compare ref vs current
    python src/python/verify_identical.py --full          # deeper, slower
    python src/python/verify_identical.py --a NAME --b NAME

The `--module` / `--out` form is how this script invokes itself in a subprocess,
and how verify_wasm.mjs asks it for a reference dump; there is no reason to run
it by hand.
"""

import argparse
import ctypes
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile

MASK64 = (1 << 64) - 1

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
BUILD_ROOT = os.path.join(REPO_ROOT, 'build')


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
# 8x8 matrices, board[row][col], 0 empty / 1 p1 man / 2 p2 man / 3 p1 king /
# 4 p2 king - the representation src/python/bitboard_converter.py consumes.
# Player 1 moves toward row 0 and promotes there, player 2 toward row 7.
#
# The set is chosen to reach code that a start-position-only test never would:
# a position with two different double jumps available (the multi-jump chain
# logic), and a spread of piece counts from 2 to 6 so that every endgame
# database branch - hit, miss, and above DB_MAX_TOTAL - is exercised.
FIXTURES = [
    ("start", [
        [0, 2, 0, 2, 0, 2, 0, 2],
        [2, 0, 2, 0, 2, 0, 2, 0],
        [0, 2, 0, 2, 0, 2, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
    ]),
    ("midgame", [
        [0, 2, 0, 2, 0, 2, 0, 2],
        [2, 0, 2, 0, 0, 0, 2, 0],
        [0, 2, 0, 2, 0, 2, 0, 2],
        [0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0],
    ]),
    # one man, two different double jumps - the case multi-jump handling and
    # forced_pos continuation both live on
    ("double_jump", [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]),
    # 2 pieces: the smallest database slice there is
    ("db_2piece", [
        [0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0],
    ]),
    # 3 pieces, kings only ("easy solve" from BoardOpperations.py)
    ("db_3piece", [
        [0, 4, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]),
    # 4 pieces, kings only
    ("db_4piece_kings", [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0],
        [0, 3, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]),
    # 4 pieces, men only - a different slice family (men rank over 28 squares,
    # kings over 32), so this covers db_extract_group's offset/max_valid path
    ("db_4piece_men", [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]),
    # 5 pieces ("medium solve")
    ("db_5piece", [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 3, 0],
    ]),
    # 6 pieces = DB_MAX_TOTAL, the last count the database can answer ("hard solve")
    ("db_6piece", [
        [0, 3, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]),
    # mixed men and kings ("struggling solve")
    ("mixed", [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 1],
        [2, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0],
    ]),
]


def to_bitboard(board):
    """8x8 matrix -> (p1, p2, p1k, p2k). Mirror of bitboard_converter.py."""
    p1 = p2 = p1k = p2k = 0
    sq = 0
    for row in board:
        for item in row:
            if item == 4:
                p2k |= 1 << sq
            elif item == 3:
                p1k |= 1 << sq
            elif item == 2:
                p2 |= 1 << sq
            elif item == 1:
                p1 |= 1 << sq
            sq += 1
    return p1, p2, p1k, p2k


# column masks that stop the diagonal shifts wrapping around the board edge;
# same constants as JUMP_COL_GE2 / JUMP_COL_LE5 in board_eval.c
_JUMP_COL_GE2 = 0xFCFCFCFCFCFCFCFC
_JUMP_COL_LE5 = 0x3F3F3F3F3F3F3F3F


def piece_has_jump(p1, p2, p1k, p2k, player, bit):
    """Can the single piece at `bit` capture, on the full board?

    A per-square narrowing of has_any_jump from board_eval.c. The occupancy is
    the real one - restricting it to the one piece would make friendly pieces
    look like empty landing squares and invent captures that are not legal.
    Used only to find genuinely valid `forced_pos` arguments.
    """
    occupied = (p1 | p2 | p1k | p2k) & MASK64
    empty = ~occupied & MASK64

    if player == 1:
        up = bit & ((p1 | p1k) & MASK64)
        down = bit & (p1k & MASK64)
        enemy = (p2 | p2k) & MASK64
    else:
        up = bit & (p2k & MASK64)
        down = bit & ((p2 | p2k) & MASK64)
        enemy = (p1 | p1k) & MASK64

    if up & _JUMP_COL_GE2 & ((enemy << 9) & MASK64) & ((empty << 18) & MASK64):
        return True
    if up & _JUMP_COL_LE5 & ((enemy << 7) & MASK64) & ((empty << 14) & MASK64):
        return True
    if down & _JUMP_COL_GE2 & (enemy >> 7) & (empty >> 14):
        return True
    if down & _JUMP_COL_LE5 & (enemy >> 9) & (empty >> 18):
        return True
    return False


def jumping_squares(p1, p2, p1k, p2k, player):
    """Squares holding a piece of `player` that has a capture available."""
    own = (p1 | p1k) if player == 1 else (p2 | p2k)
    return [sq for sq in range(64)
            if (own >> sq) & 1 and piece_has_jump(p1, p2, p1k, p2k, player, 1 << sq)]


def positions():
    """The fixed, ordered position list. Order is part of the test - see module docstring."""
    for name, board in FIXTURES:
        p1, p2, p1k, p2k = to_bitboard(board)
        for stm in (1, 2):
            yield f"{name}/p{stm}", p1, p2, p1k, p2k, stm


# ---------------------------------------------------------------------------
# engine chatter
# ---------------------------------------------------------------------------
def _crt_fflush_all():
    """fflush(NULL) in the C runtime the extension links against.

    Redirecting fd 1 is not enough on its own: the engine's printf output sits
    in the CRT's own buffer and is flushed later, landing wherever fd 1 points
    at that time. Same reasoning as nnue/engine_iface.py, duplicated here on
    purpose - importing that module would import `search_engine`, and this
    process must hold exactly one engine so the two runs cannot share a
    transposition table.
    """
    # the Windows CRTs first, then glibc - CI runs this on Linux
    for dll in ('ucrtbase', 'api-ms-win-crt-stdio-l1-1-0', 'msvcrt',
                'libc.so.6', None):
        try:
            ctypes.CDLL(dll).fflush(None)
            return
        except (OSError, AttributeError):
            continue


class suppress_output:
    def __enter__(self):
        self._null = os.open(os.devnull, os.O_WRONLY)
        self._out, self._err = os.dup(1), os.dup(2)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)
        return self

    def __exit__(self, *exc):
        _crt_fflush_all()
        os.dup2(self._out, 1)
        os.dup2(self._err, 2)
        for fd in (self._out, self._err, self._null):
            os.close(fd)
        return False


# ---------------------------------------------------------------------------
# the probe run (executed in a subprocess, one engine per process)
# ---------------------------------------------------------------------------
# search_time large enough that the elapsed-time checks can never fire, which
# is what makes a fixed-depth search a pure function of its input
NO_TIME_LIMIT = 1e9


def build_dir_for(module_name):
    major, minor = sys.version_info[:2]
    candidates = [
        os.path.join(BUILD_ROOT, f'lib.win-amd64-cpython-{major}{minor}'),
        os.path.join(BUILD_ROOT, f'lib.win-amd64-{major}.{minor}'),
    ]
    if os.path.isdir(BUILD_ROOT):
        for name in sorted(os.listdir(BUILD_ROOT), reverse=True):
            if re.match(r'lib\.[^.]+-(?:cpython-)?\d\.?\d+$', name):
                candidates.append(os.path.join(BUILD_ROOT, name))
    for d in candidates:
        if os.path.isdir(d) and any(f.startswith(module_name + '.')
                                    for f in os.listdir(d)):
            return d
    raise SystemExit(
        f"no built '{module_name}' extension under {BUILD_ROOT}.\n"
        f"Build it with:  python src/python/Package_engine.py "
        f"build --force --name {module_name}")


def run_probes(module_name, perft_depth, search_depth):
    sys.path.insert(0, build_dir_for(module_name))
    engine = importlib.import_module(module_name)

    out = {"module": module_name, "nnue_info": {}, "perft": {}, "search": {}}
    out["nnue_info"] = dict(engine.nnue_info())

    # The inputs, so this file is self-describing and the WebAssembly harness
    # (src/wasm/verify_wasm.mjs) can replay the identical sequence without
    # keeping a second copy of the fixtures. Bitboards go out as decimal
    # STRINGS: they use the full 64 bits, and JSON numbers land in a JS double,
    # which silently rounds above 2^53.
    out["positions"] = [
        {"label": label,
         "p1": str(p1), "p2": str(p2), "p1k": str(p1k), "p2k": str(p2k),
         "stm": stm,
         "forced": [-1] + jumping_squares(p1, p2, p1k, p2k, stm)}
        for label, p1, p2, p1k, p2k, stm in positions()
    ]
    out["depths"] = {"perft": perft_depth, "search": search_depth}

    with suppress_output():
        for label, p1, p2, p1k, p2k, stm in positions():
            for d in range(1, perft_depth + 1):
                out["perft"][f"{label}@{d}"] = engine.perft(p1, p2, p1k, p2k, stm, d)

        # nnue_eval over every fixture, so an evaluation change shows up as an
        # evaluation change rather than as a mysterious node-count difference
        for label, p1, p2, p1k, p2k, stm in positions():
            out.setdefault("nnue_eval", {})[label] = engine.nnue_eval(p1, p2, p1k, p2k, stm)

        # The search sequence. Deliberately one flat ordered pass: each result
        # depends on the transposition table state every earlier search left
        # behind, and that accumulated history is part of what is being compared.
        for label, p1, p2, p1k, p2k, stm in positions():
            forced_options = [-1] + jumping_squares(p1, p2, p1k, p2k, stm)
            for forced in forced_options:
                for d in range(1, search_depth + 1):
                    raw = engine.search_position(p1, p2, p1k, p2k, stm,
                                                 NO_TIME_LIMIT, d, forced)
                    move, stats = raw[0], raw[1]
                    out["search"][f"{label}|f{forced}@{d}"] = {
                        "move": list(move),
                        # every exported field, not just the interesting ones:
                        # depth, max_ply, nodes, hash_entries, eval, evals
                        "stats": list(stats),
                    }
    return out


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def compare(a, b, name_a, name_b):
    """Return a list of human-readable differences."""
    diffs = []

    if a.get("nnue_info") != b.get("nnue_info"):
        diffs.append(f"nnue_info differs:\n    {name_a}: {a.get('nnue_info')}\n"
                     f"    {name_b}: {b.get('nnue_info')}")

    for section in ("perft", "nnue_eval", "search"):
        ka, kb = set(a.get(section, {})), set(b.get(section, {}))
        if ka != kb:
            diffs.append(f"{section}: probe sets differ "
                         f"({len(ka - kb)} only in {name_a}, {len(kb - ka)} only in {name_b})")
        for key in sorted(ka & kb):
            va, vb = a[section][key], b[section][key]
            if va != vb:
                diffs.append(f"{section}[{key}]:\n    {name_a}: {va}\n    {name_b}: {vb}")
    return diffs


def dump_in_subprocess(module_name, perft_depth, search_depth, db_dir, tag):
    """Run the probes for one module in a clean process and read back the JSON."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix=f"verify_{module_name}_")
    os.close(fd)
    env = dict(os.environ)
    if db_dir is None:
        # a directory that cannot exist, so endgame_db_init loads nothing and
        # the no-database branch at board_search.c:1274 is the one under test
        env["CHECKERS_DB_DIR"] = os.path.join(REPO_ROOT, "__no_such_db__")
    else:
        env["CHECKERS_DB_DIR"] = db_dir

    cmd = [sys.executable, os.path.abspath(__file__),
           "--module", module_name, "--out", path,
           "--perft-depth", str(perft_depth),
           "--search-depth", str(search_depth)]
    print(f"  [{tag}] {module_name} ...", end="", flush=True)
    proc = subprocess.run(cmd, env=env, cwd=REPO_ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(" FAILED")
        sys.stdout.write(proc.stdout.decode("utf-8", "replace"))
        raise SystemExit(f"probe run for {module_name} exited {proc.returncode}")
    with open(path, "r") as f:
        data = json.load(f)
    os.remove(path)
    print(" ok")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="search_engine_ref",
                    help="reference module name (default: search_engine_ref)")
    ap.add_argument("--b", default="search_engine",
                    help="module under test (default: search_engine)")
    ap.add_argument("--perft-depth", type=int, default=6)
    ap.add_argument("--search-depth", type=int, default=9)
    ap.add_argument("--full", action="store_true",
                    help="perft to 8 and search to 12; slower, and what to run "
                         "before trusting a change")
    ap.add_argument("--module", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.full:
        args.perft_depth, args.search_depth = 8, 12

    # subprocess mode
    if args.module:
        data = run_probes(args.module, args.perft_depth, args.search_depth)
        with open(args.out, "w") as f:
            json.dump(data, f)
        return 0

    db_dir = os.path.join(REPO_ROOT, "db")
    have_db = os.path.isdir(db_dir)
    if not have_db:
        print(f"note: {db_dir} not found - the with-database pass will be skipped, "
              f"so the endgame probe path goes untested.")

    configs = [("no db", None)] + ([("with db", db_dir)] if have_db else [])

    total = 0
    failed = False
    for tag, d in configs:
        print(f"[{tag}] perft to {args.perft_depth}, search to {args.search_depth}")
        da = dump_in_subprocess(args.a, args.perft_depth, args.search_depth, d, tag)
        db_ = dump_in_subprocess(args.b, args.perft_depth, args.search_depth, d, tag)

        n = sum(len(da.get(s, {})) for s in ("perft", "nnue_eval", "search"))
        total += n
        diffs = compare(da, db_, args.a, args.b)
        if diffs:
            failed = True
            print(f"  {len(diffs)} DIFFERENCE(S) in {n} probes:")
            for line in diffs[:40]:
                print("   ", line)
            if len(diffs) > 40:
                print(f"    ... and {len(diffs) - 40} more")
        else:
            print(f"  {n} probes identical")

    print()
    if failed:
        print(f"FAIL: '{args.b}' does not behave identically to '{args.a}'.")
        return 1
    print(f"PASS: {total} probes identical across {len(configs)} configuration(s). "
          f"'{args.b}' is behaviourally '{args.a}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
