# Pits the new search engine against a reference opponent over the full 11-man
# ballot starting positions and reports running win/loss/draw counts plus an ELO
# estimate with a 95% confidence interval.
#
# Usage:
#   python src/python/bot_vs_bot.py [num_games] [time_per_move] [workers]
#                                   [--opponent old|kingsrow] [options]
#
#   num_games     – number of games to play (default: all ballots × 2).
#                   If fewer than the total are requested, positions are
#                   randomly sampled so side-bias is cancelled automatically.
#   time_per_move – seconds the engine may think per move (default: 1.0).
#   workers       – parallel game processes (default: 8).
#
# `--engine` and `--baseline` choose which compiled modules play; they default
# to `search_engine` and `search_engine_old`. `--json PATH` writes the result in
# a form another script can read, which is how nnue/match_net.py drives this.
#
# The opponent is `search_engine_old` (the frozen baseline build) by default.
# `--opponent kingsrow` plays Kingsrow(x64) instead, through the CheckerBoard
# engine DLL, which is a far stronger yardstick and an absolute one: the old
# build only ever says whether this change helped, while Kingsrow says where the
# engine actually stands. Three things about that mode are worth knowing before
# reading a number off it.
#
#   * **Nobody is holding the clocks.** Kingsrow has no depth control and treats
#     its time as nominal, usually returning early, so `--kr-seconds` equal to
#     `time_per_move` is not equal thinking time. The Elo is against Kingsrow
#     *at that setting*, and it only means something quoted with it.
#   * **Our engine sees only its own turns**, where in an old-vs-new match both
#     engines are asked about every position. That is how real play works and it
#     is the more honest condition, but it changes what `board_search.c` has in
#     GAME_HISTORY, so its in-search repetition handling is not being exercised
#     the same way. Do not compare a kingsrow-mode Elo against an old-mode one.
#   * **Expect draws.** Kingsrow is stronger than this engine and the ballots
#     are sound openings, so the interesting statistic is usually the loss rate,
#     not the win rate.
#
# Kingsrow's `getmove` needs the process to be chdir'd into the CheckerBoard
# directory (see nnue/kingsrow.py), which is why this module pins the engine's
# own tablebase path at import time - see CHECKERS_DB_DIR below.

import argparse
import json
import sys
import os
import math
import random
import time
from collections import namedtuple
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed

# NOTE: these run in every worker process (Windows uses 'spawn' so each worker
# re-imports this module from scratch), and they are resolved to absolute paths
# here, at import, because a Kingsrow worker chdirs itself away later.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', 'build', 'lib.win-amd64-cpython-314')))
sys.path.insert(1, os.path.join(_HERE, 'nnue'))

# `board_search.c` loads the endgame tablebase from getenv("CHECKERS_DB_DIR"),
# falling back to the *relative* path "db", on the first search of the process.
# Constructing a Kingsrow moves the process into the CheckerBoard directory,
# which has a db/ folder of its own - so without this the engine would find no
# slices it recognises there and play the entire match with no tablebase, saying
# nothing about it. Pin the path to what it would have resolved to right now,
# before anything can chdir. setdefault, so an explicit override still wins.
os.environ.setdefault('CHECKERS_DB_DIR', os.path.abspath('db'))

from copy import deepcopy
from bitboard_converter import convert_bit_move, convert_to_bitboard, convert_to_matrix
from Board_opperations import Board, check_jump_required, update_board, check_win, check_tie, generate_all_options, generate_options


# ---------------------------------------------------------------------------
# Suppress C-level stdout/stderr (engine printf output) by redirecting the
# raw file descriptors to /dev/null for the duration of each engine call.
# Safe to use in worker processes because each process has its own fd table.
# ---------------------------------------------------------------------------
@contextmanager
def _suppress_fd_output():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)


# ---------------------------------------------------------------------------
# Elo helpers
# ---------------------------------------------------------------------------
def estimate_elo(new_wins, opp_wins, draws):
    """Return (elo_diff, (ci_lo, ci_hi)) for `new` minus the opponent.

    Draws are scored as half-points to each side (FIDE convention).
    The 95% CI is a normal approximation around the score rate."""
    n = new_wins + opp_wins + draws
    if n == 0:
        return 0.0, (0.0, 0.0)
    score = (new_wins + 0.5 * draws) / n
    if score <= 0.0 or score >= 1.0:
        return float('inf') if score >= 1.0 else float('-inf'), (None, None)
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    p = score
    sigma_p = math.sqrt(p * (1 - p) / n)
    if 0.001 < p < 0.999:
        d = 400.0 / (math.log(10) * p * (1 - p))
        sigma_elo = d * sigma_p
    else:
        sigma_elo = float('inf')
    return elo, (elo - 1.96 * sigma_elo, elo + 1.96 * sigma_elo)


def _fmt_elo(v):
    if v is None:
        return "n/a"
    if v == float('inf'):
        return "+inf"
    if v == float('-inf'):
        return "-inf"
    return f"{v:+.0f}"


# ---------------------------------------------------------------------------
# 11-man ballot generator
# ---------------------------------------------------------------------------
def get11manBoards():
    boards = []
    p1, p2, _, _ = convert_to_bitboard(Board().board)
    for i in [55, 53, 51, 49, 46, 44, 42, 40]:
        next_p1 = p1 ^ (1 << i)
        for j in [8, 10, 12, 14, 17, 19, 21, 23]:
            next_p2 = p2 ^ (1 << j)
            boards.append((next_p1, next_p2, 0, 0))

    boards = [convert_to_matrix(b[0], b[1], b[2], b[3]) for b in boards]

    boards_with_moves = []
    for board in boards:
        moves = generate_all_options(board, 1, False)
        moves = [m for m in moves if m[0][1] != 0 and m[0][1] != 7]
        for move in moves:
            newBoard = deepcopy(board)
            update_board(move[0], move[1], newBoard)

            moves2 = generate_all_options(newBoard, 2, False)
            moves2 = [m for m in moves2 if m[0][1] != 0 and m[0][1] != 7]
            for move2 in moves2:
                newBoard2 = deepcopy(newBoard)
                update_board(move2[0], move2[1], newBoard2)
                boards_with_moves.append(newBoard2)

    boards_with_moves = list(set([tuple(map(tuple, b)) for b in boards_with_moves]))
    boards_with_moves = [[list(row) for row in b] for b in boards_with_moves]
    return boards_with_moves


# ---------------------------------------------------------------------------
# Dynamic in-place display (main process only)
# ---------------------------------------------------------------------------
_DISPLAY_LINES = 0


def _bar_glyphs():
    """('█', '░') if stdout can encode them, ('#', '-') otherwise.

    Redirected output on Windows is cp1252, not the console's UTF-8, so the
    block characters raise UnicodeEncodeError the moment this script is run
    under a pipe or into a file - which is exactly what match_net.py does. The
    progress display is not worth crashing a match over.
    """
    enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
    try:
        "█░".encode(enc)
        return "█", "░"
    except (UnicodeEncodeError, LookupError):
        return "#", "-"


_FULL, _EMPTY = _bar_glyphs()
_RULE = "─" if _FULL == "█" else "-"


def _render_progress(done, total, bar_width=20):
    filled = int(bar_width * done / total) if total else 0
    bar = _FULL * filled + _EMPTY * (bar_width - filled)
    return f"[{bar}] {done}/{total}"


def _update_display(games_done, total, tally, elapsed, workers, opp_name):
    global _DISPLAY_LINES
    elo, (lo, hi) = estimate_elo(tally.new_wins, tally.opp_wins, tally.draws)

    in_flight = min(workers, total - games_done)
    progress_line = (f"  Games: {_render_progress(games_done, total)}"
                     f"  (±{in_flight} in flight)")
    stats_line    = (f"  new={tally.new_wins}  {opp_name}={tally.opp_wins}  "
                     f"draw={tally.draws}  "
                     f"elo~{_fmt_elo(elo)}  CI=({_fmt_elo(lo)},{_fmt_elo(hi)})  "
                     f"elapsed={elapsed:.0f}s")

    if _DISPLAY_LINES > 0:
        sys.stdout.write(f"\x1b[{_DISPLAY_LINES}A")

    sys.stdout.write(progress_line + "\n")
    sys.stdout.write(stats_line   + "\n")
    sys.stdout.flush()
    _DISPLAY_LINES = 2


def _finalize_display(total, tally, elapsed, opp_name):
    global _DISPLAY_LINES
    elo, (lo, hi) = estimate_elo(tally.new_wins, tally.opp_wins, tally.draws)
    if _DISPLAY_LINES > 0:
        sys.stdout.write(f"\x1b[{_DISPLAY_LINES}A")
    print(f"\n{_RULE*60}")
    print(f"  Final result after {total} games:")
    print(f"  new={tally.new_wins}  {opp_name}={tally.opp_wins}  "
          f"draw={tally.draws}")
    print(f"  Elo diff ~ {_fmt_elo(elo)}  CI=({_fmt_elo(lo)},{_fmt_elo(hi)})")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"{_RULE*60}\n")
    _DISPLAY_LINES = 0


# ---------------------------------------------------------------------------
# The two opponents
# ---------------------------------------------------------------------------
#: compiled engine modules by name, imported on demand. On demand because the
#: module name is now a parameter (`--engine` / `--baseline`): a Kingsrow match
#: needs no search_engine_old.pyd lying around, and a search that builds a
#: candidate under its own name must not require the default one to exist.
#: Cached per worker process, since `import` of a 180 KB .pyd is not free and a
#: worker plays many games.
_ENGINES = {}

#: one Kingsrow per worker process. The DLL keeps global state - hash table,
#: book, search threads - so two instances in one process would share it, and
#: constructing one is far too expensive to do per game.
_KINGSROW = None


def _engine(module_name):
    mod = _ENGINES.get(module_name)
    if mod is None:
        mod = _ENGINES[module_name] = __import__(module_name)
    return mod


def _kingsrow(seconds, hashsize, cb_dir, book):
    global _KINGSROW
    if _KINGSROW is None:
        import kingsrow
        # note this chdirs the worker into the CheckerBoard directory; the paths
        # this module cares about were made absolute at import, above
        _KINGSROW = kingsrow.Kingsrow(seconds=seconds, hashsize=hashsize,
                                      cb_dir=cb_dir, book=book)
    return _KINGSROW


def _engine_move(mod, board, player, p_time, ply_depth):
    """One complete move from a compiled engine, as a list of (from, to) hops.

    The engine answers one hop at a time - a multi-jump continues with `forced`
    set to the square just landed on - so the chain is driven here instead of by
    the game loop. That is what keeps one iteration of the loop equal to one
    whole move, which in turn is what lets an outside engine be asked about
    every position the loop sees: a board halfway through a capture is not a
    position Kingsrow can be handed, because it would be free to continue the
    chain with a different piece.

    Takes and returns (x, y) squares, and does not touch `board`.
    """
    scratch = [row[:] for row in board]
    hops, forced = [], -1
    while True:
        p1, p2, p1k, p2k = convert_to_bitboard(scratch)
        with _suppress_fd_output():
            results = mod.search_position(p1, p2, p1k, p2k, player,
                                          p_time, ply_depth, forced)
        frm, to = convert_bit_move(results[-2])
        hops.append((frm, to))
        jumped = update_board(frm, to, scratch)
        if jumped and check_jump_required(scratch, player, to):
            forced = to[0] + to[1] * 8      # same player must continue the chain
            continue
        return hops


def _xy(bit):
    """0..63 bit index -> the (x, y) the Board_opperations functions take."""
    return (bit % 8, bit // 8)


def _kingsrow_move(board, player, kr):
    """One complete move from Kingsrow, as hops, or None if it named none.

    Kingsrow writes a multi-jump *compressed* - `6x29`, with the landing squares
    left out - so the hops are recovered against our own move generator. Every
    hop is checked to be legal before it is kept, and an unusable move yields
    None rather than a board that quietly stopped being a checkers position.
    """
    squares, is_capture, _source = kr.playmove(board, player)
    if squares is None:
        return None

    import pdn                              # for find_capture_path
    scratch = [row[:] for row in board]
    hops = []
    for a, b in zip(squares, squares[1:]):
        if _xy(b) in generate_options(_xy(a), scratch, only_jump=is_capture):
            step = [(a, b)]
        elif is_capture:
            # a compressed chain: find the hops that connect the two squares
            step = pdn.find_capture_path(scratch, a, b)
            if step is None:
                return None
        else:
            return None
        for frm, to in step:
            hops.append((_xy(frm), _xy(to)))
            update_board(_xy(frm), _xy(to), scratch)
    return hops


# ---------------------------------------------------------------------------
# Single-game runner  (top-level so it is picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------
#: what a worker hands back. `fallbacks` exists because a Kingsrow match every
#: ply of which had quietly fallen back to our own engine would look exactly
#: like a Kingsrow match, and be a self-play one.
GameResult = namedtuple('GameResult', 'outcome plies fallbacks')

#: running totals in the parent
Tally = namedtuple('Tally', 'new_wins opp_wins draws plies fallbacks')


def play_game(args):
    """Play one game. `outcome` is 1 (new wins), 2 (opponent wins) or 0 (draw).

    Runs entirely inside a worker process; engine output is suppressed via
    fd-level redirection which is safe because each process owns its fds.
    Player 1 is always the new engine and player 2 always the opponent;
    `start_player` only decides who moves first.
    """
    (start_board, start_player, p_time, ply_depth, move_cap,
     opponent, kr_seconds, kr_hash, cb_dir, kr_book,
     engine_mod, baseline_mod) = args
    se = _engine(engine_mod)

    kr = (_kingsrow(kr_seconds, kr_hash, cb_dir, kr_book)
          if opponent == 'kingsrow' else None)

    board = deepcopy(start_board)
    player = start_player
    history = []
    moves_played = 0
    fallbacks = 0

    while True:
        # Only whole moves reach here, so `history` holds exactly the positions
        # somebody was genuinely on move in. (It used to collect the transient
        # boards inside a multi-jump too, which `board_search.c` deliberately
        # keeps out of its own game history for the same reason.)
        history.append(deepcopy(board))

        if player == 1:
            hops = _engine_move(se, board, player, p_time, ply_depth)
        elif kr is not None:
            hops = _kingsrow_move(board, player, kr)
            if hops is None:
                # Kingsrow named no move at all - neither a PV nor a starred
                # database candidate. Rare enough to be worth reporting rather
                # than hiding, because our engine picking the opponent's moves
                # is the one way this stops being a match against Kingsrow.
                fallbacks += 1
                hops = _engine_move(se, board, player, p_time, ply_depth)
        else:
            # the reference build takes the same forced-continuation argument,
            # and it has to be passed or the old engine is allowed to continue a
            # multi jump with the wrong piece, which would make the comparison
            # meaningless
            hops = _engine_move(_engine(baseline_mod), board, player,
                                p_time, ply_depth)

        for frm, to in hops:
            update_board(frm, to, board)

        moves_played += 1
        player = abs(player - 3)

        win = check_win(board, player)
        if win != 0:
            return GameResult(win, moves_played, fallbacks)
        if check_tie(history) or moves_played >= move_cap:
            return GameResult(0, moves_played, fallbacks)


# ---------------------------------------------------------------------------
# Main  –  must be guarded with __name__ check for multiprocessing on Windows
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Play the new engine against a reference opponent and "
                    "report the Elo difference.")
    # kept positional, and in this order, because that is how every run of this
    # script has been invoked
    ap.add_argument('num_games', nargs='?', type=int, default=None,
                    help="games to play (default: every ballot twice)")
    ap.add_argument('time_per_move', nargs='?', type=float, default=1.0,
                    help="seconds our engine may think per move")
    ap.add_argument('workers', nargs='?', type=int, default=8,
                    help="parallel game processes")
    ap.add_argument('--opponent', choices=('old', 'kingsrow'), default='old',
                    help="'old' is search_engine_old, the frozen baseline "
                         "build; 'kingsrow' is Kingsrow(x64) through the "
                         "CheckerBoard DLL (default: old)")
    ap.add_argument('--kr-seconds', type=float, default=None,
                    help="Kingsrow's time per move; defaults to "
                         "time_per_move. It has no depth control and treats "
                         "this as nominal, usually returning early, so equal "
                         "numbers here are not equal thinking time")
    ap.add_argument('--kr-hash', type=int, default=64,
                    help="Kingsrow hash table in MB, per worker (default: 64)")
    ap.add_argument('--kr-book', action='store_true',
                    help="let Kingsrow use its opening book. Off by default: "
                         "the ballots already fix the openings, and a book "
                         "move is played without a search")
    ap.add_argument('--cb-dir', default=None,
                    help="CheckerBoard install directory (default: "
                         "C:\\Program Files (x86)\\CheckerBoard)")
    ap.add_argument('--move-cap', type=int, default=200,
                    help="ply cap; a game still going is adjudicated a draw. "
                         "Kingsrow games run long, so raise this if the cap "
                         "is deciding results rather than the players")
    ap.add_argument('--depth', type=int, default=50,
                    help="our engine's depth limit; the time control is "
                         "normally what binds")
    # Which compiled modules play. Both default to what every run of this script
    # has used, so nothing changes unless asked. They exist because
    # Package_engine.py --name can build a candidate alongside the engine that
    # is currently in use: match_net.py builds each net it tests under its own
    # module name, which means a match never has to relink search_engine.pyd
    # (Dropbox and any running data generation both hold that file open) and two
    # candidates can be measured without one overwriting the other.
    ap.add_argument('--engine', default='search_engine',
                    help="module played as player 1 (default: search_engine)")
    ap.add_argument('--baseline', default='search_engine_old',
                    help="module played as player 2 when --opponent old "
                         "(default: search_engine_old, the frozen baseline)")
    ap.add_argument('--json', default=None,
                    help="also write the result to this path as JSON. Parsing "
                         "the display above is not viable - it repaints itself "
                         "with cursor-movement escapes - so any script driving "
                         "this one should read the result from here")
    return ap.parse_args(argv)


def build_schedule(start_states, num_games):
    """The games to play, as (board, player_to_move) pairs."""
    total_ballots = len(start_states)
    default_games = total_ballots * 2
    if num_games is None:
        num_games = default_games

    # Default: run every ballot twice (once per side).
    if num_games >= default_games:
        schedule = []
        for board in start_states:
            schedule.append((board, 1))
            schedule.append((board, 2))
        return schedule

    random.seed(0xC4CC)
    sampled = random.sample(start_states, k=min(total_ballots, num_games))
    schedule = [(board, 1 if i % 2 == 0 else 2) for i, board in enumerate(sampled)]
    while len(schedule) < num_games:
        extra = random.sample(start_states, k=min(total_ballots, num_games - len(schedule)))
        offset = len(schedule)
        schedule += [(board, 1 if (offset + i) % 2 == 0 else 2)
                     for i, board in enumerate(extra)]
    return schedule


def main():
    args = parse_args()
    p_time = args.time_per_move
    workers = args.workers
    kr_seconds = args.kr_seconds if args.kr_seconds is not None else p_time

    if args.opponent == 'kingsrow':
        # fail here rather than in eight workers at once, several minutes in
        import kingsrow
        cb_dir = args.cb_dir or kingsrow.DEFAULT_CB_DIR
        dll = os.path.join(cb_dir, 'engines', 'Kingsrow64.dll')
        if not os.path.exists(dll):
            raise SystemExit(
                f"{dll} not found.\nKingsrow is expected where CheckerBoard "
                f"installed it; pass --cb-dir if it is elsewhere.")
        opp_name = 'kingsrow'
    else:
        cb_dir = args.cb_dir
        opp_name = 'old'

    start_states = get11manBoards()
    schedule = build_schedule(start_states, args.num_games)
    total = len(schedule)

    opp_desc = (f"kingsrow at {kr_seconds}s/move, {args.kr_hash}MB hash, "
                f"book {'on' if args.kr_book else 'off'}"
                if args.opponent == 'kingsrow' else args.baseline)
    print(f"[info] {len(start_states)} unique 11-man ballots  |  "
          f"{total} games  |  {p_time}s/move  |  {workers} workers")
    print(f"[info] {args.engine} (player 1) vs {opp_desc}")
    print(f"[info] endgame db: {os.environ['CHECKERS_DB_DIR']}"
          f"{'' if os.path.isdir(os.environ['CHECKERS_DB_DIR']) else '  (missing)'}\n")

    jobs = [(board, sp, p_time, args.depth, args.move_cap, args.opponent,
             kr_seconds, args.kr_hash, cb_dir, args.kr_book,
             args.engine, args.baseline)
            for board, sp in schedule]

    tally = Tally(0, 0, 0, 0, 0)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Submit all games up-front; the pool keeps `workers` running at once.
        futures = {pool.submit(play_game, job): idx
                   for idx, job in enumerate(jobs)}

        games_done = 0
        for fut in as_completed(futures):
            game = fut.result()   # re-raises any worker exception
            tally = Tally(tally.new_wins + (game.outcome == 1),
                          tally.opp_wins + (game.outcome == 2),
                          tally.draws + (game.outcome == 0),
                          tally.plies + game.plies,
                          tally.fallbacks + game.fallbacks)
            games_done += 1
            _update_display(games_done, total, tally,
                            time.time() - t0, workers, opp_name)

    elapsed = time.time() - t0
    _finalize_display(total, tally, elapsed, opp_name)

    if args.json:
        elo, (lo, hi) = estimate_elo(tally.new_wins, tally.opp_wins, tally.draws)
        finite = (lambda v: v if v not in (float('inf'), float('-inf'))
                  and v is not None else None)
        with open(args.json, 'w') as f:
            json.dump({
                'engine': args.engine,
                'opponent': opp_desc,
                'games': total,
                'wins': tally.new_wins, 'losses': tally.opp_wins,
                'draws': tally.draws,
                'score': (tally.new_wins + 0.5 * tally.draws) / max(total, 1),
                'elo': finite(elo), 'elo_lo': finite(lo), 'elo_hi': finite(hi),
                'plies': tally.plies, 'fallbacks': tally.fallbacks,
                'seconds_per_move': p_time, 'depth': args.depth,
                'workers': workers, 'elapsed': elapsed,
            }, f, indent=2)
        print(f"  wrote {args.json}")

    if args.opponent == 'kingsrow':
        # the only figure that says this really was a match against Kingsrow
        share = tally.fallbacks / tally.plies if tally.plies else 0.0
        print(f"  {tally.plies} plies, {tally.fallbacks} of them "
              f"({share:.2%}) played for Kingsrow by our own engine because it "
              f"named no move")
        if share > 0.01:
            print("  ! that is high enough to be biasing the result")
        print()


if __name__ == '__main__':
    main()
