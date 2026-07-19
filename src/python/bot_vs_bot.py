# Pits the new search engine against the old one over the full 11-man ballot
# starting positions and reports running win/loss/draw counts plus an ELO
# estimate with a 95% confidence interval.
#
# Usage:
#   python src/python/bot_vs_bot.py [num_games] [time_per_move] [workers]
#
#   num_games     – number of games to play (default: all ballots × 2).
#                   If fewer than the total are requested, positions are
#                   randomly sampled so side-bias is cancelled automatically.
#   time_per_move – seconds the engine may think per move (default: 1.0).
#   workers       – parallel game processes (default: 8).

import sys
import os
import math
import random
import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed

# NOTE: this path insertion runs in every worker process (Windows uses 'spawn'
# so each worker re-imports this module from scratch).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib.win-amd64-cpython-314')))

import search_engine as se
import search_engine_old as seo

from copy import deepcopy
from bitboard_converter import convert_bit_move, convert_to_bitboard, convert_to_matrix
from Board_opperations import Board, check_jump_required, update_board, check_win, check_tie, generate_all_options


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
def estimate_elo(new_wins, old_wins, draws):
    """Return (elo_diff, (ci_lo, ci_hi)) for `new` minus `old`.

    Draws are scored as half-points to each side (FIDE convention).
    The 95% CI is a normal approximation around the score rate."""
    n = new_wins + old_wins + draws
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


def _render_progress(done, total, bar_width=20):
    filled = int(bar_width * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    return f"[{bar}] {done}/{total}"


def _update_display(games_done, total, new_wins, old_wins, draws, elapsed, workers):
    global _DISPLAY_LINES
    elo, (lo, hi) = estimate_elo(new_wins, old_wins, draws)

    in_flight = min(workers, total - games_done)
    progress_line = (f"  Games: {_render_progress(games_done, total)}"
                     f"  (±{in_flight} in flight)")
    stats_line    = (f"  new={new_wins}  old={old_wins}  draw={draws}  "
                     f"elo~{_fmt_elo(elo)}  CI=({_fmt_elo(lo)},{_fmt_elo(hi)})  "
                     f"elapsed={elapsed:.0f}s")

    if _DISPLAY_LINES > 0:
        sys.stdout.write(f"\x1b[{_DISPLAY_LINES}A")

    sys.stdout.write(progress_line + "\n")
    sys.stdout.write(stats_line   + "\n")
    sys.stdout.flush()
    _DISPLAY_LINES = 2


def _finalize_display(total, new_wins, old_wins, draws, elapsed):
    global _DISPLAY_LINES
    elo, (lo, hi) = estimate_elo(new_wins, old_wins, draws)
    if _DISPLAY_LINES > 0:
        sys.stdout.write(f"\x1b[{_DISPLAY_LINES}A")
    print(f"\n{'─'*60}")
    print(f"  Final result after {total} games:")
    print(f"  new={new_wins}  old={old_wins}  draw={draws}")
    print(f"  Elo diff ~ {_fmt_elo(elo)}  CI=({_fmt_elo(lo)},{_fmt_elo(hi)})")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"{'─'*60}\n")
    _DISPLAY_LINES = 0


# ---------------------------------------------------------------------------
# Single-game runner  (top-level so it is picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------
def play_game(start_board, start_player, p_time, ply_depth):
    """Play one game and return 1 (new wins), 2 (old wins), or 0 (draw).

    Runs entirely inside a worker process; engine output is suppressed via
    fd-level redirection which is safe because each process owns its fds."""
    board = Board()
    board.board = deepcopy(start_board)
    player = start_player
    history = []
    move_cap = 200  # hard ply cap to avoid runaway games
    moves_played = 0
    forced = -1  # square of a piece mid multi-jump (new engine API only)

    while True:
        history.append(deepcopy(board.board))

        p1, p2, p1k, p2k = convert_to_bitboard(board.board)

        if player == 1:
            with _suppress_fd_output():
                results = se.search_position(p1, p2, p1k, p2k, player, p_time, ply_depth, forced)
        else:
            # the reference build takes the same forced-continuation argument, and it
            # has to be passed or the old engine is allowed to continue a multi jump
            # with the wrong piece, which would make the comparison meaningless
            with _suppress_fd_output():
                results = seo.search_position(p1, p2, p1k, p2k, player, p_time, ply_depth, forced)


        best_move = convert_bit_move(results[-2])

        turn = update_board(best_move[0], best_move[1], board.board)
        if turn and check_jump_required(board.board, player, best_move[1]):
            forced = best_move[1][0] + best_move[1][1] * 8
            continue  # chain-jump: same player moves again
        forced = -1

        moves_played += 1
        player = abs(player - 3)

        win = check_win(board.board, player)
        if win == 1:
            return 1
        if win == 2:
            return 2
        if check_tie(history) or moves_played >= move_cap:
            return 0


# ---------------------------------------------------------------------------
# Main  –  must be guarded with __name__ check for multiprocessing on Windows
# ---------------------------------------------------------------------------
def main():
    p_time    = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    workers   = int(sys.argv[3])   if len(sys.argv) > 3 else 8
    ply_depth = 50

    start_states  = get11manBoards()
    total_ballots = len(start_states)

    # Default: run every ballot twice (once per side).
    default_games = total_ballots * 2
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else default_games

    print(f"[info] {total_ballots} unique 11-man ballots  |  "
          f"{num_games} games  |  {p_time}s/move  |  {workers} workers\n")

    # Build the game schedule: list of (board, start_player).
    if num_games >= default_games:
        schedule = []
        for board in start_states:
            schedule.append((board, 1))
            schedule.append((board, 2))
    else:
        random.seed(0xC4CC)
        sampled = random.sample(start_states, k=min(total_ballots, num_games))
        schedule = [(board, 1 if i % 2 == 0 else 2) for i, board in enumerate(sampled)]
        while len(schedule) < num_games:
            extra = random.sample(start_states, k=min(total_ballots, num_games - len(schedule)))
            offset = len(schedule)
            schedule += [(board, 1 if (offset + i) % 2 == 0 else 2)
                         for i, board in enumerate(extra)]

    total    = len(schedule)
    new_wins = 0
    old_wins = 0
    draws    = 0
    t0       = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Submit all games up-front; the pool keeps `workers` running at once.
        futures = {
            pool.submit(play_game, sBoard, sp, p_time, ply_depth): idx
            for idx, (sBoard, sp) in enumerate(schedule)
        }

        games_done = 0
        for fut in as_completed(futures):
            outcome = fut.result()   # re-raises any worker exception
            if outcome == 1:
                new_wins += 1
            elif outcome == 2:
                old_wins += 1
            else:
                draws += 1
            games_done += 1
            _update_display(games_done, total, new_wins, old_wins, draws,
                            time.time() - t0, workers)

    _finalize_display(total, new_wins, old_wins, draws, time.time() - t0)


if __name__ == '__main__':
    main()
