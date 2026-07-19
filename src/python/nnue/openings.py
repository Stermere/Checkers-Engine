"""Varied starting positions for self-play.

The 11-man ballot: remove one man from each side's back two rows, then play one
move for each side. This is the same construction `train_nn.get11manBoards()`
and `bot_vs_bot.get11manBoards()` use, kept here so the nnue package does not
have to import either of those modules (both hardcode a build directory, and
`train_nn` points at a stale one).

Returned positions are bitboard tuples `(p1, p2, p1k, p2k)` with player 1 to
move, since that is what the engine and the feature encoder both want.
"""

import os
import random
import sys
from copy import deepcopy

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import Board_opperations as bo          # noqa: E402
import bitboard_converter as bc         # noqa: E402


def get_11man_boards():
    """All unique 11-man ballot positions, as matrix boards with p1 to move."""
    p1, p2, _, _ = bc.convert_to_bitboard(bo.Board().board)

    boards = []
    for i in [55, 53, 51, 49, 46, 44, 42, 40]:
        for j in [8, 10, 12, 14, 17, 19, 21, 23]:
            boards.append(bc.convert_to_matrix(p1 ^ (1 << i), p2 ^ (1 << j), 0, 0))

    with_moves = []
    for board in boards:
        # one move each, excluding moves off the back rank (those weaken the
        # position in a way that is not representative of real play)
        for move in _quiet_moves(board, 1):
            b1 = deepcopy(board)
            bo.update_board(move[0], move[1], b1)
            for move2 in _quiet_moves(b1, 2):
                b2 = deepcopy(b1)
                bo.update_board(move2[0], move2[1], b2)
                with_moves.append(b2)

    unique = list({tuple(map(tuple, b)) for b in with_moves})
    return [[list(row) for row in b] for b in unique]


def _quiet_moves(board, player):
    moves = bo.generate_all_options(board, player, False)
    return [m for m in moves if m[0][1] != 0 and m[0][1] != 7]


def get_11man_bitboards():
    """The ballots as (p1, p2, p1k, p2k) tuples."""
    return [bc.convert_to_bitboard(b) for b in get_11man_boards()]


def sample_openings(n, seed=0):
    """`n` ballot boards sampled with replacement (matrix form)."""
    boards = get_11man_boards()
    rng = random.Random(seed)
    return [deepcopy(rng.choice(boards)) for _ in range(n)]


if __name__ == '__main__':
    b = get_11man_boards()
    print(f"{len(b)} unique 11-man ballot positions")
