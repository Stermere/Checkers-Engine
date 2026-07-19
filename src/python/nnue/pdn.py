"""Read human games from a PDN file and replay them onto this engine's board.

Why this exists: every training position so far descends from `openings.py`'s
11-man ballot set. That is a few hundred starting points, and `STATUS.md`
records the consequence - the duplicate rate of the self-play generator *rises*
with dataset size (5.7% at 56 games, 20% at 17,000), because every game is
drawn from the same narrow mouth of the tree. Human games are the fix: 22,621
of them, spanning the whole published opening repertoire, each one a distinct
seed.

The file is `data/CheckersGames.pdn`.

Square numbering
----------------
PDN numbers the 32 dark squares 1..32, left to right and top to bottom, with
Black on 1..12 and White on 21..32. Black moves first.

This engine numbers a full 8x8 matrix, with **player 1 on rows 5..7** (the
bottom) moving toward row 0, and player 1 moves first. So PDN's Black is our
player 1, but it sits on the wrong end of the board. The mapping is therefore
the 180-degree rotation `sq32 -> 31 - sq32`, i.e. `bit -> 63 - bit`, which is
the same rotation `features.py` uses. After it:

    PDN Black (1..12, moves first, promotes at 29..32)
      == player 1 (rows 5..7, moves first, promotes at row 0)

The mapping is not taken on trust: `python pdn.py` replays every game in the
file and checks every single hop against `Board_opperations.generate_options`.
A wrong rotation or a transposed axis does not produce subtly worse data, it
produces an illegal move on ply 1 of every game.

Notation handled
----------------
`11-15` quiet move, `15x22` capture, `26x17x10x1` a capture chain with its
intermediate squares. Comments `{...}`, variations `(...)`, move numbers and
result tokens are discarded. Result tokens have to go *before* tokenising, not
after: `1/2-1/2` contains the substring `2-1`, which is a perfectly well formed
move token and would silently corrupt the game that ends with it.

Who won
-------
The `[Result]` header is read as well as the moves, because the game's outcome
is a training label in its own right (see `gen_pdn_data.py`). **`1-0` names PDN
Black, i.e. our player 1.** That orientation is checked rather than taken from
the spec, because reversing it would not fail loudly - it would train the net
on the negation of every human result.

Only 2 of the 22,621 games end in a terminal position, since masters resign, so
the check is statistical: a resigning player is nearly always material down, and
across 8,429 decisive games the final material balance comes out at +0.47 for
`1-0` and -0.44 for `0-1`, with drawn games at -0.01 as the control. `python
pdn.py` re-runs that measurement and fails if the sign ever flips.

Some games in this file are unreplayable - 19th century games recorded under
the huffing rule, where declining a capture forfeited the piece, plus ordinary
transcription errors. They are rejected rather than repaired; there are plenty
left.
"""

import os
import re
import sys
from collections import namedtuple

NNUE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NNUE_DIR)
sys.path.insert(1, os.path.abspath(os.path.join(NNUE_DIR, '..')))

import Board_opperations as bo          # noqa: E402

DEFAULT_PDN = os.path.join(NNUE_DIR, 'data', 'CheckersGames.pdn')

# Game outcome, player-absolute (converting to side-to-move relative is the
# caller's job, since only the caller knows which ply it is looking at).
# The values match the winning player number so `res == player` is the test for
# "the side to move won"; DRAW and UNKNOWN sit outside 1..2 so they cannot be
# mistaken for one.
RES_DRAW, RES_P1, RES_P2, RES_UNKNOWN = 0, 1, 2, -1

#: [Result] tag -> outcome. `1-0` is PDN Black, which this engine calls player 1
#: - see "Who won" in the module docstring, and the check in _main().
RESULT_TAGS = {'1-0': RES_P1, '0-1': RES_P2, '1/2-1/2': RES_DRAW}

#: one replayed game: `plies` as described in `replay`, plus who won
Game = namedtuple('Game', 'plies result')


# ---------------------------------------------------------------------------
# square mapping
# ---------------------------------------------------------------------------
def _pdn_to_bit(n):
    """PDN square 1..32 -> this engine's 0..63 bit index."""
    sq32 = 32 - n                       # 180 degree rotation, see module docstring
    y = sq32 // 4
    # dark squares have (x + y) odd, so the parity of the row picks the offset
    x = 2 * (sq32 % 4) + (1 if y % 2 == 0 else 0)
    return y * 8 + x


PDN_TO_BIT = {n: _pdn_to_bit(n) for n in range(1, 33)}
BIT_TO_PDN = {b: n for n, b in PDN_TO_BIT.items()}


def bit_to_xy(bit):
    return (bit % 8, bit // 8)


# ---------------------------------------------------------------------------
# lexing
# ---------------------------------------------------------------------------
_GAME_SPLIT = re.compile(r'(?=^\[Event )', re.M)
_HEADER = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$', re.M)
_COMMENT = re.compile(r'\{[^}]*\}', re.S)
_VARIATION = re.compile(r'\([^)]*\)', re.S)
# Only the drawn result is stripped textually, and only because `1/2-1/2`
# contains `2-1`, which is a well formed move token. `0-1` and `1-0` must NOT be
# stripped this way: as bare substrings they occur *inside* real moves - `20-16`
# contains `0-1` - and removing them shreds the game. They are discarded later
# instead, by the 1..32 range check, since square 0 does not exist. `/` never
# appears in a move, so this substitution cannot damage one.
_RESULT = re.compile(r'1/2-1/2')
_MOVE = re.compile(r'(\d+)((?:[-x]\d+)+)')


class BadGame(Exception):
    """This game cannot be replayed; skip it."""


def iter_raw_games(path):
    """Yield (headers dict, movetext str) for every game in the file."""
    with open(path, encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    for block in _GAME_SPLIT.split(text):
        if not block.startswith('[Event '):
            continue
        headers = dict(_HEADER.findall(block))
        movetext = _HEADER.sub('', block)
        yield headers, movetext


def parse_moves(movetext):
    """Movetext -> list of moves, each a (squares tuple, is_capture) pair.

    A capture chain keeps all of its squares, so `26x17x10x1` comes back as
    ((26, 17, 10, 1), True) and replays as three hops by one player.
    """
    text = _COMMENT.sub(' ', movetext)
    text = _VARIATION.sub(' ', text)
    text = _RESULT.sub(' ', text)

    moves = []
    for m in _MOVE.finditer(text):
        squares = [int(m.group(1))] + [int(s) for s in re.findall(r'\d+', m.group(2))]
        if any(not 1 <= s <= 32 for s in squares):
            # not a move: a stray result fragment or a mangled comment
            continue
        moves.append((tuple(squares), 'x' in m.group(2)))
    return moves


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def _expand_chain(board, player, squares):
    """A capture token -> the list of single hops it stands for.

    Usually the token already lists every landing square and this is just a
    pairwise walk. Some sources compress `26x17x10x1` to `26x1`, so when a pair
    is not one legal hop we search for a capture path between the two squares
    and use it if it is unique.
    """
    hops = []
    for a, b in zip(squares, squares[1:]):
        fa, fb = PDN_TO_BIT[a], PDN_TO_BIT[b]
        if bo.generate_options(bit_to_xy(fa), board, only_jump=True) and \
                bit_to_xy(fb) in bo.generate_options(bit_to_xy(fa), board, only_jump=True):
            hops.append((fa, fb))
            _apply(board, fa, fb)
        else:
            path = find_capture_path(board, fa, fb)
            if path is None:
                raise BadGame(f"no capture path {a}x{b}")
            for hop in path:
                hops.append(hop)
                _apply(board, *hop)
    return hops


def find_capture_path(board, start, end, depth=0):
    """Depth-first search for a capture chain from `start` to `end`.

    Takes and returns 0..63 bit indices, and leaves `board` as it found it.
    Returns the hop list, or None if the two squares are not connected by a
    chain of captures.

    Public because it is not only the PDN reader that meets a compressed
    capture: Kingsrow prints its move the same way, `6x29` with the landing
    squares left out, so `gen_selfplay_data.py` reconstructs its chains through
    here rather than carrying a second copy of this.

    Slow, and allowed to be: it only runs on the compressed notation.

    If two different chains connect the two squares - possible, and they may
    capture different pieces - this returns the first one found. Both are legal
    moves, so for the purpose of continuing a game either will do.
    """
    if depth > 12:
        return None
    for landing in bo.generate_options(bit_to_xy(start), board, only_jump=True):
        land_bit = landing[0] + landing[1] * 8
        snapshot = [row[:] for row in board]
        more = _apply(board, start, land_bit)
        if land_bit == end:
            board[:] = snapshot
            return [(start, land_bit)]
        if more:
            rest = find_capture_path(board, land_bit, end, depth + 1)
            if rest is not None:
                board[:] = snapshot
                return [(start, land_bit)] + rest
        board[:] = snapshot
    return None


def _apply(board, from_bit, to_bit):
    """Play one hop. Returns True if the same player must jump again."""
    start, end = bit_to_xy(from_bit), bit_to_xy(to_bit)
    jumped = bo.update_board(start, end, board)
    return bool(jumped and bo.check_jump_required(board, 0, end))


def replay(moves, strict_forced=True):
    """Replay a parsed game. Returns a list of plies.

    Each ply is `(board_before, player, hops)` where `board_before` is a fresh
    matrix board with `player` to move and `hops` is the list of
    `(from_bit, to_bit)` single hops that make up the move. The board is
    *before* the move, which is the position we want to label.

    Raises `BadGame` on anything that does not replay legally. `strict_forced`
    rejects a quiet move played while a capture was available, which is how the
    huffing games in this file give themselves away.
    """
    board = [row[:] for row in bo.Board().board]
    player = 1
    plies = []

    for squares, is_capture in moves:
        from_bit = PDN_TO_BIT[squares[0]]
        piece = board[from_bit // 8][from_bit % 8]
        if piece == 0:
            raise BadGame(f"no piece on {squares[0]}")
        if piece not in (player, player + 2):
            raise BadGame(f"piece on {squares[0]} is not player {player}'s")

        jumpers = bo.check_jump_required(board, player)
        if is_capture and not jumpers:
            raise BadGame("capture recorded with no capture available")
        if not is_capture and jumpers and strict_forced:
            raise BadGame("quiet move recorded with a capture available")

        before = [row[:] for row in board]
        if is_capture:
            hops = _expand_chain(board, player, squares)
        else:
            to_bit = PDN_TO_BIT[squares[1]]
            if bit_to_xy(to_bit) not in bo.generate_options(bit_to_xy(from_bit), board):
                raise BadGame(f"illegal move {squares[0]}-{squares[1]}")
            hops = [(from_bit, to_bit)]
            _apply(board, from_bit, to_bit)

        plies.append((before, player, hops))
        player ^= 3

    return plies


def final_board(plies):
    """The position after the last recorded move, and whose turn it would be.

    `replay` hands back each position *before* its move, which is what wants
    labelling, so the end of the game is not in that list at all. Reconstructed
    here rather than returned by `replay` because only the outcome checks need
    it.
    """
    board, player, hops = plies[-1]
    board = [row[:] for row in board]
    for from_bit, to_bit in hops:
        _apply(board, from_bit, to_bit)
    return board, player ^ 3


def load_games(path=DEFAULT_PDN, limit=None, strict_forced=True, verbose=False):
    """Every replayable game in the file, as `Game(plies, result)`.

    Also returns a stats dict, because the rejection rate is the only signal
    that the square mapping is right - it should be a couple of percent, not a
    third.
    """
    games, stats = [], {'total': 0, 'ok': 0, 'rejected': 0, 'reasons': {},
                        'no_result': 0}
    for headers, movetext in iter_raw_games(path):
        stats['total'] += 1
        moves = parse_moves(movetext)
        if not moves:
            stats['rejected'] += 1
            stats['reasons']['empty'] = stats['reasons'].get('empty', 0) + 1
            continue
        try:
            plies = replay(moves, strict_forced=strict_forced)
        except BadGame as exc:
            stats['rejected'] += 1
            key = str(exc).split(' ')[0] + ' ' + str(exc).split(' ')[1] if ' ' in str(exc) else str(exc)
            stats['reasons'][key] = stats['reasons'].get(key, 0) + 1
            if verbose:
                print(f"  reject {headers.get('Event', '?')}: {exc}")
            continue
        except (KeyError, IndexError, RecursionError) as exc:
            stats['rejected'] += 1
            stats['reasons'][type(exc).__name__] = \
                stats['reasons'].get(type(exc).__name__, 0) + 1
            continue
        # an unreadable or missing Result costs the outcome label but not the
        # positions, which are still perfectly good search targets
        result = RESULT_TAGS.get(headers.get('Result', ''), RES_UNKNOWN)
        if result == RES_UNKNOWN:
            stats['no_result'] += 1
        stats['ok'] += 1
        games.append(Game(plies, result))
        if limit and len(games) >= limit:
            break
    return games, stats


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------
def _main():
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pdn', default=DEFAULT_PDN)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--loose', action='store_true',
                    help="accept quiet moves played with a capture available "
                         "(the huffing rule) instead of rejecting the game")
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    # the mapping itself, before any game is read
    assert PDN_TO_BIT[1] == 62 and PDN_TO_BIT[32] == 1, PDN_TO_BIT
    start = bo.Board().board
    for n in range(1, 13):          # PDN black -> our player 1
        b = PDN_TO_BIT[n]
        assert start[b // 8][b % 8] == 1, (n, b)
    for n in range(21, 33):         # PDN white -> our player 2
        b = PDN_TO_BIT[n]
        assert start[b // 8][b % 8] == 2, (n, b)
    for n in range(13, 21):
        b = PDN_TO_BIT[n]
        assert start[b // 8][b % 8] == 0, (n, b)
    print("square mapping: PDN 1-12 -> player 1, 21-32 -> player 2, 13-20 empty  OK")

    t0 = time.time()
    games, stats = load_games(args.pdn, limit=args.limit,
                              strict_forced=not args.loose, verbose=args.verbose)
    el = time.time() - t0

    plies = sum(len(g.plies) for g in games)
    print(f"\n{stats['total']:,} games in file")
    print(f"{stats['ok']:,} replayed cleanly ({stats['ok'] / stats['total']:.1%})")
    print(f"{stats['rejected']:,} rejected")
    for reason, n in sorted(stats['reasons'].items(), key=lambda kv: -kv[1]):
        print(f"    {n:>6,}  {reason}")
    print(f"\n{plies:,} plies total, {plies / max(len(games), 1):.1f} per game")
    print(f"{el:.1f}s")

    _check_result_orientation(games)

    # unique positions available, as a diversity ceiling
    import bitboard_converter as bc
    seen = set()
    for g in games:
        for board, player, _ in g.plies:
            seen.add(bc.convert_to_bitboard(board) + (player,))
    print(f"{len(seen):,} unique positions ({len(seen) / max(plies, 1):.1%} of plies)")


def _check_result_orientation(games, king_value=1.5):
    """Assert that `1-0` names player 1, by looking at who is winning at the end.

    This is the gate on the one thing about the Result tag that cannot be
    recovered from the moves. Getting it backwards trains on the negation of
    every human outcome, and nothing else in the pipeline would notice, so it is
    measured on every run of this file rather than trusted.

    Masters resign, so only a handful of games in this file end in a position
    with no legal move - too few to conclude anything from. The material balance
    at the last recorded ply is the usable signal instead: a resigning player is
    normally material down. It is noisy per game (checkers is won by trapping as
    often as by capturing) and that is fine, because the question is only which
    of two opposite signs is right.
    """
    counts = {RES_P1: [], RES_P2: [], RES_DRAW: []}
    for g in games:
        if g.result not in counts or not g.plies:
            continue
        board, _ = final_board(g.plies)
        p1 = sum(r.count(1) for r in board) + king_value * sum(r.count(3) for r in board)
        p2 = sum(r.count(2) for r in board) + king_value * sum(r.count(4) for r in board)
        counts[g.result].append(p1 - p2)

    def mean(v):
        return sum(v) / len(v) if v else float('nan')

    win1, win2, draw = (mean(counts[k]) for k in (RES_P1, RES_P2, RES_DRAW))
    print(f"\nresult orientation: final material (player 1 minus player 2)")
    print(f"  1-0      n={len(counts[RES_P1]):>6,}  mean {win1:+.2f}")
    print(f"  0-1      n={len(counts[RES_P2]):>6,}  mean {win2:+.2f}")
    print(f"  1/2-1/2  n={len(counts[RES_DRAW]):>6,}  mean {draw:+.2f}   (control, want ~0)")

    if not (win1 > 0 > win2):
        raise SystemExit(
            "\nFAIL: '1-0' games do not end with player 1 ahead on material.\n"
            "RESULT_TAGS maps '1-0' to player 1; this file disagrees, so either\n"
            "the mapping is wrong for this source or the reader has drifted.\n"
            "Do not generate outcome labels until this is resolved.")
    print(f"  OK: '1-0' == player 1 (PDN Black), which is what RESULT_TAGS says")


if __name__ == '__main__':
    _main()
