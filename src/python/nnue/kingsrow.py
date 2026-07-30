"""Label positions with Kingsrow, a much stronger teacher than our own search.

Distillation is capped by the teacher. Our labels have been a depth-9 search of
our own engine, which means the net can only ever learn to imitate an eval it
already has. Kingsrow(x64) 1.19e is a far stronger engine and it is already
installed as a CheckerBoard engine, so it can label the same positions for
about the same money.

It can also *play*: `bestmove()` returns the move it chose along with its score,
from the same single search, which is how `gen_selfplay_data.py --teacher
kingsrow` gets Kingsrow-played games for one search per ply instead of two.

Everything below was established by probing the DLL, because the CheckerBoard
API documents the calling convention and nothing else. The five facts that
matter, each of which cost a crash or a wrong answer to find:

  * **The working directory must be the CheckerBoard folder.** Kingsrow
    resolves its book and database relative to the cwd, and called from
    anywhere else `getmove` dereferences a null pointer and takes the process
    down. This is why the class chdirs, which is process-global - see below.
  * **`egdb64.dll` lives in the CheckerBoard root**, not next to the engine, so
    both directories have to go on the DLL search path or the load fails with
    a bare "could not find module".
  * **The `CBmove *` parameter cannot be NULL** even though `simplech.c` calls
    it unused for English checkers. Kingsrow never fills it in (it comes back
    all zeros), it just writes through it.
  * **The score is absolute, positive favouring player 2**, and it only exists
    as text in the status string (`value=-12, depth 15/14.2/29, ...`). There is
    no eval-only entry point: `staticevaluation` replies "This engine does not
    use static evaluation". `eval()` converts to our side-to-move-relative
    convention, which is what `features.py` encodes in.
  * **The move is only in that same string too**, as the first token of the PV,
    in standard 1-32 notation. `bestmove()` maps it back onto our board;
    `SQUARE_TO_BIT` is that mapping and `python kingsrow.py` checks all 32
    squares of it against Kingsrow's own PV rather than trusting the arithmetic.
    A capture comes back *compressed* - `6x29` with the landing squares left
    out - so a chain has to be reconstructed against a move generator, which is
    the caller's job (see `pdn.find_capture_path`).

Verified against our own engine on 98 quiet PDN positions: r = +0.88 with our
depth-9 eval, and the units come out the same (std 44 vs our 48), so existing
thresholds like `MATE_BAND` carry over without rescaling. The correlation is
only meaningful on positions with no capture available - which is the filter
the generator applies anyway - because in the middle of an exchange the two
engines are scoring different points of the sequence.

Cost is about 35 ms per position at `seconds=0.05` (Kingsrow treats the time as
nominal and often returns early), reaching depth 15-19 in the midgame.

One instance per process. The DLL keeps global state - hash table, book, search
threads - so instances in one process would share it; `searchthreads 1` is set
because the parallelism here comes from running many processes.

    from kingsrow import Kingsrow
    kr = Kingsrow(seconds=0.05)
    cp = kr.eval(board, player)     # side-to-move relative centipawns
    cp, squares, is_capture = kr.bestmove(board, player)    # ...and its move
    squares, is_capture, src = kr.playmove(board, player)   # a move every ply
"""

import ctypes as C
import os
import re

#: piece encoding, from CheckerBoard's own source\simplech.c:71
OCCUPIED, WHITE, BLACK, MAN, KING, FREE = 0, 1, 2, 4, 8, 16

DEFAULT_CB_DIR = r"C:\Program Files (x86)\CheckerBoard"

Board8 = C.c_int * 8 * 8


class _Coor(C.Structure):
    _fields_ = [('x', C.c_int), ('y', C.c_int)]


class _CBmove(C.Structure):
    """simplech.c:105. Unused for English checkers but must not be NULL."""
    _fields_ = [('ismove', C.c_int), ('newpiece', C.c_int),
                ('oldpiece', C.c_int), ('from_', _Coor), ('to', _Coor),
                ('path', _Coor * 12), ('del_', _Coor * 12),
                ('delpiece', C.c_int * 12)]


_VALUE_RE = re.compile(r'value\s*[:=]?\s*([+-]?\d+)', re.I)
#: the PV, of which only the first token is a move we can trust to be legal in
#: the position we asked about. `-` is a quiet move, `x` a capture.
_PV_RE = re.compile(r'\bpv\s+(\d+(?:[-x]\d+)+)', re.I)
#: the candidate list a database position gets instead of a value and a PV:
#: `depth 6; 21-17* (0.250), 21-25 (1.000),`. Only `playmove` reads it; see
#: there for what was probed about it and what is safe to conclude.
_CAND_RE = re.compile(r'(\d+)([-x])(\d+)(\*?)\s*\(')

#: our matrix values -> CheckerBoard pieces. Our p1 starts on row 7 and
#: promotes on row 0, which is CheckerBoard's WHITE; our playable squares are
#: (x + y) odd against CheckerBoard's (x + y) even, so x is mirrored. Confirmed
#: by round-trip: Kingsrow's own move was legal on our board in 40/40 positions
#: under this mapping and 0/40 under the alternatives.
_PIECE = {1: WHITE | MAN, 3: WHITE | KING, 2: BLACK | MAN, 4: BLACK | KING}


def _bit_to_square(bit):
    """Our 0..63 bit index -> the 1..32 square number Kingsrow prints.

    The same transform `_fill` applies to the pieces, read back the other way:
    standard notation numbers the dark squares in reading order, and after the
    mirror in x that `_fill` does, our `(x, y)` lands at `4y + x // 2`. The
    parity term is because our playable squares are `(x + y)` odd, so the
    leftmost playable x is 1 on even rows and 0 on odd ones.
    """
    x, y = bit % 8, bit // 8
    return 1 + 4 * y + (x - (1 if y % 2 == 0 else 0)) // 2


#: 1..32 <-> our bit indices. Not taken on trust: `python kingsrow.py` plants a
#: lone king on each square in turn and checks the square Kingsrow names in its
#: own PV, which passes 32/32 here and fails on every square under a wrong
#: rotation or a transposed axis.
BIT_TO_SQUARE = {b: _bit_to_square(b) for b in range(64) if (b % 8 + b // 8) % 2}
SQUARE_TO_BIT = {n: b for b, n in BIT_TO_SQUARE.items()}


def _squares(token):
    """`21-17` or `6x29` -> our bit indices, or None if a number is not a square.

    A capture token may be *compressed*, with the intermediate landing squares
    left out, so the result can be two squares for a chain of several hops. The
    caller reconstructs those against a move generator; this module knows
    nothing about our board rules.
    """
    nums = [int(s) for s in re.findall(r'\d+', token)]
    if any(n not in SQUARE_TO_BIT for n in nums):
        return None
    return tuple(SQUARE_TO_BIT[n] for n in nums)


class Kingsrow:
    """A Kingsrow instance bound to this process.

    `chdir` is process-global, so constructing this moves the whole process to
    the CheckerBoard directory. Everything in the nnue scripts uses absolute
    paths (`NNUE_DIR`), so that is safe there, but do not construct one in a
    process that resolves its own paths relatively.
    """

    def __init__(self, seconds=0.05, hashsize=64, cb_dir=None, book=False):
        self.cb_dir = cb_dir or DEFAULT_CB_DIR
        self.seconds = seconds
        engines = os.path.join(self.cb_dir, 'engines')
        dll_path = os.path.join(engines, 'Kingsrow64.dll')
        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"{dll_path} not found. Kingsrow is expected where "
                f"CheckerBoard installed it; pass cb_dir= if it is elsewhere.")

        # the cookies must outlive the load: dropping one removes the
        # directory from the search path again
        self._dll_dirs = [os.add_dll_directory(engines),
                          os.add_dll_directory(self.cb_dir)]  # egdb64.dll here
        os.chdir(self.cb_dir)                   # required, see module docstring

        dll = C.WinDLL(dll_path)
        dll.enginecommand.argtypes = [C.c_char_p, C.c_char_p]
        dll.enginecommand.restype = C.c_int
        dll.enginename.argtypes = [C.c_char_p]
        dll.getmove.argtypes = [Board8, C.c_int, C.c_double, C.c_char_p,
                                C.POINTER(C.c_int), C.c_int, C.c_int,
                                C.POINTER(_CBmove)]
        dll.getmove.restype = C.c_int
        self.dll = dll

        name = C.create_string_buffer(256)
        dll.enginename(name)
        self.name = name.value.decode(errors='replace')

        # book off: a book move is returned without a search and so without a
        # value, and book positions would all get the same label anyway
        for c in ("set protocolversion 2", f"set book {1 if book else 0}",
                  "set searchthreads 1", f"set hashsize {hashsize}"):
            self.command(c)

        self._str = C.create_string_buffer(4096)
        self._playnow = C.c_int(0)
        self._move = _CBmove()
        self._board = Board8()

    def command(self, text):
        reply = C.create_string_buffer(4096)
        self.dll.enginecommand(text.encode(), reply)
        return reply.value.decode(errors='replace')

    # -- searching ---------------------------------------------------------
    def _fill(self, board):
        b = self._board
        for x in range(8):
            for y in range(8):
                b[x][y] = OCCUPIED
        for y in range(8):
            for x in range(8):
                if (x + y) % 2:                 # playable for us
                    b[7 - x][y] = _PIECE.get(board[y][x], FREE)
        return b

    def search(self, board, player, seconds=None):
        """(kingsrow's own absolute value, status string) for one position."""
        b = self._fill(board)
        # zero the buffer: a call that writes nothing would otherwise leave the
        # previous position's value to be parsed as this one's
        C.memset(self._str, 0, 4096)
        self.dll.getmove(b, WHITE if player == 1 else BLACK,
                         C.c_double(self.seconds if seconds is None
                                    else seconds),
                         self._str, C.byref(self._playnow), 0, 0,
                         C.byref(self._move))
        s = self._str.value.decode(errors='replace')
        m = _VALUE_RE.search(s)
        return (int(m.group(1)) if m else None), s

    def eval(self, board, player, seconds=None):
        """Side-to-move relative centipawns, or None if the engine gave none.

        Kingsrow reports absolutely, positive favouring player 2, so the sign
        is flipped when player 1 is to move.
        """
        value, _ = self.search(board, player, seconds)
        if value is None:
            return None
        return value if player == 2 else -value

    def bestmove(self, board, player, seconds=None):
        """`(cp, squares, is_capture)` from one search: the score and the move.

        `cp` is side-to-move relative, as in `eval()`. `squares` are our 0..63
        bit indices, and there are more than two of them when Kingsrow wrote out
        the intermediate landing squares of a capture chain. Either may be None
        independently - the score and the move are parsed out of the same status
        string and it does not always carry both:

          * A position Kingsrow answers from its endgame databases reports
            `depth 6; 21-17* (0.250), 21-25 (1.000),` - candidate moves with win
            probabilities, no `value=` and no PV. Which one it would play is not
            stated, so this returns no move rather than guessing at one.
          * A capture is written *compressed*, `6x29`, so the caller has to
            reconstruct the hops against a move generator. That is deliberately
            not done here: this module knows nothing about our board rules.
        """
        value, status = self.search(board, player, seconds)
        cp = None if value is None else (value if player == 2 else -value)

        m = _PV_RE.search(status)
        if m is None:
            return cp, None, False
        squares = _squares(m.group(1))
        if squares is None:
            return cp, None, False
        return cp, squares, 'x' in m.group(1)

    def playmove(self, board, player, seconds=None):
        """`(squares, is_capture, source)`: the move to play here, one search.

        `bestmove` is the entry point when the score is what you want, and it
        deliberately names no move on a position Kingsrow answers out of its
        endgame databases. This one is for actually *playing* Kingsrow, where
        that ply still has to produce a move and it must be Kingsrow's own -
        handing those plies to another engine would no longer be a match
        against Kingsrow.

        A database position gets a candidate list in place of a value and a PV:

            depth 6; 19-16* (0.394), 5-9* (0.417), 5-1* (0.417), 19-15 (0.608),

        Probed over 200 such positions (all of them drawn - a decisive database
        position comes back with an ordinary PV, so this path only ever chooses
        between moves that hold the same result): the list is sorted ascending
        by the parenthesised number, the starred moves are always a prefix of
        it, and the first starred move was legal on our board 200 times out of
        200. What the number counts is undocumented and is not needed here - the
        star is Kingsrow's own mark for the moves it rates best, so this takes
        the first of them.

        `squares` is None, with `source` None, only when neither reading
        produced a move; `source` is 'pv' or 'db' otherwise. As in `bestmove`,
        a capture may be compressed and is the caller's to reconstruct.
        """
        _value, status = self.search(board, player, seconds)

        m = _PV_RE.search(status)
        if m is not None:
            squares = _squares(m.group(1))
            if squares is not None:
                return squares, 'x' in m.group(1), 'pv'

        for frm, sep, to, star in _CAND_RE.findall(status):
            if not star:
                break           # the stars are a prefix; past them is worse
            squares = _squares(f"{frm}{sep}{to}")
            if squares is not None:
                return squares, sep == 'x', 'db'
        return None, False, None


if __name__ == '__main__':
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(1, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..')))
    import bitboard_converter as bc                            # noqa: E402
    import openings                                            # noqa: E402

    kr = Kingsrow()
    print(f"engine: {kr.name}")

    board = openings.get_11man_boards()[0]
    for player in (1, 2):
        # one search, both readings of it - calling eval() here would search a
        # second time and print a value that does not match `raw`
        v, s = kr.search(board, player)
        print(f"  p{player}: raw={v} stm_relative={v if player == 2 else -v}")
        print(f"      {s[:100]}")

    # a lopsided position must read as a rout for the side with the pieces
    lop = [row[:] for row in board]
    gone = 0
    for y in range(8):
        for x in range(8):
            if lop[y][x] == 2 and gone < 3:
                lop[y][x] = 0
                gone += 1
    print(f"\n  p1 up {gone} men:")
    for player in (1, 2):
        print(f"    p{player} to move -> {kr.eval(lop, player):+d} "
              f"(should be strongly {'+' if player == 1 else '-'})")

    # ---- the square mapping, checked against Kingsrow itself ----
    # A wrong rotation or a transposed axis does not make the labels subtly
    # worse, it makes the parsed move name a square with nothing on it. With one
    # king of ours on the board and one of theirs far away, the only move
    # Kingsrow can name is our king's, so the square it prints must be the
    # square we put it on.
    #
    # Through `playmove` rather than `bestmove`, because two kings is deep
    # inside Kingsrow's endgame databases and a database position is answered
    # with a candidate list and no PV - which is exactly the reading `bestmove`
    # declines to make. It is the same square mapping either way.
    print("\n  square mapping (a lone king on each of the 32 squares):")
    playable = [b for b in range(64) if (b % 8 + b // 8) % 2]
    bad = []
    for bit in playable:
        probe = [[0] * 8 for _ in range(8)]
        probe[bit // 8][bit % 8] = 3                    # our player 1 king
        for other in playable:                          # theirs, out of reach
            if abs(other % 8 - bit % 8) > 3 or abs(other // 8 - bit // 8) > 3:
                probe[other // 8][other % 8] = 4
                break
        squares, _cap, _src = kr.playmove(probe, 1, seconds=0.02)
        if squares is None or squares[0] != bit:
            bad.append((bit, None if squares is None else squares[0]))
    if bad:
        print(f"    FAIL on {len(bad)}/{len(playable)}: {bad[:8]}")
    else:
        print(f"    {len(playable)}/{len(playable)} OK")

    t = time.perf_counter()
    for _ in range(50):
        kr.eval(board, 1)
    el = time.perf_counter() - t
    print(f"\n  {50 / el:.1f} pos/s ({el / 50 * 1000:.1f} ms each) "
          f"at seconds={kr.seconds}")
