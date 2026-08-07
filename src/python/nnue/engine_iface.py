"""Thin wrapper around the compiled `search_engine` extension.

Everything that talks to the C engine goes through here so that the awkward
parts are solved exactly once:

  * **Finding the build.** `src/python/train_nn.py` hardcodes
    `lib.win-amd64-cpython-310`, which is stale - the extension currently in use
    is built for whichever interpreter you are running. We derive the directory
    from `sys.version_info` and fall back to scanning `build/` so this keeps
    working after the next Python upgrade.

  * **The return layout.** `search_position` returns a *list* of two tuples,
    built in `board_search.c`:

        [0] = (move_from, move_to)                       Py_BuildValue("ii")
        [1] = (depth, max_ply, nodes,
               hash_entries, eval, evals)                Py_BuildValue("iiLLiL")

    Existing callers index this as `results[-2]` and `results[-1][4]` (and also
    `results[1][4]`, which happens to be the same element because the list has
    length two). `SearchResult` gives the fields names so nothing has to rely on
    that coincidence.

    **Field 1 is max_ply, not extended_depth.** They are different numbers:
    extended_depth stops at the last node that still had depth budget left, while
    max_ply counts the capture-only quiescence past the horizon too - which in
    checkers is most of what "how deep did it look" means. The engine's own
    printed report shows depth / ext / max ply side by side; only max_ply is
    exported.

    **Fields are read by index, not unpacked.** This tuple has grown twice, and
    a positional unpack of a fixed arity turns every future addition into a
    ValueError in the data generation scripts - which is exactly what happened
    when `evals` was appended. New fields may only ever go on the END, and this
    reader takes a prefix of what it is given.

  * **The eval sign.** `search_info->eval` comes from `negmax` at the root, and
    `get_eval` flips the sign for player 2, so the score is **from the side to
    move's point of view**: positive means the player who is about to move is
    better. `verify_eval_convention.py` proves this empirically rather than
    taking this comment's word for it.

  * **Engine chatter.** The engine `printf`s progress at the C level, which
    `contextlib.redirect_stdout` cannot capture. We redirect fd 1/2 instead.
"""

import ctypes
import os
import re
import sys
from contextlib import contextmanager

MASK64 = (1 << 64) - 1


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
BUILD_ROOT = os.path.join(REPO_ROOT, 'build')


def _candidate_build_dirs():
    """Most-likely-correct build directories, best first."""
    major, minor = sys.version_info[:2]
    yield os.path.join(BUILD_ROOT, f'lib.win-amd64-cpython-{major}{minor}')
    yield os.path.join(BUILD_ROOT, f'lib.win-amd64-{major}.{minor}')

    # fall back to anything that looks like a build dir, newest version first
    if os.path.isdir(BUILD_ROOT):
        entries = []
        for name in os.listdir(BUILD_ROOT):
            m = re.match(r'lib\.[^.]+-(?:cpython-)?(\d)\.?(\d+)$', name)
            if m:
                entries.append((int(m.group(1)), int(m.group(2)), name))
        for _, _, name in sorted(entries, reverse=True):
            yield os.path.join(BUILD_ROOT, name)


def add_engine_to_path():
    """Put the directory holding search_engine*.pyd on sys.path."""
    for d in _candidate_build_dirs():
        if not os.path.isdir(d):
            continue
        if any(f.startswith('search_engine.') for f in os.listdir(d)):
            if d not in sys.path:
                sys.path.insert(0, d)
            return d
    raise ImportError(
        f"no built search_engine extension found under {BUILD_ROOT}. "
        f"Run build.bat with the interpreter you are using "
        f"(python {sys.version_info[0]}.{sys.version_info[1]}).")


add_engine_to_path()
import search_engine as se  # noqa: E402


# ---------------------------------------------------------------------------
# output suppression
# ---------------------------------------------------------------------------
def _crt_fflush_all():
    """fflush(NULL) in the C runtime the extension links against.

    Redirecting fd 1 is not enough on its own: the engine's printf output sits
    in the CRT's own stdout buffer (fully buffered when stdout is a pipe) and is
    flushed later, landing wherever fd 1 points *at that time*. Without this the
    engine's chatter reappears in the middle of a report. Python's sys.stdout is
    a separate io layer that writes to the fd directly, so flushing the CRT here
    cannot disturb it.
    """
    for dll in ('ucrtbase', 'api-ms-win-crt-stdio-l1-1-0', 'msvcrt'):
        try:
            ctypes.CDLL(dll).fflush(None)
            return True
        except (OSError, AttributeError):
            continue
    return False


@contextmanager
def suppress_engine_output(enabled=True):
    """Silence the engine's C-level printf output.

    Redirects the real file descriptors, because the writes happen inside the
    extension module and never pass through sys.stdout.
    """
    if not enabled:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # drain the engine's buffered output into devnull *before* fd 1 is the
        # real stdout again
        _crt_fflush_all()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)



# ---------------------------------------------------------------------------
# searching
# ---------------------------------------------------------------------------
class SearchResult:
    """Named view over the list `search_position` returns.

    Field order is dictated by the two Py_BuildValue calls in board_search.c;
    see the module docstring.
    """

    __slots__ = ('move_from', 'move_to', 'depth', 'max_ply',
                 'nodes', 'hash_entries', 'eval')

    # how many leading elements of raw[1] this class knows how to name
    _STATS_FIELDS = 5

    def __init__(self, raw):
        if len(raw) != 2:
            raise ValueError(f"unexpected search_position result shape: {raw!r}")
        (self.move_from, self.move_to) = raw[0]
        stats = raw[1]
        # a PREFIX, deliberately: the engine appends new stats to the end of this
        # tuple, and a positional unpack of a fixed arity would make every such
        # addition a ValueError here - which is how `evals` broke the data
        # generation scripts. Too few fields is still an error, since that means
        # the layout changed rather than grew.
        if len(stats) < self._STATS_FIELDS:
            raise ValueError(
                f"search_position stats tuple has {len(stats)} fields, "
                f"expected at least {self._STATS_FIELDS}: {raw!r}")
        (self.depth, self.max_ply, self.nodes,
         self.hash_entries, self.eval) = stats[:self._STATS_FIELDS]

    @property
    def move(self):
        """(from_square, to_square) as 0..63 bit indices."""
        return (self.move_from, self.move_to)

    def __repr__(self):
        return (f"SearchResult(move={self.move}, eval={self.eval}, "
                f"depth={self.depth}, max_ply={self.max_ply}, "
                f"nodes={self.nodes})")


def search(p1, p2, p1k, p2k, stm, seconds, depth, forced=-1, quiet=True):
    """Search one position. `eval` in the result is side-to-move relative."""
    with suppress_engine_output(quiet):
        raw = se.search_position(int(p1), int(p2), int(p1k), int(p2k),
                                 int(stm), float(seconds), int(depth),
                                 int(forced))
    return SearchResult(raw)


# ---------------------------------------------------------------------------
# bitboard helpers (mirrors of the C originals)
# ---------------------------------------------------------------------------
def mirror_bb(bb):
    """Rotate a single bitboard 180 degrees: bit b -> bit 63 - b."""
    bb &= MASK64
    out = 0
    while bb:
        low = bb & -bb
        out |= 1 << (63 - low.bit_length() + 1)
        bb ^= low
    return out


def mirror_position(p1, p2, p1k, p2k, stm):
    """Colour-swapped 180 degree mirror.

    The resulting position is the *same* position seen from the other chair:
    the pieces that were about to move are still about to move, they are just
    called player `3 - stm` now. A side-to-move relative evaluation must
    therefore be unchanged; a player-1 relative one must negate.
    """
    return (mirror_bb(p2), mirror_bb(p1), mirror_bb(p2k), mirror_bb(p1k),
            3 - stm)


# column masks that stop the diagonal shifts from wrapping around the edge
_JUMP_COL_GE2 = 0xFCFCFCFCFCFCFCFC  # column >= 2
_JUMP_COL_LE5 = 0x3F3F3F3F3F3F3F3F  # column <= 5


def has_any_jump(p1, p2, p1k, p2k, player):
    """Does `player` have a capture available? Port of has_any_jump in board_eval.c.

    Used to drop tactical positions from the training set: the static eval of a
    position with a capture pending is meaningless, because the position on the
    board is not the position that will actually be reached.
    """
    occupied = (p1 | p2 | p1k | p2k) & MASK64
    empty = ~occupied & MASK64

    if player == 1:
        up, down, enemy = (p1 | p1k) & MASK64, p1k & MASK64, (p2 | p2k) & MASK64
    else:
        up, down, enemy = p2k & MASK64, (p2 | p2k) & MASK64, (p1 | p1k) & MASK64

    if up & _JUMP_COL_GE2 & ((enemy << 9) & MASK64) & ((empty << 18) & MASK64):
        return True
    if up & _JUMP_COL_LE5 & ((enemy << 7) & MASK64) & ((empty << 14) & MASK64):
        return True
    if down & _JUMP_COL_GE2 & (enemy >> 7) & (empty >> 14):
        return True
    if down & _JUMP_COL_LE5 & (enemy >> 9) & (empty >> 18):
        return True
    return False


def popcount(*bbs):
    return sum(int(b).bit_count() for b in bbs)
