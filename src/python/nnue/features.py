"""Board -> network input encoding.

The encoding is deliberately simple so the C engine can reproduce it exactly:

  * Only the 32 dark squares matter, so everything is indexed by "sq32"
    (0..31, the same indexing the endgame tablebase uses):
        sq32 = (bit64 >> 3) * 4 + ((bit64 & 7) >> 1)

  * The board is always encoded **from the side to move's point of view**.
    When it is player 2's turn we rotate the board 180 degrees and swap the
    colors, so the network only ever sees "my men move toward row 0".
    Under a 180 degree rotation bit64 -> 63 - bit64, which on dark squares is
    exactly sq32 -> 31 - sq32.

    This buys three things:
      - the side-to-move input disappears,
      - every pattern only has to be learned once instead of once per color,
      - the eval is antisymmetric by construction, which is what negamax wants.

  * Men cannot stand on their own promotion row, so each man plane needs only
    28 slots instead of 32:
        own men   live on sq32 4..31  (row 0 would have promoted)  -> index s - 4
        enemy men live on sq32 0..27  (row 7 would have promoted)  -> index s

Feature layout (120 inputs total):

    [  0 ..  27]  own men      (28)
    [ 28 ..  59]  own kings    (32)
    [ 60 ..  87]  enemy men    (28)
    [ 88 .. 119]  enemy kings  (32)

At most 24 of the 120 inputs are ever set, which is why the first layer is
implemented as a sum of columns rather than a matrix multiply.
"""

import numpy as np

# ---------------------------------------------------------------------------
# layout constants (mirrored in the C implementation and in export.py)
# ---------------------------------------------------------------------------
NUM_SQ32 = 32
NUM_MAN_SQ = 28

OWN_MEN_OFFSET = 0
OWN_KING_OFFSET = OWN_MEN_OFFSET + NUM_MAN_SQ      # 28
ENEMY_MEN_OFFSET = OWN_KING_OFFSET + NUM_SQ32      # 60
ENEMY_KING_OFFSET = ENEMY_MEN_OFFSET + NUM_MAN_SQ  # 88

NUM_FEATURES = ENEMY_KING_OFFSET + NUM_SQ32        # 120
MAX_ACTIVE = 24                                    # 12 pieces per side
PAD_INDEX = NUM_FEATURES                           # embedding padding slot


# ---------------------------------------------------------------------------
# square index tables
# ---------------------------------------------------------------------------
def _build_sq_tables():
    """sq32 <-> bit64 lookup tables for the dark squares."""
    sq32_to_bit64 = np.zeros(NUM_SQ32, dtype=np.int64)
    for s in range(NUM_SQ32):
        row = s >> 2
        col = ((s & 3) << 1) + (0 if (row & 1) else 1)
        sq32_to_bit64[s] = row * 8 + col
    return sq32_to_bit64


SQ32_TO_BIT64 = _build_sq_tables()


def bit64_to_sq32(b):
    return (b >> 3) * 4 + ((b & 7) >> 1)


# byte-level bit reversal table, used to rotate a bitboard 180 degrees
_REV_BYTE = np.array(
    [int('{:08b}'.format(i)[::-1], 2) for i in range(256)], dtype=np.uint8
)


def mirror_bitboards(bb):
    """Rotate an array of bitboards 180 degrees (bit b -> bit 63 - b).

    Done as "reverse the bits in each byte, then reverse the byte order",
    which is the same thing as reversing all 64 bits.
    """
    bb = np.ascontiguousarray(bb, dtype=np.uint64)
    as_bytes = bb.view(np.uint8).reshape(-1, 8)
    flipped = _REV_BYTE[as_bytes][:, ::-1]
    return np.ascontiguousarray(flipped).view(np.uint64).reshape(bb.shape)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
def to_perspective(p1, p2, p1k, p2k, stm):
    """Return (own_men, own_kings, enemy_men, enemy_kings) as seen by the side
    to move, with "own" always advancing toward row 0.

    All arguments are numpy arrays; stm is 1 or 2.
    """
    p1 = np.asarray(p1, dtype=np.uint64)
    p2 = np.asarray(p2, dtype=np.uint64)
    p1k = np.asarray(p1k, dtype=np.uint64)
    p2k = np.asarray(p2k, dtype=np.uint64)
    stm = np.asarray(stm)

    flip = (stm == 2)

    own_men = np.where(flip, mirror_bitboards(p2), p1)
    own_kings = np.where(flip, mirror_bitboards(p2k), p1k)
    enemy_men = np.where(flip, mirror_bitboards(p1), p2)
    enemy_kings = np.where(flip, mirror_bitboards(p1k), p2k)

    return own_men, own_kings, enemy_men, enemy_kings


def _dense_planes(own_men, own_kings, enemy_men, enemy_kings):
    """(N, 120) uint8 occupancy matrix in feature order."""
    def sq32_bits(bb):
        # unpack to (N, 64) with column index == bit position, then keep dark squares
        as_bytes = np.ascontiguousarray(bb, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
        bits = np.unpackbits(as_bytes, axis=1, bitorder='little')
        return bits[:, SQ32_TO_BIT64]

    om = sq32_bits(own_men)[:, 4:32]      # own men never on sq32 0..3
    ok = sq32_bits(own_kings)
    em = sq32_bits(enemy_men)[:, 0:28]    # enemy men never on sq32 28..31
    ek = sq32_bits(enemy_kings)

    return np.concatenate([om, ok, em, ek], axis=1)


def encode_dense(p1, p2, p1k, p2k, stm):
    """(N, 120) uint8 dense feature matrix. Mostly useful for tests."""
    return _dense_planes(*to_perspective(p1, p2, p1k, p2k, stm))


def encode_indices(p1, p2, p1k, p2k, stm, validate=False):
    """(N, MAX_ACTIVE) int16 array of active feature indices, PAD_INDEX padded.

    This is the format the model consumes: an EmbeddingBag over these indices
    is exactly the sparse accumulator the C engine will compute.
    """
    own_men, own_kings, enemy_men, enemy_kings = to_perspective(p1, p2, p1k, p2k, stm)

    if validate:
        # a man on its own promotion row would already be a king
        om_bits = np.ascontiguousarray(own_men, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
        om_bits = np.unpackbits(om_bits, axis=1, bitorder='little')[:, SQ32_TO_BIT64]
        if om_bits[:, 0:4].any():
            raise ValueError("own man found on its promotion row")
        em_bits = np.ascontiguousarray(enemy_men, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
        em_bits = np.unpackbits(em_bits, axis=1, bitorder='little')[:, SQ32_TO_BIT64]
        if em_bits[:, 28:32].any():
            raise ValueError("enemy man found on its promotion row")

    dense = _dense_planes(own_men, own_kings, enemy_men, enemy_kings)
    return dense_to_indices(dense)


def dense_to_indices(dense):
    """(N, 120) occupancy -> (N, MAX_ACTIVE) padded index array."""
    n = dense.shape[0]
    counts = dense.sum(axis=1).astype(np.int64)
    if counts.max(initial=0) > MAX_ACTIVE:
        raise ValueError(f"position with {counts.max()} pieces exceeds MAX_ACTIVE={MAX_ACTIVE}")

    rows, cols = np.nonzero(dense)
    # position of each active feature within its own row
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    slot = np.arange(rows.shape[0], dtype=np.int64) - np.repeat(starts, counts)

    out = np.full((n, MAX_ACTIVE), PAD_INDEX, dtype=np.int16)
    out[rows, slot] = cols.astype(np.int16)
    return out


# ---------------------------------------------------------------------------
# reference implementation (slow, scalar) used to check the vectorized path
# ---------------------------------------------------------------------------
def encode_reference(p1, p2, p1k, p2k, stm):
    """Plain-Python encoding of a single position -> sorted list of indices."""
    def squares(bb):
        return [bit64_to_sq32(b) for b in range(64) if (bb >> b) & 1]

    if stm == 2:
        own_m, own_k = squares(p2), squares(p2k)
        enemy_m, enemy_k = squares(p1), squares(p1k)
        own_m = [31 - s for s in own_m]
        own_k = [31 - s for s in own_k]
        enemy_m = [31 - s for s in enemy_m]
        enemy_k = [31 - s for s in enemy_k]
    else:
        own_m, own_k = squares(p1), squares(p1k)
        enemy_m, enemy_k = squares(p2), squares(p2k)

    idx = []
    idx += [OWN_MEN_OFFSET + (s - 4) for s in own_m]
    idx += [OWN_KING_OFFSET + s for s in own_k]
    idx += [ENEMY_MEN_OFFSET + s for s in enemy_m]
    idx += [ENEMY_KING_OFFSET + s for s in enemy_k]
    return sorted(idx)


def indices_to_sorted_list(row):
    """Strip padding from one row of encode_indices output."""
    return sorted(int(v) for v in row if v != PAD_INDEX)
