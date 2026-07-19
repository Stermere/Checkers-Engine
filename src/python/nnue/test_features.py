"""Checks on the feature encoding.

The encoding has to be reproduced bit-for-bit in C later, so it is worth
pinning down its properties now rather than debugging a silent mismatch
between the trainer and the engine.

Run:  python src/python/nnue/test_features.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F


def random_positions(n, rng):
    """Generate n legal-ish random positions (men kept off their promotion row)."""
    p1 = np.zeros(n, dtype=np.uint64)
    p2 = np.zeros(n, dtype=np.uint64)
    p1k = np.zeros(n, dtype=np.uint64)
    p2k = np.zeros(n, dtype=np.uint64)
    stm = rng.integers(1, 3, size=n).astype(np.uint8)

    for i in range(n):
        squares = rng.permutation(32)
        n1 = int(rng.integers(1, 7))
        n2 = int(rng.integers(1, 7))
        for s in squares[:n1]:
            b = np.uint64(1) << np.uint64(F.SQ32_TO_BIT64[s])
            # p1 men may not sit on sq32 0..3 (they would have promoted)
            if s >= 4 and rng.random() < 0.5:
                p1[i] |= b
            else:
                p1k[i] |= b
        for s in squares[n1:n1 + n2]:
            b = np.uint64(1) << np.uint64(F.SQ32_TO_BIT64[s])
            # p2 men may not sit on sq32 28..31
            if s < 28 and rng.random() < 0.5:
                p2[i] |= b
            else:
                p2k[i] |= b

    return p1, p2, p1k, p2k, stm


def test_sq_tables():
    for s in range(32):
        b = int(F.SQ32_TO_BIT64[s])
        assert F.bit64_to_sq32(b) == s, f"sq32 roundtrip failed at {s}"
        row, col = b // 8, b % 8
        assert (row + col) % 2 == 1, f"sq32 {s} -> bit {b} is not a dark square"
    print("  sq32 <-> bit64 tables consistent")


def test_mirror():
    rng = np.random.default_rng(1)
    bb = rng.integers(0, 2**63, size=1000, dtype=np.uint64)
    mirrored = F.mirror_bitboards(bb)
    # mirroring twice is the identity
    assert np.array_equal(F.mirror_bitboards(mirrored), bb)
    # and bit b really does land on bit 63-b
    for i in range(50):
        v = int(bb[i])
        expect = 0
        for b in range(64):
            if (v >> b) & 1:
                expect |= 1 << (63 - b)
        assert int(mirrored[i]) == expect, "mirror mismatch"
    print("  180 degree bitboard mirror correct")


def test_vectorized_matches_reference():
    rng = np.random.default_rng(2)
    p1, p2, p1k, p2k, stm = random_positions(500, rng)
    idx = F.encode_indices(p1, p2, p1k, p2k, stm, validate=True)

    for i in range(len(p1)):
        got = F.indices_to_sorted_list(idx[i])
        want = F.encode_reference(int(p1[i]), int(p2[i]), int(p1k[i]), int(p2k[i]), int(stm[i]))
        assert got == want, f"row {i}: {got} != {want}"
    print("  vectorized encoding matches scalar reference")


def test_perspective_symmetry():
    """A position with p1 to move and its color-swapped mirror with p2 to move
    are the same position from the mover's point of view, so they must produce
    identical features. This is what makes the eval antisymmetric."""
    rng = np.random.default_rng(3)
    p1, p2, p1k, p2k, _ = random_positions(500, rng)
    stm1 = np.ones(len(p1), dtype=np.uint8)
    stm2 = np.full(len(p1), 2, dtype=np.uint8)

    a = F.encode_indices(p1, p2, p1k, p2k, stm1)
    # swap colors and rotate the board: the same position seen from p2's side
    b = F.encode_indices(F.mirror_bitboards(p2), F.mirror_bitboards(p1),
                         F.mirror_bitboards(p2k), F.mirror_bitboards(p1k), stm2)
    assert np.array_equal(a, b), "perspective flip is not symmetric"
    print("  perspective flip is symmetric (eval will be antisymmetric)")


def test_feature_ranges():
    rng = np.random.default_rng(4)
    p1, p2, p1k, p2k, stm = random_positions(2000, rng)
    dense = F.encode_dense(p1, p2, p1k, p2k, stm)
    assert dense.shape[1] == F.NUM_FEATURES == 120
    idx = F.dense_to_indices(dense)

    active = idx[idx != F.PAD_INDEX]
    assert active.min() >= 0 and active.max() < F.NUM_FEATURES
    # feature count must equal piece count
    assert np.array_equal((idx != F.PAD_INDEX).sum(1), dense.sum(1))
    print(f"  feature indices in range, {F.NUM_FEATURES} inputs, "
          f"max {int(dense.sum(1).max())} active in sample")


def test_distinct_squares_distinct_features():
    """Every (piece type, square) pair must map to its own input."""
    seen = {}
    for s in range(4, 32):
        seen.setdefault(F.OWN_MEN_OFFSET + (s - 4), []).append(("own man", s))
    for s in range(32):
        seen.setdefault(F.OWN_KING_OFFSET + s, []).append(("own king", s))
    for s in range(28):
        seen.setdefault(F.ENEMY_MEN_OFFSET + s, []).append(("enemy man", s))
    for s in range(32):
        seen.setdefault(F.ENEMY_KING_OFFSET + s, []).append(("enemy king", s))

    collisions = {k: v for k, v in seen.items() if len(v) > 1}
    assert not collisions, f"feature index collisions: {collisions}"
    assert len(seen) == F.NUM_FEATURES
    print(f"  all {len(seen)} (piece, square) pairs map to distinct inputs")


if __name__ == '__main__':
    print("feature encoding tests")
    test_sq_tables()
    test_mirror()
    test_distinct_squares_distinct_features()
    test_vectorized_matches_reference()
    test_perspective_symmetry()
    test_feature_ranges()
    print("all feature tests passed")
