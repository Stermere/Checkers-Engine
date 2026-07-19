"""Sample labeled positions straight out of the endgame tablebase.

Phase 1 training data. The point of using the tablebase is that the labels are
*exact*: there is no search noise and no "every position in this game gets the
game's result" correlation. If the network cannot fit this, the problem is the
model or the encoding, not the labels.

The slice files are 2-bit packed arrays indexed by the colex rank of each piece
group (see src/engine/endgame_db.c). Rather than probing the engine one
position at a time we invert that index directly in numpy, which lets us pull
millions of labeled positions in seconds.

Run:  python src/python/nnue/gen_db_data.py --total 4000000
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F

DB_DRAW, DB_WIN, DB_LOSS, DB_BROKEN = 0, 1, 2, 3

NNUE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(NNUE_DIR, '..', '..', '..'))
DEFAULT_DB_DIR = os.path.join(REPO_ROOT, 'db')          # the engine's tablebase
DEFAULT_OUT = os.path.join(NNUE_DIR, 'data', 'db_positions.npz')



# ---------------------------------------------------------------------------
# binomial table, matching db_init_binomials() in the engine
# ---------------------------------------------------------------------------
def build_binomials(max_n=33, max_k=7):
    c = np.zeros((max_n, max_k), dtype=np.int64)
    for n in range(max_n):
        c[n, 0] = 1
        for k in range(1, max_k):
            if n > 0:
                c[n, k] = c[n - 1, k - 1] + c[n - 1, k]
    return c


C = build_binomials()


def slice_size(m1, k1, m2, k2):
    """Number of indices in a slice (men have 28 legal squares, kings 32)."""
    return int(C[28, m1] * C[32, k1] * C[28, m2] * C[32, k2] * 2)


# ---------------------------------------------------------------------------
# colex rank <-> square list
# ---------------------------------------------------------------------------
def unrank_group(r, count, num_squares):
    """Invert db_rank_group: colex rank -> (N, count) increasing square indices.

    Greedy from the top: the largest square is the largest s with C[s][i] <= r.
    """
    n = r.shape[0]
    out = np.zeros((n, count), dtype=np.int64)
    r = r.copy()
    for i in range(count, 0, -1):
        col = C[:num_squares, i]           # increasing in s
        s = np.searchsorted(col, r, side='right') - 1
        out[:, i - 1] = s
        r -= col[s]
    return out


def rank_group(sq):
    """db_rank_group: (N, count) increasing squares -> colex rank."""
    if sq.shape[1] == 0:
        return np.zeros(sq.shape[0], dtype=np.int64)
    idx = np.arange(1, sq.shape[1] + 1)
    return C[sq, idx].sum(axis=1)


def decode_indices(idx, m1, k1, m2, k2):
    """Slice-local index -> (p1, p2, p1k, p2k, stm) bitboards.

    Exactly inverts db_encode().
    """
    idx = np.asarray(idx, dtype=np.int64)
    stm = (idx & 1).astype(np.uint8) + 1
    rest = idx >> 1

    n_k2, n_m2, n_k1 = int(C[32, k2]), int(C[28, m2]), int(C[32, k1])
    r_k2, rest = rest % n_k2, rest // n_k2
    r_m2, rest = rest % n_m2, rest // n_m2
    r_k1, rest = rest % n_k1, rest // n_k1
    r_m1 = rest

    g_m1 = unrank_group(r_m1, m1, 28)
    g_k1 = unrank_group(r_k1, k1, 32)
    g_m2 = unrank_group(r_m2, m2, 28)
    g_k2 = unrank_group(r_k2, k2, 32)

    def to_bb(groups, offset):
        bb = np.zeros(idx.shape[0], dtype=np.uint64)
        for j in range(groups.shape[1]):
            bits = F.SQ32_TO_BIT64[groups[:, j] + offset]
            bb |= np.uint64(1) << bits.astype(np.uint64)
        return bb

    p1 = to_bb(g_m1, 4)     # p1 men are stored with a 4 square offset
    p1k = to_bb(g_k1, 0)
    p2 = to_bb(g_m2, 0)
    p2k = to_bb(g_k2, 0)

    return p1, p2, p1k, p2k, stm, (g_m1, g_k1, g_m2, g_k2)


def encode_check(groups, stm, m1, k1, m2, k2):
    """Re-encode decoded groups back to a slice index (self-consistency check)."""
    g_m1, g_k1, g_m2, g_k2 = groups
    i = rank_group(g_m1)
    i = i * int(C[32, k1]) + rank_group(g_k1)
    i = i * int(C[28, m2]) + rank_group(g_m2)
    i = i * int(C[32, k2]) + rank_group(g_k2)
    return i * 2 + (stm.astype(np.int64) - 1)


# ---------------------------------------------------------------------------
# slice discovery and sampling
# ---------------------------------------------------------------------------
def discover_slices(db_dir):
    out = []
    pattern = re.compile(r'^wld_(\d)(\d)(\d)(\d)\.bin$')
    for name in sorted(os.listdir(db_dir)):
        m = pattern.match(name)
        if not m:
            continue
        m1, k1, m2, k2 = (int(g) for g in m.groups())
        size = slice_size(m1, k1, m2, k2)
        expect_bytes = (size + 3) // 4
        actual = os.path.getsize(os.path.join(db_dir, name))
        if actual != expect_bytes:
            print(f"  skipping {name}: size {actual} != expected {expect_bytes}")
            continue
        out.append((m1, k1, m2, k2, size, os.path.join(db_dir, name)))
    return out


def allocate(slices, total):
    """Split the sample budget across slices proportional to sqrt(size).

    Proportional-to-size would let the couple of huge slices (e.g. 2 men + 2
    kings vs 1 king) drown out everything else; sqrt keeps the small material
    configurations meaningfully represented.
    """
    weights = np.array([np.sqrt(s[4]) for s in slices])
    weights /= weights.sum()
    want = np.maximum(1, (weights * total).astype(np.int64))
    # never ask for more than a slice actually holds
    sizes = np.array([s[4] for s in slices], dtype=np.int64)
    return np.minimum(want, sizes)


def sample_slice(path, m1, k1, m2, k2, size, n_want, rng, verify):
    buf = np.memmap(path, dtype=np.uint8, mode='r')

    if n_want >= size:
        idx = np.arange(size, dtype=np.int64)
    else:
        # oversample then dedupe, cheaper than a full permutation for big slices
        draw = min(size, int(n_want * 1.2) + 64)
        idx = np.unique(rng.integers(0, size, size=draw, dtype=np.int64))[:n_want]

    values = (buf[idx >> 2] >> ((idx & 3) * 2).astype(np.uint8)) & 3
    keep = values != DB_BROKEN
    idx, values = idx[keep], values[keep]
    if idx.size == 0:
        return None

    p1, p2, p1k, p2k, stm, groups = decode_indices(idx, m1, k1, m2, k2)

    if verify:
        back = encode_check(groups, stm, m1, k1, m2, k2)
        if not np.array_equal(back, idx):
            bad = int((back != idx).sum())
            raise AssertionError(f"{os.path.basename(path)}: {bad} indices failed re-encode")

    return p1, p2, p1k, p2k, stm, values.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db-dir', default=DEFAULT_DB_DIR)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--total', type=int, default=4_000_000,
                    help='approximate number of positions to sample')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-verify', action='store_true',
                    help='skip the decode/re-encode consistency check')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    slices = discover_slices(args.db_dir)
    if not slices:
        print(f"no usable slice files found in {args.db_dir}")
        return 1
    total_available = sum(s[4] for s in slices)
    print(f"found {len(slices)} slices, {total_available:,} positions total")

    quota = allocate(slices, args.total)

    parts = []
    for (m1, k1, m2, k2, size, path), n_want in zip(slices, quota):
        got = sample_slice(path, m1, k1, m2, k2, size, int(n_want), rng,
                           verify=not args.no_verify)
        if got is not None:
            parts.append(got)

    p1 = np.concatenate([p[0] for p in parts])
    p2 = np.concatenate([p[1] for p in parts])
    p1k = np.concatenate([p[2] for p in parts])
    p2k = np.concatenate([p[3] for p in parts])
    stm = np.concatenate([p[4] for p in parts])
    wld = np.concatenate([p[5] for p in parts])

    # shuffle so that any later prefix slice is still representative
    order = rng.permutation(len(p1))
    p1, p2, p1k, p2k, stm, wld = (a[order] for a in (p1, p2, p1k, p2k, stm, wld))

    n_win = int((wld == DB_WIN).sum())
    n_loss = int((wld == DB_LOSS).sum())
    n_draw = int((wld == DB_DRAW).sum())
    n = len(wld)
    print(f"sampled {n:,} positions: "
          f"{n_win / n:.1%} win, {n_draw / n:.1%} draw, {n_loss / n:.1%} loss")

    # sanity: encoding must not choke on any of them
    step = max(1, n // 200_000)
    F.encode_indices(p1[::step], p2[::step], p1k[::step], p2k[::step], stm[::step],
                     validate=True)
    print("  encoding validated on a subsample")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, p1=p1, p2=p2, p1k=p1k, p2k=p2k, stm=stm, wld=wld)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
