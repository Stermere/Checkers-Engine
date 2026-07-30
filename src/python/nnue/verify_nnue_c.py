"""Prove that the C network in src/engine/nnue.c is the same function as the
Python one in export.py.

A quantized net is only useful if the engine and the trainer agree on what it
says. "Close enough" is not good enough: a one centipawn disagreement is a
different evaluation function, the training signal stops describing the thing
that plays the games, and every later measurement is quietly wrong. So this
script requires **exact equality on every position it tests**.

It checks four things, in the order that makes a failure easiest to localize:

  1. dimensions          nnue_info() vs features.py / the generated header
  2. the weights         src/engine/nnue_weights.h vs quantize(checkpoint)
                         - catches a header that was never regenerated
  3. the encoder         nnue_features() vs features.encode_reference()
                         - an encoder bug shows up as an encoder bug
  4. the arithmetic      nnue_eval_many() vs export.int_forward()
                         - the whole pipeline, on real positions

Plus two properties that no amount of agreement with a reference would catch if
both sides shared the same misunderstanding: the eval must be antisymmetric
under a colour-swapped mirror, and the batch and single-position entry points
must return the same numbers.

    python src/python/nnue/verify_nnue_c.py
    python src/python/nnue/verify_nnue_c.py --data data/selfplay_1m.npz -n 500000
"""

import argparse
import importlib
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import features  # noqa: E402
import engine_iface  # noqa: E402

REPO_ROOT = engine_iface.REPO_ROOT
DEFAULT_HEADER = os.path.join(REPO_ROOT, 'src', 'engine', 'nnue_weights.h')
DEFAULT_DATA = os.path.join(HERE, 'data', 'db_positions.npz')
DEFAULT_CKPT = os.path.join(HERE, 'models', 'net_db.pt')


# ---------------------------------------------------------------------------
# the header the C was compiled from
# ---------------------------------------------------------------------------
def parse_header(path):
    """Read nnue_weights.h back into numpy arrays.

    The point of parsing the header rather than re-quantizing the checkpoint is
    that the header is what the compiler actually saw. If it is stale, the eval
    comparison would otherwise fail with a confusing "the arithmetic is wrong"
    instead of the true "you forgot to re-run export.py".
    """
    with open(path) as f:
        text = f.read()

    defines = {m.group(1): m.group(2)
               for m in re.finditer(r'#define\s+(NNUE_\w+)\s+([^\s/]+)', text)}

    arrays = {}
    for m in re.finditer(r'static const (int16_t|int32_t) (nnue_\w+)\[(\d+)\]\s*=\s*\{(.*?)\};',
                         text, re.DOTALL):
        dtype = np.int16 if m.group(1) == 'int16_t' else np.int32
        name = m.group(2)
        expected = int(m.group(3))
        # explicit integer scan rather than np.fromstring: the generated header
        # has a trailing comma and line breaks, which fromstring rejects
        values = np.array([int(v) for v in re.findall(r'-?\d+', m.group(4))],
                          dtype=np.int64)
        if values.size != expected:

            raise ValueError(f"{name}: header declares {expected} values, found {values.size}")
        arrays[name] = values.astype(dtype)

    l1 = int(defines['NNUE_L1'])
    l2 = int(defines['NNUE_L2'])
    n_feat = int(defines['NNUE_NUM_FEATURES'])

    # the transformer has one extra row: the padding slot encode_indices uses
    ft_rows = arrays['nnue_ft_weight'].size // l1
    if ft_rows not in (n_feat, n_feat + 1):
        raise ValueError(f"ft_weight has {ft_rows} rows, expected {n_feat} or {n_feat + 1}")

    q = dict(
        ft_w=arrays['nnue_ft_weight'].reshape(ft_rows, l1),
        ft_b=arrays['nnue_ft_bias'],
        h_w=arrays['nnue_h_weight'].reshape(l2, l1),
        h_b=arrays['nnue_h_bias'],
        o_w=arrays['nnue_o_weight'].reshape(1, l2),
        o_b=arrays['nnue_o_bias'],
    )
    return q, {k: v for k, v in defines.items()}


def compare_to_checkpoint(q_header, ckpt):
    """Is the header the current checkpoint, quantized? Returns a status string."""
    if not os.path.exists(ckpt):
        return f"SKIP  (no checkpoint at {os.path.relpath(ckpt, REPO_ROOT)})"
    try:
        import torch
        from model import CheckersNet
        import export
    except Exception as exc:  # torch missing, etc - not a reason to fail the port
        return f"SKIP  ({type(exc).__name__}: {exc})"

    # same checkpoint layout export.py's main() uses: dims live alongside the
    # weights so the net can be rebuilt without guessing its shape
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    net = CheckersNet(blob['l1'], blob['l2'])
    net.load_state_dict(blob['state_dict'])
    net.eval()
    q_ckpt = export.quantize(net)


    for name in ('ft_w', 'ft_b', 'h_w', 'h_b', 'o_w', 'o_b'):
        a = np.asarray(q_ckpt[name]).reshape(-1).astype(np.int64)
        b = np.asarray(q_header[name]).reshape(-1).astype(np.int64)
        if a.size != b.size or not np.array_equal(a, b):
            return (f"FAIL  {name} differs from quantize({os.path.basename(ckpt)}) "
                    f"- regenerate with export.py --out src/engine/nnue_weights.h")
    return "OK    header matches quantize(checkpoint)"


# ---------------------------------------------------------------------------
# the reference forward pass, run on the header's weights
# ---------------------------------------------------------------------------
def int_forward(q, indices, eval_scale):
    """export.int_forward, restated against header-parsed weights.

    Kept in sync with export.py by test 4 below being the thing that fails if it
    ever drifts; it is repeated here only so this script does not need torch.

    `eval_scale` is a parameter and not a constant on purpose: it was hardcoded
    at 120.0, which silently stopped being the trained value when model.py
    recalibrated it to 400. On a machine without torch - the only way this copy
    is ever reached - that made every evaluation come back at 30% of its true
    magnitude, and the C would have been blamed for a difference that was
    entirely this function's.
    """
    quant_act = 127
    quant_w = 64

    ft_w = q['ft_w'].astype(np.int32)
    acc = ft_w[indices].sum(axis=1) + q['ft_b'].astype(np.int32)
    acc = np.clip(acc, 0, quant_act)

    h = acc @ q['h_w'].astype(np.int32).T + q['h_b'].astype(np.int32)
    h = (h + quant_w // 2) // quant_w
    h = np.clip(h, 0, quant_act)

    o = h @ q['o_w'].astype(np.int32).T + q['o_b'].astype(np.int32)
    o = o[:, 0]
    return (o * eval_scale / (quant_act * quant_w)).astype(np.int32)


def pick_reference():
    """Prefer the real export.int_forward; fall back to the copy above.

    Using export.py itself is the stronger test - it is the code the training
    pipeline trusts - but it pulls in torch through `model`, and the port should
    still be verifiable on a machine that only has numpy. parse_header produces
    exactly the dict export.int_forward expects, so the two are interchangeable.
    """
    try:
        import export
        return export.int_forward, 'export.int_forward'
    except Exception as exc:
        return int_forward, f'local copy ({type(exc).__name__} importing export)'



# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_dimensions(engine, defines):
    info = engine.nnue_info()
    expected = {
        'num_features': features.NUM_FEATURES,
        'max_active': features.MAX_ACTIVE,
        'l1': int(defines['NNUE_L1']),
        'l2': int(defines['NNUE_L2']),
        'quant_act': int(defines['NNUE_QUANT_ACT']),
        'quant_w': int(defines['NNUE_QUANT_W']),
    }
    bad = {k: (info[k], v) for k, v in expected.items() if info[k] != v}
    if bad:
        for k, (got, want) in bad.items():
            print(f"  {k}: engine says {got}, expected {want}")
        return False
    print(f"  L1={info['l1']} L2={info['l2']} features={info['num_features']} "
          f"USE_NNUE={info['use_nnue']}")
    return True


def check_encoder(engine, boards, n):
    """C encoder vs the plain-Python reference encoder."""
    p1, p2, p1k, p2k, stm = boards
    bad = 0
    for i in range(n):
        got = sorted(engine.nnue_features(int(p1[i]), int(p2[i]), int(p1k[i]),
                                          int(p2k[i]), int(stm[i])))
        want = features.encode_reference(int(p1[i]), int(p2[i]), int(p1k[i]),
                                         int(p2k[i]), int(stm[i]))
        if got != want:
            bad += 1
            if bad <= 3:
                print(f"  position {i}: C {got}")
                print(f"  position {i}: py {want}")
    print(f"  {n:,} positions, {bad} mismatches")
    return bad == 0


def eval_many(engine, p1, p2, p1k, p2k, stm):
    raw = engine.nnue_eval_many(np.ascontiguousarray(p1, dtype=np.uint64).tobytes(),
                                np.ascontiguousarray(p2, dtype=np.uint64).tobytes(),
                                np.ascontiguousarray(p1k, dtype=np.uint64).tobytes(),
                                np.ascontiguousarray(p2k, dtype=np.uint64).tobytes(),
                                np.ascontiguousarray(stm, dtype=np.uint8).tobytes())
    return np.frombuffer(raw, dtype=np.int32)


def check_eval(engine, q, boards, n, reference, eval_scale, chunk=100_000):
    """The whole forward pass, C vs Python, on n real positions."""
    p1, p2, p1k, p2k, stm = boards
    total_bad = 0
    worst = 0
    elapsed = 0.0
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        idx = features.encode_indices(p1[sl], p2[sl], p1k[sl], p2k[sl], stm[sl])
        want = reference(q, idx.astype(np.int64), eval_scale)


        t0 = time.perf_counter()
        got = eval_many(engine, p1[sl], p2[sl], p1k[sl], p2k[sl], stm[sl])
        elapsed += time.perf_counter() - t0

        diff = got.astype(np.int64) - want.astype(np.int64)
        bad = np.nonzero(diff)[0]
        total_bad += bad.size
        if bad.size:
            worst = max(worst, int(np.abs(diff).max()))
            for j in bad[:3]:
                i = start + int(j)
                print(f"  position {i}: C {got[j]} vs python {want[j]} "
                      f"(p1={p1[i]} p2={p2[i]} p1k={p1k[i]} p2k={p2k[i]} stm={stm[i]})")

    rate = n / elapsed if elapsed else float('inf')
    print(f"  {n:,} positions, {total_bad} mismatches"
          + (f", worst {worst} cp" if total_bad else ""))
    print(f"  C throughput {rate / 1e6:.2f}M evals/s ({1e9 / rate:.0f} ns per eval)")
    return total_bad == 0


def check_antisymmetry(engine, boards, n):
    """A colour-swapped mirror is the same position with the other side to move,
    so it must encode identically and evaluate identically. This is the property
    that makes get_eval's "do not negate the net output" correct."""
    p1, p2, p1k, p2k, stm = (b[:n] for b in boards)
    direct = eval_many(engine, p1, p2, p1k, p2k, stm)

    m_p1 = features.mirror_bitboards(p2)
    m_p2 = features.mirror_bitboards(p1)
    m_p1k = features.mirror_bitboards(p2k)
    m_p2k = features.mirror_bitboards(p1k)
    m_stm = np.where(stm == 1, 2, 1).astype(np.uint8)
    mirrored = eval_many(engine, m_p1, m_p2, m_p1k, m_p2k, m_stm)

    bad = int(np.count_nonzero(direct != mirrored))
    print(f"  {n:,} positions, {bad} mismatches")
    return bad == 0


def check_single_matches_batch(engine, boards, n):
    p1, p2, p1k, p2k, stm = boards
    batch = eval_many(engine, p1[:n], p2[:n], p1k[:n], p2k[:n], stm[:n])
    bad = 0
    for i in range(n):
        one = engine.nnue_eval(int(p1[i]), int(p2[i]), int(p1k[i]), int(p2k[i]), int(stm[i]))
        if one != batch[i]:
            bad += 1
            if bad <= 3:
                print(f"  position {i}: single {one} vs batch {batch[i]}")
    print(f"  {n:,} positions, {bad} mismatches")
    return bad == 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=DEFAULT_DATA, help='positions to test on (.npz)')
    ap.add_argument('--header', default=DEFAULT_HEADER, help='generated weights header')
    ap.add_argument('--ckpt', default=DEFAULT_CKPT, help='checkpoint the header should match')
    ap.add_argument('-n', type=int, default=200_000, help='positions for the eval check')
    ap.add_argument('--encoder-n', type=int, default=5_000,
                    help='positions for the (slow, scalar) encoder check')
    ap.add_argument('--module', default='search_engine',
                    help='extension module to test (Package_engine.py --name '
                         'builds variants that can be verified without '
                         'disturbing the engine in use)')
    args = ap.parse_args()

    engine_iface.add_engine_to_path()
    engine = importlib.import_module(args.module)

    if not hasattr(engine, 'nnue_eval'):
        print("the compiled engine has no nnue hooks - rebuild with build.bat")
        return 1

    data = np.load(args.data if os.path.isabs(args.data) else os.path.join(HERE, args.data))
    total = data['p1'].shape[0]
    n = min(args.n, total)
    # spread the sample over the whole file: the first N rows of a generated
    # dataset are correlated (one game, or one tablebase slice)
    step = max(1, total // n)
    sel = np.arange(0, total, step)[:n]
    boards = tuple(np.ascontiguousarray(data[k][sel]) for k in ('p1', 'p2', 'p1k', 'p2k', 'stm'))
    n = boards[0].shape[0]

    q, defines = parse_header(args.header)

    reference, ref_name = pick_reference()

    print(f"data   {os.path.relpath(args.data, REPO_ROOT) if os.path.isabs(args.data) else args.data}"
          f"  ({total:,} positions, sampling {n:,})")
    print(f"header {os.path.relpath(args.header, REPO_ROOT)}")
    print(f"ref    {ref_name}\n")


    results = []

    print("1. dimensions")
    results.append(('dimensions', check_dimensions(engine, defines)))

    print("\n2. header vs checkpoint")
    status = compare_to_checkpoint(q, args.ckpt)
    print(f"  {status}")
    results.append(('header vs checkpoint', not status.startswith('FAIL')))

    print("\n3. encoder (C vs features.encode_reference)")
    results.append(('encoder', check_encoder(engine, boards, min(args.encoder_n, n))))

    print("\n4. evaluation (C vs int_forward)")
    # the header's scale, not model.py's: a net trained at an older calibration
    # can still be the one deployed, and the reference has to be the function
    # the engine was actually compiled from
    eval_scale = float(defines['NNUE_EVAL_SCALE'].rstrip('f'))
    results.append(('evaluation',
                    check_eval(engine, q, boards, n, reference, eval_scale)))


    print("\n5. antisymmetry under colour-swapped mirror")
    results.append(('antisymmetry', check_antisymmetry(engine, boards, min(50_000, n))))

    print("\n6. single-position API vs batch API")
    results.append(('single vs batch', check_single_matches_batch(engine, boards, min(2_000, n))))

    print()
    ok = all(passed for _, passed in results)
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("\nthe C network is bit exact" if ok else "\nMISMATCH - do not enable USE_NNUE")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
