"""Quantize the trained net to int16 and emit a C header.

The engine cannot use float32 weights in a hot eval loop, so the deployable
artifact is a fixed point version. This script does three things:

  1. Converts the weights to integers.
  2. Re-implements inference using *only* integer arithmetic, in numpy, exactly
     as the C code will do it - same shifts, same clamps, same intermediate
     widths. This is the reference the C implementation must match.
  3. Measures the quantized net's accuracy against the float net on the
     validation set, so we know what precision costs us before writing any C.

Fixed point layout:

  accumulator   int16, units of 1/127   (clipped ReLU output lands in [0,127])
  layer weights int16, units of 1/64
  a dot product of layer inputs (x*127) with weights (w*64) comes out in units
  of 1/(127*64), so dividing by 64 returns it to units of 1/127 ready for the
  next clipped ReLU.

Run:  python src/python/nnue/export.py
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F
from model import CheckersNet, EVAL_SCALE, QUANT_ACT, QUANT_W
from train import encode_all, DB_DRAW, DB_WIN, DB_LOSS


NNUE_DIR = os.path.dirname(os.path.abspath(__file__))
INT16_MAX = 32767



def quantize(model):
    """float weights -> integer weights, with range checks."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    ft_w = np.rint(sd['ft.weight'] * QUANT_ACT).astype(np.int32)
    ft_b = np.rint(sd['ft_bias'] * QUANT_ACT).astype(np.int32)
    h_w = np.rint(sd['hidden.weight'] * QUANT_W).astype(np.int32)
    h_b = np.rint(sd['hidden.bias'] * QUANT_ACT * QUANT_W).astype(np.int32)
    o_w = np.rint(sd['out.weight'] * QUANT_W).astype(np.int32)
    o_b = np.rint(sd['out.bias'] * QUANT_ACT * QUANT_W).astype(np.int32)

    # the accumulator is a sum of up to MAX_ACTIVE feature columns plus the
    # bias; make sure the worst case cannot overflow an int16.
    #
    # SLACK extra columns of headroom, because the C accumulator is now
    # incremental: nnue.c reaches a child position by subtracting the rows of
    # the pieces that left and adding the rows of the pieces that arrived, so a
    # partial sum can transiently hold a few features that no single legal
    # position has at once. A move touches at most 3 features per perspective,
    # so 4 is already generous.
    ACC_SLACK = 4
    top = np.sort(np.abs(ft_w), axis=0)[-(F.MAX_ACTIVE + ACC_SLACK):].sum(axis=0) + np.abs(ft_b)

    if top.max() > INT16_MAX:
        raise ValueError(f"accumulator can overflow int16 (worst case {top.max()})")

    for name, arr in (('ft', ft_w), ('hidden', h_w), ('out', o_w)):
        if np.abs(arr).max() > INT16_MAX:
            raise ValueError(f"{name} weights exceed int16 ({np.abs(arr).max()})")

    print(f"  accumulator worst case {top.max()} (int16 limit {INT16_MAX})")
    print(f"  |ft| max {np.abs(ft_w).max()}, |hidden| max {np.abs(h_w).max()}, "
          f"|out| max {np.abs(o_w).max()}")

    return dict(ft_w=ft_w.astype(np.int16), ft_b=ft_b.astype(np.int16),
                h_w=h_w.astype(np.int16), h_b=h_b.astype(np.int32),
                o_w=o_w.astype(np.int16), o_b=o_b.astype(np.int32),
                acc_abs_max=int(top.max()))


def int_forward(q, indices, eval_scale=EVAL_SCALE):
    """Integer-only inference, the reference implementation for the C port.

    indices: (B, MAX_ACTIVE) int array, PAD_INDEX for empty slots.
    Returns evaluation in centipawns as int32.

    `eval_scale` defaults to the module constant but should be passed from the
    checkpoint - see the note in main() about why they can differ.
    """
    ft_w, ft_b = q['ft_w'].astype(np.int32), q['ft_b'].astype(np.int32)

    # accumulator: gather and sum the active feature columns
    acc = ft_w[indices].sum(axis=1) + ft_b          # (B, L1), units 1/127
    acc = np.clip(acc, 0, QUANT_ACT)                # clipped ReLU

    # hidden layer
    h = acc @ q['h_w'].astype(np.int32).T + q['h_b']  # units 1/(127*64)
    # back to units of 1/127. Round rather than truncate: truncation biases
    # every activation downward by up to 1 unit, and those biases accumulate
    # into a systematic eval error. In C this is (x + 32) >> 6.
    h = (h + QUANT_W // 2) // QUANT_W
    h = np.clip(h, 0, QUANT_ACT)


    # output layer
    o = h @ q['o_w'].astype(np.int32).T + q['o_b']    # units 1/(127*64)
    o = o[:, 0]

    # logit -> centipawns
    return (o * eval_scale / (QUANT_ACT * QUANT_W)).astype(np.int32)


def wdl_from_cp(cp, eval_scale=EVAL_SCALE):
    prob = 1.0 / (1.0 + np.exp(-cp / eval_scale))
    out = np.full(len(cp), DB_DRAW, dtype=np.int64)
    out[prob > 2.0 / 3.0] = DB_WIN
    out[prob < 1.0 / 3.0] = DB_LOSS
    return out


def write_header(q, path, eval_scale=EVAL_SCALE):
    l1, l2 = q['h_w'].shape[1], q['h_w'].shape[0]

    def carr(name, arr, ctype):
        flat = arr.reshape(-1)
        # NNUE_ALIGN64: the C side walks these tables with 256 bit vector loads.
        # The loads themselves are unaligned-safe, but a 64 byte aligned base
        # keeps every feature row inside whole cache lines, which is free to ask
        # for and measurably cheaper to execute.
        lines = [f"NNUE_ALIGN64 static const {ctype} {name}[{flat.size}] = {{"]
        for s in range(0, flat.size, 16):
            lines.append("    " + ", ".join(str(int(v)) for v in flat[s:s + 16]) + ",")
        lines.append("};")
        return "\n".join(lines)

    with open(path, 'w') as f:
        f.write("/* Generated by src/python/nnue/export.py - do not edit.\n"
                " *\n"
                " * Fixed point evaluation network.\n"
                f" *   {F.NUM_FEATURES} sparse inputs -> {l1} -> {l2} -> 1\n"
                " *   accumulator and activations: int16, units of 1/%d\n"
                " *   layer weights: int16, units of 1/%d\n"
                " */\n\n" % (QUANT_ACT, QUANT_W))
        f.write("#ifndef NNUE_WEIGHTS_H\n#define NNUE_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NNUE_NUM_FEATURES {F.NUM_FEATURES}\n")
        f.write(f"#define NNUE_MAX_ACTIVE {F.MAX_ACTIVE}\n")
        f.write(f"#define NNUE_L1 {l1}\n")
        f.write(f"#define NNUE_L2 {l2}\n")
        f.write(f"#define NNUE_QUANT_ACT {QUANT_ACT}\n")
        f.write(f"#define NNUE_QUANT_W {QUANT_W}\n")
        f.write(f"#define NNUE_EVAL_SCALE {eval_scale:.1f}f\n")
        # The largest magnitude any accumulator lane can reach, over every
        # subset of features a position (or a partially updated position) can
        # have. nnue.c keeps the accumulator in int16 so that it can be added a
        # whole vector at a time; this is what makes that safe, and it is
        # emitted rather than assumed so a retrained net cannot silently
        # invalidate it. nnue.c #errors if it exceeds 32767.
        f.write(f"#define NNUE_ACC_ABS_MAX {q['acc_abs_max']}\n\n")
        f.write("#if defined(_MSC_VER)\n"
                "  #define NNUE_ALIGN64 __declspec(align(64))\n"
                "#elif defined(__GNUC__) || defined(__clang__)\n"
                "  #define NNUE_ALIGN64 __attribute__((aligned(64)))\n"
                "#else\n"
                "  #define NNUE_ALIGN64\n"
                "#endif\n\n")

        # store the transformer transposed: column-major by feature, so that
        # accumulating a feature is a contiguous 128-wide int16 add
        f.write("/* [feature][L1], one contiguous row per feature */\n")
        f.write(carr("nnue_ft_weight", q['ft_w'], "int16_t") + "\n\n")
        f.write(carr("nnue_ft_bias", q['ft_b'], "int16_t") + "\n\n")
        f.write("/* [L2][L1] row major */\n")
        f.write(carr("nnue_h_weight", q['h_w'], "int16_t") + "\n\n")
        f.write(carr("nnue_h_bias", q['h_b'], "int32_t") + "\n\n")
        f.write(carr("nnue_o_weight", q['o_w'], "int16_t") + "\n\n")
        f.write(carr("nnue_o_bias", q['o_b'], "int32_t") + "\n\n")
        f.write("#endif /* NNUE_WEIGHTS_H */\n")


def load_check_set(path, n):
    """First n positions of a dataset from either generator, encoded.

    Returns (indices, label, kind). The leading slice is what the trainer holds
    out, so this is a validation set, not training data.
    """
    d = np.load(path)
    sl = slice(0, n)
    idx = encode_all(d['p1'][sl], d['p2'][sl], d['p1k'][sl], d['p2k'][sl],
                     d['stm'][sl]).astype(np.int32)
    if 'wld' in d.files:
        return idx, d['wld'][sl].astype(np.int64), 'wld'
    return idx, d['eval_cp'][sl].astype(np.float64), 'eval_cp'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(NNUE_DIR, 'models', 'net.pt'))

    ap.add_argument('--data', default=os.path.join(NNUE_DIR, 'data', 'db_positions.npz'))
    # the header stays here until there is C code to include it; at
    # integration time point --out at src/engine/ instead
    ap.add_argument('--out', default=os.path.join(NNUE_DIR, 'nnue_weights.h'))

    ap.add_argument('--check-n', type=int, default=200_000)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, weights_only=False)
    model = CheckersNet(ckpt['l1'], ckpt['l2'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    eval_scale = float(ckpt.get('eval_scale', EVAL_SCALE))
    if eval_scale != EVAL_SCALE:
        print(f"  note: checkpoint was trained at EVAL_SCALE={eval_scale:g}, "
              f"model.py now says {EVAL_SCALE:g}. Using the checkpoint's.")
    # 'val_acc' is the phase-1 checkpoint format, 'val_loss' the current one
    if 'val_loss' in ckpt:
        print(f"loaded {args.ckpt} (val loss {ckpt['val_loss']:.4f}, "
              f"trained on {', '.join(ckpt.get('datasets', ['?']))})")
    else:
        print(f"loaded {args.ckpt} (float val WDL {ckpt['val_acc']:.3%})")

    print("quantizing:")
    q = quantize(model)

    # validate on the same positions the trainer held out (the leading slice)
    idx_np, label, kind = load_check_set(args.data, args.check_n)
    n = len(idx_np)

    with torch.no_grad():
        t = torch.from_numpy(idx_np)
        cp_float = (model(t) * eval_scale).numpy()
    cp_int = int_forward(q, idx_np, eval_scale)

    diff = np.abs(cp_float - cp_int)
    print(f"\nchecked {n:,} positions from {os.path.basename(args.data)} ({kind}):")

    # what quantization costs against the labels...
    if kind == 'wld':
        acc_float = (wdl_from_cp(cp_float, eval_scale) == label).mean()
        acc_int = (wdl_from_cp(cp_int, eval_scale) == label).mean()
        print(f"  float WDL accuracy      {acc_float:.3%}")
        print(f"  quantized WDL accuracy  {acc_int:.3%}  ({acc_int - acc_float:+.3%})")
    else:
        mae_float = np.abs(cp_float - label).mean()
        mae_int = np.abs(cp_int - label).mean()
        print(f"  float MAE               {mae_float:.1f} cp")
        print(f"  quantized MAE           {mae_int:.1f} cp  ({mae_int - mae_float:+.1f})")

    # ...and, label-independent, how far the integer net drifts from the float
    # one. This is the number the C port has to reproduce exactly.
    print(f"  eval difference: mean {diff.mean():.2f} cp, median "
          f"{np.median(diff):.2f} cp, max {diff.max():.2f} cp")
    print(f"  agreement on WDL bucket: "
          f"{(wdl_from_cp(cp_float, eval_scale) == wdl_from_cp(cp_int, eval_scale)).mean():.3%}")


    write_header(q, args.out, eval_scale)
    size = (q['ft_w'].nbytes + q['ft_b'].nbytes + q['h_w'].nbytes +
            q['h_b'].nbytes + q['o_w'].nbytes + q['o_b'].nbytes)
    print(f"\nwrote {args.out} ({size / 1024:.1f} KB of weights)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
