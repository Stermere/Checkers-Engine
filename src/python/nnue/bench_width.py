"""What does a wider first layer actually cost the search?

The point of making inference fast is to be able to afford a bigger network, so
the useful number is not "the eval got N times faster" but "how much node rate
does L1 = 512 cost, compared to L1 = 128". That question can be answered before
training anything: the *speed* of the forward pass depends only on the shape of
the weights, not on their values. So this builds the engine against synthetic
headers of each width, measures nodes per second, and throws them away.

The numbers are a speed budget, nothing else. The nets it builds play at random,
and their evaluations are meaningless - only the time they take is real.

    python src/python/nnue/bench_width.py --widths 128 256 512 1024

It swaps src/engine/nnue_weights.h while it runs and restores it at the end,
including on Ctrl-C. If it is killed hard enough to skip that, the backup path is
printed at startup, and any `python src/python/nnue/export.py --out
src/engine/nnue_weights.h` regenerates the real one from a checkpoint anyway.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import engine_iface  # noqa: E402

REPO_ROOT = engine_iface.REPO_ROOT
HEADER = os.path.join(REPO_ROOT, 'src', 'engine', 'nnue_weights.h')
BACKUP = HEADER + '.real'

NUM_FEATURES = 120
MAX_ACTIVE = 24
QUANT_ACT = 127
QUANT_W = 64
EVAL_SCALE = 120.0


def synthetic_header(path, l1, l2, seed=0):
    """A header of the right shape with plausible but meaningless weights.

    The magnitudes match the trained net (|ft| ~ 170, |hidden| <= 127, which is
    the HIDDEN_WEIGHT_CLIP bound in model.py), so the accumulator bound and the
    clipped ReLU saturation behave realistically. Anything else about them is
    noise, on purpose.
    """
    rng = np.random.default_rng(seed)
    ft_w = np.clip(rng.normal(0, 20, size=(NUM_FEATURES + 1, l1)), -174, 174).astype(np.int16)
    ft_w[NUM_FEATURES] = 0  # the padding row must stay zero
    ft_b = np.clip(rng.normal(0, 20, size=l1), -174, 174).astype(np.int16)
    h_w = np.clip(rng.normal(0, 30, size=(l2, l1)), -127, 127).astype(np.int16)
    h_b = rng.integers(-2000, 2000, size=l2).astype(np.int32)
    o_w = np.clip(rng.normal(0, 30, size=l2), -127, 127).astype(np.int16)
    o_b = np.array([0], dtype=np.int32)

    top = np.sort(np.abs(ft_w.astype(np.int64)), axis=0)[-(MAX_ACTIVE + 4):].sum(axis=0)
    acc_abs_max = int((top + np.abs(ft_b.astype(np.int64))).max())
    if acc_abs_max > 32767:
        raise ValueError(f"synthetic weights overflow the int16 accumulator ({acc_abs_max})")

    def carr(name, arr, ctype):
        flat = np.asarray(arr).reshape(-1)
        out = [f"NNUE_ALIGN64 static const {ctype} {name}[{flat.size}] = {{"]
        for s in range(0, flat.size, 16):
            out.append("    " + ", ".join(str(int(v)) for v in flat[s:s + 16]) + ",")
        out.append("};")
        return "\n".join(out)

    with open(path, 'w') as f:
        f.write("/* SYNTHETIC weights written by bench_width.py - NOT a trained net.\n"
                " * Only the shape is meaningful. Regenerate the real header with\n"
                " * src/python/nnue/export.py before playing a single game.\n */\n\n")
        f.write("#ifndef NNUE_WEIGHTS_H\n#define NNUE_WEIGHTS_H\n\n#include <stdint.h>\n\n")
        f.write(f"#define NNUE_NUM_FEATURES {NUM_FEATURES}\n")
        f.write(f"#define NNUE_MAX_ACTIVE {MAX_ACTIVE}\n")
        f.write(f"#define NNUE_L1 {l1}\n#define NNUE_L2 {l2}\n")
        f.write(f"#define NNUE_QUANT_ACT {QUANT_ACT}\n#define NNUE_QUANT_W {QUANT_W}\n")
        f.write(f"#define NNUE_EVAL_SCALE {EVAL_SCALE:.1f}f\n")
        f.write(f"#define NNUE_ACC_ABS_MAX {acc_abs_max}\n\n")
        f.write("#if defined(_MSC_VER)\n  #define NNUE_ALIGN64 __declspec(align(64))\n"
                "#elif defined(__GNUC__) || defined(__clang__)\n"
                "  #define NNUE_ALIGN64 __attribute__((aligned(64)))\n"
                "#else\n  #define NNUE_ALIGN64\n#endif\n\n")
        f.write(carr("nnue_ft_weight", ft_w, "int16_t") + "\n\n")
        f.write(carr("nnue_ft_bias", ft_b, "int16_t") + "\n\n")
        f.write(carr("nnue_h_weight", h_w, "int16_t") + "\n\n")
        f.write(carr("nnue_h_bias", h_b, "int32_t") + "\n\n")
        f.write(carr("nnue_o_weight", o_w, "int16_t") + "\n\n")
        f.write(carr("nnue_o_bias", o_b, "int32_t") + "\n\n")
        f.write("#endif /* NNUE_WEIGHTS_H */\n")


def build(name, defines, python):
    cmd = [python, os.path.join(REPO_ROOT, 'src', 'python', 'Package_engine.py'),
           'build', '--force', '--name', name]
    for d in defines:
        cmd += ['--define', d]
    # the LNK1104 that Dropbox causes by grabbing the .pyd mid-sync is transient
    # and cannot damage the previous file, so retrying is the whole fix
    for attempt in range(6):
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            return True
        if 'LNK1104' not in r.stdout + r.stderr:
            print(r.stdout[-3000:])
            print(r.stderr[-2000:])
            return False
        time.sleep(3)
    print(f"  {name}: the linker could not write the .pyd after 6 tries")
    return False


def measure(names, depth, positions, python):
    cmd = [python, os.path.join(HERE, 'bench_eval_speed.py'),
           '--depth', str(depth), '--positions', str(positions), '--modules'] + names
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0] in names and parts[-1] == 'nodes/s':
            out[parts[0]] = float(parts[-2].rstrip('M'))
    if not out:
        print(r.stdout[-2000:], r.stderr[-1000:])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--widths', type=int, nargs='+', default=[128, 256, 512, 1024])
    ap.add_argument('--l2', type=int, default=16)
    ap.add_argument('--depth', type=int, default=11)
    ap.add_argument('--positions', type=int, default=16)
    args = ap.parse_args()

    python = sys.executable
    print(f"backing up {os.path.relpath(HEADER, REPO_ROOT)} -> "
          f"{os.path.relpath(BACKUP, REPO_ROOT)}\n")
    shutil.copyfile(HEADER, BACKUP)

    rows = []
    try:
        for l1 in args.widths:
            synthetic_header(HEADER, l1, args.l2)
            ok = (build('search_engine_w', [], python)
                  and build('search_engine_wnr', ['NNUE_INCREMENTAL=0'], python))
            if not ok:
                print(f"L1={l1}: build failed, skipping")
                continue
            nps = measure(['search_engine_w', 'search_engine_wnr'],
                          args.depth, args.positions, python)
            inc = nps.get('search_engine_w', 0.0)
            ref = nps.get('search_engine_wnr', 0.0)
            rows.append((l1, inc, ref))
            print(f"  L1={l1:<5} incremental {inc:5.2f}M nodes/s   "
                  f"from scratch {ref:5.2f}M nodes/s")
    finally:
        shutil.copyfile(BACKUP, HEADER)
        os.remove(BACKUP)
        print(f"\nrestored {os.path.relpath(HEADER, REPO_ROOT)}")

    if rows:
        base_inc = rows[0][1]
        base_ref = rows[0][2]
        print(f"\n  {'L1':>6}  {'incremental':>12}  {'from scratch':>13}  "
              f"{'vs L1=' + str(rows[0][0]):>12}")
        for l1, inc, ref in rows:
            rel = inc / base_inc if base_inc else 0.0
            print(f"  {l1:>6}  {inc:>9.2f} M/s  {ref:>10.2f} M/s  {rel:>11.2f}x")
        print("\n  the last column is what a wider net costs in search speed;\n"
              "  compare the two middle columns to see whether incremental\n"
              "  updates are worth keeping at that width")
    return 0


if __name__ == '__main__':
    sys.exit(main())
