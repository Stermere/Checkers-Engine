# Neural evaluation

A small NNUE-style evaluation network, sized so it can be implemented in the C
engine. This replaces the old `src/python/NeuralNet.py`, which could not learn
because of how it was set up (see "What was wrong before").

## Results so far

Trained on 4.77M positions with exact win/draw/loss labels from the endgame
tablebase, held-out 5% validation split:

| metric | value |
| --- | --- |
| majority-class baseline | 49.8% |
| **validation WDL accuracy** | **95.29%** |
| sign accuracy on decisive positions | 99.51% |
| after int16 quantization | 95.17% (-0.14%) |
| training time | 102 s on an RTX 3070 |
| parameters | 17,569 (34.6 KB as int16) |

Accuracy by piece count: 3 pieces 97.2%, 4 pieces 93.8%, 5 pieces 95.5%.

**This does not yet mean the engine will play better.** See "Status" below.

## Architecture

```
120 sparse binary inputs
  |   feature transformer (sum of active columns, not a matmul)
 128  clipped ReLU
  |
 16   clipped ReLU
  |
  1   win/loss logit;  centipawns = logit * 120
```

Design constraints came from the engine, not from accuracy:

- **Sparse binary first layer.** Only 2-24 inputs are ever active, so the first
  layer is a sum of at most 24 contiguous 128-wide int16 rows. That is roughly
  24x8 AVX2 vector adds, which is cheap enough to run at every leaf.
- **Clipped ReLU** (clamp to [0,1]) rather than plain ReLU, so activations are
  bounded and fixed-point conversion cannot overflow. `model.clip_weights()`
  enforces the weight bound during training, so the float net never drifts
  somewhere quantization cannot follow.
- **17k parameters**, small enough to bake into a generated C header.

## Input encoding (`features.py`)

120 features = 4 piece types x 32 squares, minus the 8 impossible
(man, back rank) combinations.

Everything is encoded **from the side to move's perspective**: when it is
player 2's turn the board is rotated 180 degrees and the colors are swapped.
The network therefore only ever answers "how good is this for me", and the
evaluation is antisymmetric by construction instead of having to learn that
symmetry from data. This halves the effective problem size.

`test_features.py` checks the square tables round-trip, the 180-degree mirror
is correct, all 120 (piece, square) pairs map to distinct inputs, the
vectorized encoder matches a scalar reference, and the perspective flip is
symmetric.

## Data (`gen_db_data.py`)

Rather than probing the engine one position at a time, this inverts the
tablebase index directly in numpy, so millions of exactly-labeled positions
come out in seconds. It decodes the 2-bit packed slice files in `db/` by
un-ranking the colex index of each piece group.

Two independent correctness checks, because a decoder that is merely
self-consistent can still be wrong:

1. `--verify` (on by default) re-encodes every decoded position and requires
   the original index back.
2. `verify_db_decode.py` hands decoded positions to the **C engine** and
   compares its verdict to the label. Currently **423/425 (99.5%)**; the two
   misses are the engine's repetition handling at the root, not decode errors.

## Layout

Everything for the network lives in this directory:

```
src/python/nnue/
  features.py           board -> feature indices
  test_features.py      encoding unit tests
  gen_db_data.py        tablebase sampler
  verify_db_decode.py   cross-check the sampler against the C engine
  gen_selfplay_data.py  midgame sampler: self-play labelled by the search
  engine_iface.py       locates and wraps the built C extension
  openings.py           the 11-man ballot set
  verify_eval_convention.py  proves the search eval is side-to-move relative
  model.py              the network
  train.py              training loop (WLD and centipawn labels, blended)
  export.py             int16 quantization + C header generation

  data/                 generated datasets      (gitignored)
  models/               trained checkpoints     (gitignored)
  nnue_weights.h        generated C header      (gitignored)
```

## Usage

```bash
python src/python/nnue/test_features.py                # unit tests
python src/python/nnue/gen_db_data.py --total 6000000  # ~30 s
python src/python/nnue/verify_db_decode.py             # cross-check vs engine
python src/python/nnue/gen_selfplay_data.py --games 17000 --workers 14
python src/python/nnue/train.py --epochs 60            # ~100 s on a 3070
python src/python/nnue/export.py                       # -> nnue_weights.h
```

## Training hyperparameters
```bash
python src/python/nnue/train.py --data src\python\nnue\data\selfplay_all.npz --out src/python/nnue/models/net.pt --l1 512 --l2 32 --epochs 75 --lr 0.002 --lambda 0.9 --piece-balance 0.7 --piece-cap 3
```

`train.py --data` takes several `.npz` files and works out each one's label
type (`wld` from the tablebase, `eval_cp` from self-play). `--weights` sets how
much each set pulls, independent of how many rows it has, and every set is
validated and reported separately:

```bash
python src/python/nnue/train.py \
    --data data/db_positions.npz data/selfplay_1m.npz --weights 1 2
```


Generated files stay in this directory rather than the repo root. At engine
integration time, point `export.py --out` at `src/engine/`.


## What was wrong before

`NeuralNet.py` was not a bad architecture so much as an unusable setup:

- **The input was never defined.** `INPUT_SIZE = ((32 + 28) * 2) + 1 = 121` with
  no encoder anywhere in the repo, so nothing could produce a training example.
- **No perspective normalization**, so the net would have had to learn the
  board symmetry from scratch, and evaluations would not have been
  antisymmetric - which a negamax search requires.
- **`torch.set_grad_enabled(False)` in `predict()` was never restored**, so
  calling `predict()` once silently disabled gradients process-wide and all
  subsequent training became a no-op.
- `test()` looped one position at a time, and `save()` used `torch.save(self)`
  (pickling the class, which breaks whenever the file moves).
- Plain ReLU with unbounded activations is not quantization friendly, so even a
  trained net could not have been ported to the engine.

## Status

- [x] Feature encoding, verified
- [x] Tablebase data pipeline, cross-checked against the engine
- [x] Architecture that learns (95.3% WDL vs 49.8% baseline)
- [x] int16 quantization, costs 0.14% accuracy
- [x] **Midgame training data** - not done
- [x] C implementation and integration - not done
