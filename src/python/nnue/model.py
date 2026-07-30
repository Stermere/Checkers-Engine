"""The evaluation network.

Architecture (NNUE style), chosen so the C engine can run it cheaply:

    120 sparse binary inputs
      |  feature transformer: sum of the columns of the active features
      v
    L1 = 128  -> clipped ReLU
      |
    L2 = 16   -> clipped ReLU
      |
    1 output  (a win/loss logit; multiply by EVAL_SCALE for centipawns)

Why this shape:

  * The first layer has binary inputs, so it is not a matrix multiply at all -
    it is a sum of at most 24 columns of 128 int16s. On AVX2 that is ~24*8
    vector adds. This is the whole reason the net can live inside a search that
    evaluates millions of nodes per second, and it is why we do not need
    incremental accumulator updates to make v1 fast.

  * Clipped ReLU (clamp to [0, 1]) instead of plain ReLU. Activations are then
    bounded, which is what makes fixed-point quantization safe: [0,1] maps onto
    [0,127] with no risk of an int16 overflow partway through a dot product.

  * ~17k parameters total. Small enough to train in minutes, and small enough
    that the weights can be baked into a generated C header.

The forward pass here is float, but it is written to stay quantization
friendly, and clip_weights() keeps every weight inside the range the int16
export can represent.
"""

import torch
import torch.nn as nn

from features import NUM_FEATURES, PAD_INDEX

L1 = 128
L2 = 16

# The network output is a logit. Centipawns = logit * EVAL_SCALE, i.e. a
# position the net scores at +1.0 logit is about a 73% win.
#
# 400 was tried and reverted (2026-08-01): it fits the recorded game outcomes
# better and loses ~157 Elo, because the search is not scale free. Static evals
# are clamped into +-EVAL_MAX (1500) and TERMINATE_EARLY_THRESHOLD is 20
# absolute centipawns, so a wider scale saturates the eval in exactly the won
# positions - exporting a 120-trained net at 324 clamped 70% of evaluations.
# Rank any change to this by games; val_loss moves with the target and mae_cp
# was not enough to catch it either.
EVAL_SCALE = 120

# Fixed point scales used by export.py. Activations are in [0,1] and are stored
# as int16 in units of 1/QUANT_ACT. Weights are stored in units of 1/QUANT_W.
QUANT_ACT = 127
QUANT_W = 64
# int16 weights must stay representable: |w| * QUANT_W <= 32767
WEIGHT_CLIP = 32767 / QUANT_W  # ~511, but we clip much tighter below
HIDDEN_WEIGHT_CLIP = 127 / QUANT_W  # ~1.98, the usual NNUE bound


def clipped_relu(x):
    return torch.clamp(x, 0.0, 1.0)


class CheckersNet(nn.Module):
    def __init__(self, l1=L1, l2=L2):
        super().__init__()
        self.l1_size = l1
        self.l2_size = l2

        # feature transformer. padding_idx contributes nothing, which is how
        # the variable piece count is handled without any masking.
        self.ft = nn.EmbeddingBag(NUM_FEATURES + 1, l1, mode='sum',
                                  padding_idx=PAD_INDEX)
        self.ft_bias = nn.Parameter(torch.zeros(l1))

        self.hidden = nn.Linear(l1, l2)
        self.out = nn.Linear(l2, 1)

        self._init_weights()

    def _init_weights(self):
        # small feature weights: an accumulator is a sum of up to 24 of them,
        # so large init values would saturate the clipped ReLU immediately
        nn.init.uniform_(self.ft.weight, -0.05, 0.05)
        with torch.no_grad():
            self.ft.weight[PAD_INDEX].zero_()
        nn.init.zeros_(self.ft_bias)
        nn.init.xavier_uniform_(self.hidden.weight, gain=0.5)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.out.weight, gain=0.5)
        nn.init.zeros_(self.out.bias)

    def forward(self, indices):
        """indices: (B, MAX_ACTIVE) int tensor of active features, PAD padded.

        Returns (B,) logits from the side-to-move's point of view.
        """
        acc = self.ft(indices) + self.ft_bias
        x = clipped_relu(acc)
        x = clipped_relu(self.hidden(x))
        return self.out(x).squeeze(-1)

    def eval_cp(self, indices):
        """Evaluation in centipawn-like units, for comparing against the engine."""
        return self.forward(indices) * EVAL_SCALE

    @torch.no_grad()
    def clip_weights(self):
        """Keep weights inside the range the fixed point export can represent.

        Called after every optimizer step so that training never wanders into a
        region that quantization would have to distort.
        """
        self.hidden.weight.clamp_(-HIDDEN_WEIGHT_CLIP, HIDDEN_WEIGHT_CLIP)
        self.out.weight.clamp_(-HIDDEN_WEIGHT_CLIP, HIDDEN_WEIGHT_CLIP)
        # the padding feature must stay exactly zero or empty slots would
        # silently add a bias that depends on the piece count
        self.ft.weight[PAD_INDEX].zero_()

    def num_params(self):
        # the padding row is not a real parameter
        return sum(p.numel() for p in self.parameters()) - self.ft.weight.shape[1]


def describe(model):
    return (f"CheckersNet {NUM_FEATURES}->{model.l1_size}->{model.l2_size}->1, "
            f"{model.num_params():,} parameters")
