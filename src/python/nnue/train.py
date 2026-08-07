"""Train the evaluation network.

Two kinds of label, and the point of this file is that they train the same net:

  * **Tablebase WLD** (`gen_db_data.py`, key `wld`): exact win/draw/loss for
    positions with <=5 pieces. Target probability 1.0 / 0.5 / 0.0.
  * **Self-play centipawns** (`gen_selfplay_data.py`, key `eval_cp`): the
    search's own score for a midgame position. Target probability
    `sigmoid(eval_cp / EVAL_SCALE)`.

Both are win probabilities, both are side-to-move relative, and both train
under the same `BCEWithLogitsLoss` against the same logit output - the only
difference is how the target is derived and how the result is reported. So the
two sets can simply be concatenated.

**Game results.** The midgame generators also record how the games through each
position finished (`result`, `result_n`), and `--lambda` mixes that into the
target:

    target = lam * sigmoid(eval_cp / EVAL_SCALE) + (1 - lam) * result

The two labels fail in opposite directions, which is the point of having both.
The search score is precise but only ever as good as the teacher, so distilling
it alone caps the net at the teacher's understanding and reproduces its
mistakes faithfully. The game result is the ground truth the score is trying to
predict, but it is one sample of a noisy process - a won position thrown away
still records a loss. Blending keeps the score's resolution and lets the result
correct its systematic errors.

`result_n` is how many games back the result, which matters because
`gen_pdn_data.py` deduplicates its pool across games: a popular opening position
carries the average over hundreds of games, a deep midgame position over one.
`result_n == 0` means no outcome is known, and those rows keep the pure search
target at every lambda rather than being pulled toward 0.5.

**Blending.** The sets are very different sizes (4.8M tablebase vs ~1M
self-play), so plain concatenation would let the endgame set dominate by 5:1
and the net would go on being an endgame specialist. `--weights` sets the
relative *aggregate* pull of each set, defaulting to equal regardless of size;
per-sample weights are then normalised to mean 1 so the effective learning rate
does not move when you change the mix.

**Reporting.** Every set gets its own validation split and its own numbers -
a mix reported as one aggregate can hide the endgame set falling apart while
the (larger, easier) midgame set carries the average. Checkpoint selection uses
the combined weighted validation loss, which is the one quantity comparable
across both label types.

Metrics per set, by label type:
    wld      3-class bucket accuracy, sign accuracy on decisive positions,
             mean |p - target| in probability
    eval_cp  MAE in centipawns, sign agreement (on |cp| >= 25, since the sign
             of a 3 cp eval is noise), mean |p - target| in probability

The whole dataset lives on the GPU as an int32 index tensor, so there is no
dataloader and no host/device copy in the training loop.

**Starting from a saved net.** `--init` loads the weights out of an existing
checkpoint instead of initialising randomly, which is what makes staged
training possible: train on the tablebase set, then `--init` that net and train
on midgame data. Only the weights carry over - the optimizer state and the LR
schedule start fresh, because the second stage is a new run over different data
and wants its own cycle, not the tail of the first one's. It is usually worth a
lower `--lr` than the pretraining stage used.

**Stopping early.** Ctrl-C stops after the current batch, validates, and lets
that partial epoch take the checkpoint like any other before returning
normally - so a run can be cut short without losing it, and picked up again
with `--init`. A second Ctrl-C aborts immediately. Note that a net stopped
mid-schedule is still at a high learning rate and is not a converged net; it is
a starting point, not a result.

Run:
    python src/python/nnue/train.py --data data/db_positions.npz
    python src/python/nnue/train.py \
        --data data/db_positions.npz data/selfplay_1m.npz --weights 1 2

    # pretrain on the endgame, then fine-tune on the midgame
    python src/python/nnue/train.py --data data/db_positions.npz \
        --out models/endgame.pt
    python src/python/nnue/train.py --data data/selfplay_1m.npz \
        --init models/endgame.pt --lr 5e-4 --out models/net.pt
"""

import argparse
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as F
from model import CheckersNet, EVAL_SCALE, describe, L1, L2

NNUE_DIR = os.path.dirname(os.path.abspath(__file__))


DB_DRAW, DB_WIN, DB_LOSS = 0, 1, 2
# target win probability for each tablebase verdict
WLD_TO_PROB = np.array([0.5, 1.0, 0.0], dtype=np.float32)

# below this the sign of a search score is noise, not an opinion
SIGN_DEADZONE_CP = 25.0


def popcount(bb):
    b = np.ascontiguousarray(bb, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
    return np.unpackbits(b, axis=1).sum(axis=1).astype(np.int8)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    name: str
    kind: str           # 'wld' or 'eval_cp'
    idx: torch.Tensor   # (N, MAX_ACTIVE) int32 feature indices
    target: torch.Tensor  # (N,) float target probability from the label
    label: torch.Tensor   # (N,) raw label: wld class, or centipawns
    pieces: np.ndarray    # (N,) piece count, host side
    weight: float = 1.0   # per-sample loss weight
    outcome: torch.Tensor | None = None
    has_outcome: torch.Tensor | None = None
    #: the scale `target` was derived under, so target_for() knows when the
    #: cached column is already the right one
    eval_scale: float = EVAL_SCALE

    def __len__(self):
        return len(self.idx)

    def target_for(self, eval_scale=None):
        """The label as a win probability, squashed at `eval_scale`.

        Recomputed from the raw centipawn label rather than cached, so that a
        search can vary the scale across trials without reloading 17M rows. It
        is one sigmoid over the label column, which is nothing next to an epoch.

        WLD labels are exact win/draw/loss and carry no scale, so they are
        returned unchanged - which is the point: the scale changes what the
        *teacher's score* is taken to mean, not what a tablebase verdict means.
        """
        if (eval_scale is None or self.kind == 'wld'
                or eval_scale == self.eval_scale):
            return self.target
        return torch.sigmoid(self.label / eval_scale)

    def blended_target(self, lam, eval_scale=None):
        """The target the loss actually sees, at this lambda.

        `lam` weights the teacher's search score against the game result:
        1.0 is the search alone (what every dataset before outcomes existed
        trained on), 0.0 is the result alone. Rows with no known outcome keep
        the search target whatever `lam` is - they are not evidence about the
        result, and blending them toward 0.5 would teach the net that an
        unlabelled position is drawn.
        """
        target = self.target_for(eval_scale)
        if self.outcome is None or lam >= 1.0:
            return target
        blended = lam * target + (1.0 - lam) * self.outcome
        return torch.where(self.has_outcome, blended, target)


def encode_all(p1, p2, p1k, p2k, stm, chunk=500_000):
    n = len(p1)
    idx = np.empty((n, F.MAX_ACTIVE), dtype=np.int16)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        idx[s:e] = F.encode_indices(p1[s:e], p2[s:e], p1k[s:e], p2k[s:e], stm[s:e])
    return idx


def load_dataset(path, device, limit=0, min_pieces=0, log=print, frac=0.0):
    """Read an .npz from either generator and encode it onto the device.

    `min_pieces` drops positions at or below the piece count the endgame
    tablebase owns. `board_search.c` probes the database *before* `get_eval`, so
    at those piece counts the net's output is computed and then thrown away -
    training on them spends capacity on positions the engine never asks the net
    about. `DB_MAX_TOTAL` in `endgame_db.c` is 6, so `--min-pieces 7` is the
    setting that matches the engine; on `selfplay_large_kr` that is 18% of the
    rows. Filtering here rather than at split time also saves encoding and
    device memory for rows that would never be trained on.

    Applied *after* `limit`, which stays "the first N rows of the file", so the
    two compose predictably rather than `limit` silently meaning something
    different once a filter is on.
    """
    d = np.load(path)

    # `frac` is `limit` expressed proportionally, which is what a hyperparameter
    # search wants: one flag that shrinks every set by the same factor and so
    # leaves the relative sizes - and with them what a blend weight means -
    # exactly where they were. A flat `--limit 500000` across a 8.5M set and a
    # 4M one silently reweights the mix instead.
    #
    # `stm` is read to get the row count because it is one byte per row where
    # the boards are eight, so this costs a fraction of what loading p1 does.
    # Both generators shuffle, so a prefix is a random sample.
    if frac and 0.0 < frac < 1.0:
        n_total = int(d['stm'].shape[0])
        limit = max(1, int(n_total * frac))

    sl = slice(0, limit) if limit else slice(None)
    p1, p2, p1k, p2k, stm = (d['p1'][sl], d['p2'][sl], d['p1k'][sl],
                             d['p2k'][sl], d['stm'][sl])

    keep = None
    if min_pieces > 0:
        pc = popcount(p1) + popcount(p2) + popcount(p1k) + popcount(p2k)
        keep = pc >= min_pieces
        if not keep.any():
            # An all-tablebase set (db_positions.npz is every row <= 5 pieces)
            # empties completely here. Silently training on nothing, or on a
            # zero-length set the blend still reserves pull for, is worse than
            # saying so: the fix is a decision about --data, not about this flag.
            raise SystemExit(
                f"{os.path.basename(path)} has no positions with >= "
                f"{min_pieces} pieces, so --min-pieces {min_pieces} empties it.\n"
                f"Drop it from --data (or give it weight 0) if that is what you "
                f"meant, or lower --min-pieces.")
        if not keep.all():
            log(f"  {os.path.basename(path):<28} -{(~keep).sum():,} rows below "
                f"{min_pieces} pieces ({(~keep).mean():.1%})")
            p1, p2, p1k, p2k, stm = (p1[keep], p2[keep], p1k[keep],
                                     p2k[keep], stm[keep])


    def column(name):
        """One label column, sliced and filtered the same way the boards were."""
        col = d[name][sl]
        return col if keep is None else col[keep]

    if 'wld' in d.files:
        kind = 'wld'
        wld = column('wld').astype(np.int64)
        target = WLD_TO_PROB[wld]
        label = torch.from_numpy(wld).to(device)
    elif 'eval_cp' in d.files:
        kind = 'eval_cp'
        cp = column('eval_cp').astype(np.float32)

        # the same squashing the engine's eval implies: EVAL_SCALE centipawns
        # is one logit, so this is exactly the inverse of model.eval_cp()
        target = 1.0 / (1.0 + np.exp(-cp / EVAL_SCALE))
        label = torch.from_numpy(cp).to(device)
    else:
        raise ValueError(f"{path}: no 'wld' or 'eval_cp' array "
                         f"(found {d.files})")

    # how the games through each position finished. `result_n` is the number of
    # games behind `result` (several, for the deduplicated PDN pool), and 0
    # means no outcome is known - which is not the same as a draw, so it is
    # carried as a mask rather than folded into the value.
    outcome = has_outcome = None
    if 'result' in d.files and 'result_n' in d.files:
        res = column('result').astype(np.float32)
        res_n = column('result_n')
        outcome = torch.from_numpy(np.ascontiguousarray(res)).to(device)
        has_outcome = torch.from_numpy(np.ascontiguousarray(res_n > 0)).to(device)

    idx = encode_all(p1, p2, p1k, p2k, stm)
    pieces = popcount(p1) + popcount(p2) + popcount(p1k) + popcount(p2k)

    return Dataset(
        name=os.path.basename(path), kind=kind,
        idx=torch.from_numpy(idx.astype(np.int32)).to(device),
        target=torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)).to(device),
        label=label, pieces=pieces,
        outcome=outcome, has_outcome=has_outcome, eval_scale=EVAL_SCALE)


def split(ds, n_val):
    """Prefix split. Both generators shuffle, so a prefix split is random."""
    def part(s):
        return Dataset(ds.name, ds.kind, ds.idx[s], ds.target[s], ds.label[s],
                       ds.pieces[s], ds.weight,
                       None if ds.outcome is None else ds.outcome[s],
                       None if ds.has_outcome is None else ds.has_outcome[s],
                       ds.eval_scale)
    return part(slice(n_val, None)), part(slice(0, n_val))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def wld_metrics(logits, wld):
    prob = torch.sigmoid(logits)
    pred = torch.full_like(wld, DB_DRAW)
    pred[prob > 2.0 / 3.0] = DB_WIN
    pred[prob < 1.0 / 3.0] = DB_LOSS

    acc = (pred == wld).float().mean().item()

    target = torch.empty_like(prob)
    target[wld == DB_WIN] = 1.0
    target[wld == DB_LOSS] = 0.0
    target[wld == DB_DRAW] = 0.5
    mae = (prob - target).abs().mean().item()

    decisive = wld != DB_DRAW
    if decisive.any():
        want_win = wld[decisive] == DB_WIN
        got_win = prob[decisive] > 0.5
        sign_acc = (want_win == got_win).float().mean().item()
    else:
        sign_acc = float('nan')

    return {'acc': acc, 'sign': sign_acc, 'mae_prob': mae}


def cp_metrics(logits, cp, eval_scale=EVAL_SCALE):
    pred_cp = logits * eval_scale
    mae_cp = (pred_cp - cp).abs().mean().item()

    prob = torch.sigmoid(logits)
    mae_prob = (prob - torch.sigmoid(cp / eval_scale)).abs().mean().item()

    decisive = cp.abs() >= SIGN_DEADZONE_CP
    if decisive.any():
        sign_acc = ((pred_cp[decisive] > 0) == (cp[decisive] > 0)).float().mean().item()
    else:
        sign_acc = float('nan')

    return {'mae_cp': mae_cp, 'sign': sign_acc, 'mae_prob': mae_prob}


def metrics_for(ds, logits, eval_scale=None):
    if ds.kind == 'wld':
        return wld_metrics(logits, ds.label)
    return cp_metrics(logits, ds.label, eval_scale or ds.eval_scale)


def fmt_metrics(kind, m):
    if kind == 'wld':
        return f"wdl {m['acc']:.3%}  decisive {m['sign']:.3%}  mae {m['mae_prob']:.4f}"
    return f"mae {m['mae_cp']:6.1f}cp  sign {m['sign']:.3%}  mae_p {m['mae_prob']:.4f}"


@torch.no_grad()
def infer(model, idx, batch=131072):
    model.eval()
    logits = torch.empty(len(idx), device=idx.device)
    for s in range(0, len(idx), batch):
        logits[s:s + batch] = model(idx[s:s + batch])
    model.train()
    return logits


@torch.no_grad()
def evaluate(model, val_sets, loss_fn, lam=1.0, eval_scale=None):
    """Per-set metrics and per-set loss sums.

    The sums are returned rather than a single combined number because the
    weighting used to combine them is a choice: training uses the blend
    weights, but a hyperparameter search has to score every trial under one
    fixed weighting or a trial that merely down-weights the harder set looks
    like an improvement. See combine_loss().

    `lam` is the same choice one level down. Scoring against the target the run
    is training on is what checkpoint selection wants, since otherwise the best
    epoch is chosen by a quantity nothing is optimising; scoring every trial at
    one fixed lambda is what a search wants, since otherwise a trial can improve
    its number by moving its own target.
    """
    out = []
    for ds in val_sets:
        logits = infer(model, ds.idx)
        losses = loss_fn(logits, ds.blended_target(lam, eval_scale))
        out.append((ds, logits, metrics_for(ds, logits, eval_scale),
                    losses.sum().item()))
    return out


def combine_loss(results, sample_weights):
    """Weighted mean of per-sample validation loss across sets.

    `sample_weights[i]` is the per-sample weight for the i'th set - the same
    quantity as `Dataset.weight`, passed explicitly so the caller can score
    against a weighting other than the one being trained on.
    """
    num = den = 0.0
    for (ds, _, _, loss_sum), w in zip(results, sample_weights):
        num += loss_sum * w
        den += len(ds) * w
    return num / max(den, 1e-9)


def sample_weights(sets, req):
    """Turn per-set *aggregate* weights into per-sample weights.

    A weight is "how much aggregate pull this set has", so the per-sample
    weight is that divided by the set size; without the division the bigger set
    wins by default and the mix ratio is a lie. Normalised to mean 1 so the
    effective learning rate does not shift with the mix.
    """
    total_n = sum(len(d) for d in sets)
    scale = total_n / sum(req)
    return [w * scale / len(ds) for ds, w in zip(sets, req)]


def piece_balance_weights(pieces, alpha, cap):
    """Per-sample weights that flatten the piece-count distribution, mean 1.

    The generated sets are wildly uneven in piece count, and not in the
    direction the engine needs. Kingsrow-vs-Kingsrow games run to the 200-ply
    cap, so most plies of a game are endgame manoeuvring: measured on
    `selfplay_large_kr`, 18% of positions are at <= 6 pieces - where the
    tablebase answers before `get_eval` is ever called, so the net's opinion is
    discarded - 53% are at 7-10, and only 29% are at the 11+ piece counts where
    a learned eval decides games.

    `alpha` is how far to correct that: 0 leaves the natural distribution
    alone, 1 gives every piece count equal aggregate pull, and values between
    interpolate geometrically. Weight is `(1 / share(pieces)) ** alpha`,
    renormalised to mean 1 so the *set's* aggregate pull, and with it the blend
    ratio and the effective learning rate, are unchanged.

    `cap` is not optional at alpha near 1. Inverse frequency alone would hand a
    22-piece bucket holding a few hundred positions thousands of times the pull
    of an 8-piece bucket holding eighty thousand, and the net would fit the
    handful of rare rows and nothing else. The cap bounds the ratio between the
    largest and smallest weight, applied before the renormalisation.
    """
    if alpha <= 0.0:
        return np.ones(len(pieces), dtype=np.float32)

    counts = np.bincount(pieces.astype(np.int64))
    share = counts / max(counts.sum(), 1)
    with np.errstate(divide='ignore'):
        w = np.where(share > 0, share, 1.0) ** (-alpha)
    # bound the spread before normalising, so `cap` means the same thing
    # whatever the distribution underneath it looks like
    seen = counts > 0
    w[~seen] = 1.0
    lo = w[seen].min()
    w = np.minimum(w, lo * cap)
    out = w[pieces.astype(np.int64)].astype(np.float32)
    return out / out.mean()


# ---------------------------------------------------------------------------
# one training run
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Everything a single training run is allowed to vary.

    Kept as a dataclass rather than reading argparse directly so that
    grid_search.py can drive the same loop without re-parsing a command line -
    and, more importantly, without reloading and re-encoding the datasets for
    every trial.
    """
    out: str
    #: checkpoint to take the starting weights from, None for a random init.
    #: Weights only - see load_init().
    init: str | None = None
    epochs: int = 30
    batch: int = 8192
    lr: float = 2e-3
    val_frac: float = 0.05
    seed: int = 0
    l1: int = L1
    l2: int = L2
    weight_decay: float = 1e-6
    pct_start: float = 0.15
    weights: list | None = None  # per-set aggregate blend weights, None = equal
    # search score vs game result in the target: 1.0 is the score alone, 0.0 the
    # result alone. See Dataset.blended_target.
    lam: float = 1.0
    # how far to flatten each set's piece-count distribution; 0 is off.
    # See piece_balance_weights.
    piece_balance: float = 0.0
    piece_cap: float = 8.0
    # centipawns per logit. Two distinct jobs are wearing one number here, and
    # it is worth knowing which is which before sweeping it:
    #   * here, in training, it decides how hard sigmoid(cp / scale) squashes
    #     the teacher's score, i.e. how much of the range the net is given any
    #     gradient to distinguish. Changing it changes what is learned.
    #   * at export, it decides how a logit becomes centipawns for the engine.
    #     The search is not scale free - EVAL_MAX clamps static evals at 1500
    #     and TERMINATE_EARLY_THRESHOLD is 20 absolute centipawns - so this end
    #     of it is a real strength knob that costs a build and a match, with no
    #     retraining at all (match_net.py --eval-scale).
    # They are separable and need not be equal. `val_loss` is not comparable
    # across values of this, for the same reason `--score-lambda` exists: the
    # target itself moves. Rank scale changes by mae_cp, or by games.
    eval_scale: float = EVAL_SCALE


def load_init(path, cfg):
    """The checkpoint `--init` names, checked against the net this run builds.

    A shape mismatch is an error rather than something to adapt to. Adopting the
    checkpoint's width would silently ignore `--l1`/`--l2`, which is exactly
    wrong when the flags came from a shape sweep: every trial would train the
    saved width and the sweep would report that width does not matter.

    Only the weights are read. The optimizer state is deliberately not carried
    over even though it would be easy to save - a fine-tune runs over different
    data with its own OneCycle schedule, and Adam moments from a finished cycle
    on a different distribution are not a head start on it.
    """
    ckpt = torch.load(path, weights_only=False, map_location='cpu')
    l1, l2 = ckpt.get('l1', L1), ckpt.get('l2', L2)
    if (l1, l2) != (cfg.l1, cfg.l2):
        raise SystemExit(
            f"--init {os.path.basename(path)} is a {l1}x{l2} net but this run "
            f"builds {cfg.l1}x{cfg.l2}.\nPass --l1 {l1} --l2 {l2} to continue "
            f"it, or drop --init to train the new shape from scratch.")

    return ckpt


class StopSignal:
    """Ctrl-C stops the run at the next batch boundary instead of killing it.

    Training runs for tens of minutes and the checkpoint on disk is only ever
    the best epoch so far, so an abort throws away everything since that epoch -
    including a partial epoch that may well be the best yet. Handling the signal
    turns that into a clean stop: the batch loop breaks, the partial epoch is
    validated and allowed to take the checkpoint, and `run_training` returns
    normally with `stopped` set so the caller knows the run did not finish.

    The second Ctrl-C restores the default handler and raises, so a run that has
    somehow wedged itself between two batch boundaries can still be killed the
    ordinary way.

    Handlers are restored on exit, which matters for grid_search.py: it calls
    `run_training` in a loop and needs its own KeyboardInterrupt back between
    trials.
    """

    def __init__(self, log=print):
        self.requested = False
        self._log = log
        self._prev = {}

    def __enter__(self):
        for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                self._prev[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):
                # not the main thread, or the platform will not take this one
                pass
        return self

    def __exit__(self, *exc):
        while self._prev:
            sig, prev = self._prev.popitem()
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        return False

    def _handle(self, signum, frame):
        if self.requested:
            self.__exit__()
            raise KeyboardInterrupt
        self.requested = True
        self._log("\nstopping after this batch - the partial epoch is still "
                  "validated and still saved if it is the best so far "
                  "(Ctrl-C again to abort now)")


def run_training(sets, cfg, score_req=None, score_lam=None, log=print,
                 verbose=True):
    """Train one net. Returns the split, the best checkpoint's numbers and timing.

    `score_req` selects the weighting used to pick the best epoch and to report
    `val`. Leave it None and it follows the training blend, which is what a
    plain `train.py` run wants. Pass a fixed weighting and the reported loss is
    comparable across runs that trained on *different* blends - which is the
    only way a blend sweep can be ranked.

    `score_lam` is the same, for the search-score / game-result mix: None
    follows `cfg.lam`, and a fixed value makes runs at different lambdas
    comparable.
    """
    device = sets[0].idx.device
    torch.manual_seed(cfg.seed)
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)

    req = cfg.weights or [1.0] * len(sets)
    train_sw = sample_weights(sets, req)
    for ds, w in zip(sets, train_sw):
        ds.weight = w
    score_sw = train_sw if score_req is None else sample_weights(sets, score_req)
    score_lam = cfg.lam if score_lam is None else score_lam

    # ---- split ----
    train_sets, val_sets = [], []
    for ds in sets:
        n_val = int(len(ds) * cfg.val_frac)
        tr, va = split(ds, n_val)
        train_sets.append(tr)
        val_sets.append(va)
        if verbose:
            cov = ''
            if ds.has_outcome is not None:
                cov = f"  outcomes {ds.has_outcome.float().mean().item():.0%}"
            elif cfg.lam < 1.0:
                # silent otherwise: this set is training on the search score
                # alone while the run was asked to blend
                cov = "  outcomes none"
            log(f"  {ds.name:<28} train {len(tr):>9,}  val {len(va):>7,}  "
                f"sample weight {ds.weight:.3f}{cov}")

    if verbose:
        for ds in val_sets:
            if ds.kind == 'wld':
                counts = np.bincount(ds.label.cpu().numpy(), minlength=3)
                log(f"  {ds.name}: majority class baseline "
                    f"{counts.max() / max(1, counts.sum()):.1%}")
            else:
                cp = ds.label.cpu().numpy()
                # what you get by always predicting 0.0 - the number to beat
                log(f"  {ds.name}: predict-zero baseline {np.abs(cp).mean():.1f} cp")

    tr_idx = torch.cat([d.idx for d in train_sets])
    tr_target = torch.cat([d.blended_target(cfg.lam, cfg.eval_scale)
                           for d in train_sets])

    # Per-sample weight = the set's aggregate pull x the piece-count correction.
    # The correction has mean 1 within each set, so it redistributes pull across
    # piece counts without moving it between sets - `--weights` still means what
    # it says. Training only: the validation split keeps the natural
    # distribution so that `val_loss` stays comparable across --piece-balance
    # values, the same reason `--score-weights` exists for the blend.
    tr_weight = torch.cat([
        torch.from_numpy(d.weight * piece_balance_weights(
            d.pieces, cfg.piece_balance, cfg.piece_cap)).to(device)
        for d in train_sets])
    n_train = len(tr_idx)

    if verbose and cfg.piece_balance > 0:
        log(f"\npiece balance {cfg.piece_balance:g} (cap {cfg.piece_cap:g}x) - "
            f"share of training pull by piece count:")
        for ds in train_sets:
            pw = piece_balance_weights(ds.pieces, cfg.piece_balance,
                                       cfg.piece_cap)
            rows = []
            for pc in sorted(set(ds.pieces.tolist())):
                m = ds.pieces == pc
                rows.append(f"{pc}:{m.mean():.0%}->{pw[m].sum() / pw.sum():.0%}")
            log(f"  {ds.name:<28} {'  '.join(rows)}")

    if verbose:
        mem = sum(d.idx.element_size() * d.idx.nelement() for d in sets) / 1e6
        log(f"\ntrain {n_train:,} total  ({mem:.0f} MB of indices on {device})")
        if cfg.lam < 1.0:
            log(f"target = {cfg.lam:g} * search score + {1 - cfg.lam:g} * game "
                f"result, where a result is known")
        log("")

    model = CheckersNet(cfg.l1, cfg.l2).to(device)
    if verbose:
        log(describe(model))
    if cfg.init:
        init_ckpt = load_init(cfg.init, cfg)
        model.load_state_dict(init_ckpt['state_dict'])
        # the saved weights were clipped by the run that produced them, but a
        # checkpoint from a build with different quantization bounds would not
        # be - and this is what guarantees the padding row is still zero
        model.clip_weights()
        if verbose:
            prev = init_ckpt.get('datasets') or ['?']
            log(f"initialised from {cfg.init}\n"
                f"  trained on {', '.join(prev)}, val loss "
                f"{init_ckpt.get('val_loss', float('nan')):.4f}")
            # not fatal, but worth saying out loud: the loaded output layer was
            # trained to mean centipawns at one scale and is about to be
            # trained against a target squashed at another
            prev_scale = init_ckpt.get('eval_scale', cfg.eval_scale)
            if abs(prev_scale - cfg.eval_scale) > 1e-9:
                log(f"  NOTE: it was trained at eval_scale {prev_scale:g} and "
                    f"this run uses {cfg.eval_scale:g}, so its output means "
                    f"something different than the target it is about to be "
                    f"trained against")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    steps_per_epoch = (n_train + cfg.batch - 1) // cfg.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch,
        pct_start=cfg.pct_start)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    best_loss = float('inf')
    best_metrics = {}
    best_epoch = -1
    stopped = False
    epochs_done = 0
    t0 = time.time()
    with StopSignal(log) as stop:
        for epoch in range(cfg.epochs):
            perm = torch.randperm(n_train, device=device)
            total_loss = 0.0
            steps = 0

            for s in range(0, n_train, cfg.batch):
                b = perm[s:s + cfg.batch]
                logits = model(tr_idx[b])
                w = tr_weight[b]
                loss = (loss_fn(logits, tr_target[b]) * w).sum() / w.sum()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                model.clip_weights()

                total_loss += loss.item()
                steps += 1
                if stop.requested:
                    break

            # A partial epoch is validated and allowed to take the checkpoint
            # like any other. It is a fair comparison - the holdout does not
            # care how many batches went into the weights - and it is the whole
            # reason stopping early keeps the work rather than discarding it.
            epochs_done = epoch + 1
            results = evaluate(model, val_sets, loss_fn, score_lam, cfg.eval_scale)
            val_loss = combine_loss(results, score_sw)
            flag = ''
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_metrics = {d.name: m for d, _, m, _ in results}
                flag = ' *'
                torch.save({'state_dict': model.state_dict(),
                            'l1': model.l1_size, 'l2': model.l2_size,
                            'eval_scale': cfg.eval_scale, 'val_loss': val_loss,
                            'datasets': [d.name for d in sets],
                            'weights': list(req), 'lam': cfg.lam,
                            'piece_balance': cfg.piece_balance,
                            # where the weights came from, so a staged run's
                            # lineage survives in the file rather than only in
                            # the shell history that produced it
                            'init': cfg.init,
                            'epoch': epoch + 1, 'epochs': cfg.epochs,
                            'val': best_metrics},
                           cfg.out)

            if verbose:
                line = (f"epoch {epoch + 1:3d}/{cfg.epochs}  "
                        f"loss {total_loss / max(steps, 1):.4f}  "
                        f"val {val_loss:.4f}")
                for ds, _, m, _ in results:
                    line += f"  | {ds.name.split('.')[0]}: {fmt_metrics(ds.kind, m)}"
                if steps < steps_per_epoch:
                    line += f"  [stopped {steps:,}/{steps_per_epoch:,} batches in]"
                log(line + flag)

            if stop.requested:
                stopped = True
                break

    seconds = time.time() - t0
    if verbose:
        if stopped:
            log(f"\nstopped after {epochs_done} of {cfg.epochs} epochs. The "
                f"schedule never annealed, so this is a starting point rather "
                f"than a finished net - continue it with --init {cfg.out}")
        log(f"trained in {seconds:.0f}s, best combined val loss {best_loss:.4f} "
            f"(epoch {best_epoch})")

    return {'val_loss': best_loss, 'best_epoch': best_epoch,
            'metrics': best_metrics, 'seconds': seconds,
            'val_sets': val_sets, 'model': model,
            'params': model.num_params(),
            'stopped': stopped, 'epochs_done': epochs_done}


def piece_breakdown(model, val_sets, log=print):
    """Where the error actually lives, from the best checkpoint."""
    for ds in val_sets:
        logits = infer(model, ds.idx)
        log(f"\n{ds.name} ({ds.kind}) by piece count:")
        if ds.kind == 'wld':
            prob = torch.sigmoid(logits)
            pred = torch.full_like(ds.label, DB_DRAW)
            pred[prob > 2.0 / 3.0] = DB_WIN
            pred[prob < 1.0 / 3.0] = DB_LOSS
            score = (pred == ds.label).float().cpu().numpy()
            unit = lambda v: f"{v:.3%} correct"
        else:
            score = (logits * EVAL_SCALE - ds.label).abs().cpu().numpy()
            unit = lambda v: f"{v:.1f} cp mae"
        for pc in sorted(set(ds.pieces.tolist())):
            m = ds.pieces == pc
            if m.sum():
                log(f"  {pc:2d} pieces: {unit(score[m].mean())}  "
                    f"({int(m.sum()):,} positions)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+',
                    default=[os.path.join(NNUE_DIR, 'data', 'db_positions.npz')],
                    help="one or more .npz files (wld or eval_cp labelled)")
    ap.add_argument('--weights', nargs='+', type=float, default=None,
                    help="relative aggregate weight per dataset, default equal "
                         "regardless of size")
    ap.add_argument('--lambda', dest='lam', type=float, default=1.0,
                    help="search score vs game result in the target: 1.0 (the "
                         "default) is the score alone, 0.0 the result alone. "
                         "Only affects rows whose dataset carries outcomes")
    ap.add_argument('--piece-balance', type=float, default=0.0,
                    help="flatten each set's piece-count distribution: 0 (the "
                         "default) trains on it as generated, 1 gives every "
                         "piece count equal pull, 0.3-0.5 corrects it part of "
                         "the way. The generated sets are ~18%% tablebase-owned "
                         "endings and only ~29%% at the 11+ pieces where the "
                         "net decides games")
    ap.add_argument('--piece-cap', type=float, default=3.0,
                    help="most extra pull any one piece count may get under "
                         "--piece-balance. Without it the rarest bucket, which "
                         "may hold a few hundred rows, would outweigh the "
                         "commonest")
    ap.add_argument('--min-pieces', type=int, default=0,
                    help="drop positions with fewer than N pieces. The engine "
                         "probes the endgame tablebase before the net, and "
                         "DB_MAX_TOTAL is 6, so at 6 pieces and below the net's "
                         "output is discarded - making 7 the setting that "
                         "matches the engine. Refuses to empty a set: an "
                         "all-tablebase set like db_positions.npz has to leave "
                         "--data instead")
    ap.add_argument('--out', default=os.path.join(NNUE_DIR, 'models', 'net.pt'))
    ap.add_argument('--init', default=None,
                    help="start from the weights in this checkpoint instead of "
                         "a random init, which is what makes staged training "
                         "possible: train on db_positions.npz, then --init that "
                         "net and train on midgame data. Weights only - the "
                         "optimizer and the LR schedule start fresh, so the "
                         "second stage runs a full OneCycle of its own and "
                         "usually wants a lower --lr than the first. The shape "
                         "must match (--l1/--l2). If --out names the same file, "
                         "it is copied to <out>.bak first")

    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--limit', type=int, default=0,
                    help="use only the first N positions of each set (smoke tests)")
    ap.add_argument('--seed', type=int, default=0)
    # Network shape. The checkpoint records both, export.py reads them back, and
    # the C engine takes them from the generated header, so a wider net needs no
    # code change anywhere - only a retrain, a re-export and a rebuild.
    #
    # L1 should be a multiple of 128. src/engine/nnue.c holds a tile of the
    # accumulator in registers and that tile is 8 vectors wide (128 int16 under
    # AVX2, 64 under SSE2); a multiple of 16 still vectorises but gives up the
    # register tiling, and anything else silently drops to the scalar path.
    # bench_width.py measures what each width costs before you spend a training
    # run on it.
    ap.add_argument('--l1', type=int, default=L1,
                    help=f"feature transformer width (default {L1}, use multiples of 128)")
    ap.add_argument('--l2', type=int, default=L2,
                    help=f"hidden layer width (default {L2}, any size)")
    ap.add_argument('--eval-scale', type=float, default=EVAL_SCALE,
                    help=f"centipawns per logit in the TRAINING target "
                         f"(default {EVAL_SCALE:g}, from model.py). Recorded in "
                         f"the checkpoint, so export.py and the engine follow "
                         f"it automatically. Note the export-side scale can be "
                         f"overridden separately by match_net.py --eval-scale, "
                         f"which needs no retraining")
    ap.add_argument('--weight-decay', type=float, default=1e-6)
    ap.add_argument('--pct-start', type=float, default=0.15,
                    help="fraction of the schedule spent warming up (OneCycleLR)")
    args = ap.parse_args()

    if args.l1 % 128:
        print(f"warning: L1={args.l1} is not a multiple of 128, so the engine "
              f"gives up register tiling in the feature transformer"
              + ("" if args.l1 % 16 == 0 else " and falls back to scalar code"))

    if args.weights and len(args.weights) != len(args.data):
        print(f"--weights has {len(args.weights)} entries for "
              f"{len(args.data)} datasets")
        return 1

    if args.init and not os.path.exists(args.init):
        print(f"--init {args.init} does not exist")
        return 1

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Continuing a net in place is a reasonable thing to ask for, but the first
    # improving epoch overwrites the very weights the run started from, and the
    # pretrained stage is then unrecoverable. Copy it aside rather than refuse:
    # the stage that produced it may have cost hours.
    if args.init and os.path.abspath(args.init) == os.path.abspath(args.out):
        backup = args.out + '.bak'
        shutil.copyfile(args.init, backup)
        print(f"--init and --out are the same file; kept the starting net as "
              f"{backup}")

    cfg = TrainConfig(out=args.out, init=args.init,
                      epochs=args.epochs, batch=args.batch,
                      lr=args.lr, val_frac=args.val_frac, seed=args.seed,
                      l1=args.l1, l2=args.l2, weight_decay=args.weight_decay,
                      pct_start=args.pct_start, weights=args.weights,
                      lam=args.lam, piece_balance=args.piece_balance,
                      piece_cap=args.piece_cap, eval_scale=args.eval_scale)

    # A shape mismatch is checked here as well as in run_training, because
    # encoding 17M positions takes a minute and a wrong --l1 should not cost
    # one. The result is thrown away; run_training loads it again and reports.
    if cfg.init:
        load_init(cfg.init, cfg)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("WARNING: CUDA not available, training on CPU")
    else:
        print(f"training on {torch.cuda.get_device_name(0)}")

    # ---- load ----
    t0 = time.time()
    sets = []
    for path in args.data:
        ds = load_dataset(path, device, args.limit, args.min_pieces)
        sets.append(ds)

        print(f"  {ds.name:<28} {len(ds):>9,} positions  labels={ds.kind}")
    print(f"loaded and encoded in {time.time() - t0:.1f}s")

    res = run_training(sets, cfg)

    # ---- breakdown by piece count, from the best checkpoint ----
    model = res['model']
    ckpt = torch.load(cfg.out, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    try:
        piece_breakdown(model, res['val_sets'])
    except KeyboardInterrupt:
        # the checkpoint is already on disk; only the report is being skipped
        print("\n(breakdown skipped)")

    # which epoch's weights are actually in the file, since "the best so far"
    # is not the last one and after an early stop it may be well short of it
    print(f"\nsaved {cfg.out} (epoch {ckpt['epoch']} of "
          + (f"{res['epochs_done']}, run stopped early - continue it with "
             f"--init {cfg.out})" if res['stopped'] else f"{cfg.epochs})"))
    return 0


if __name__ == '__main__':
    sys.exit(main())
