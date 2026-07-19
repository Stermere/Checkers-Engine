"""Midgame training data: self-play games labelled with the search's own eval.

The tablebase sampler (`gen_db_data.py`) can only produce positions with <=5
pieces, which is exactly the region where the engine already has a perfect
lookup. A net trained on that alone has never seen a midgame position. This
script fills the gap by distillation: play games with the engine and label every
position with what a fixed-depth search thinks of it.

The eval is **side-to-move relative** (proved by `verify_eval_convention.py`,
which also pins down the return tuple layout), which is the same perspective
`features.py` encodes in, so labels are used with no sign correction.

`--teacher kingsrow` hands the games to a much stronger engine (see
`kingsrow.py`): it picks every move *and* labels every position, both off one
search per ply, in the same units and the same side-to-move-relative
convention. It is cheaper than it sounds, because it replaces our own search
rather than adding to it - a Kingsrow run costs one search per ply where it used
to cost two (ours to move, its to label).

Letting it play took reading its move out of the PV string, since the `CBmove`
struct comes back all zeros, and rebuilding captures: Kingsrow writes a
multi-jump compressed, `6x29`, with the landing squares left out, so the hops
are recovered with `pdn.find_capture_path`. On a position it names no move for -
its endgame databases answer with a list of candidates and no PV - that one ply
falls back to our own search, and the count is reported, because a run that
silently fell back everywhere would look like a Kingsrow run and be a self-play
one.

The consequence is worth stating plainly: the positions are now the ones
*Kingsrow* reaches, not the ones our engine does. Its tree is the better place
to learn from, which is the whole point of a stronger teacher, but it means a
`kingsrow` set and a `self` set differ in their positions as well as their
labels - a comparison between two nets trained on them measures both at once.

Its games are also much longer, so size a run differently: strong play on both
sides rarely resolves, and from one measured seed our engine's game ended after
50 plies where Kingsrow's ran to the 200-ply `--move-cap`, giving 186 labels
against 45. Expect roughly 4x the positions per game, with a tail of drawn
manoeuvring in it, and read positions/game off the first progress lines rather
than sizing `--games` from a `self` run.

What gets thrown away, and why - the filters matter more than the volume:

  * **Positions with a capture available.** A static eval of a position in the
    middle of an exchange is meaningless: the score belongs to the position at
    the end of the forced sequence, not this one. Training on them teaches the
    net to predict tactics it cannot see. This is standard NNUE practice and it
    is the single most important filter here.
  * **The opening plies.** Near-identical across games, so they contribute
    duplicates and an over-representation of one narrow region.
  * **Mate and tablebase scores.** A proof comes back above WIN_MIN (2000),
    decaying down from WIN_SCORE (4000), while every ordinary evaluation is
    clamped to EVAL_MAX (1500); `MATE_BAND` sits in that gap. They are exact
    terminal facts, not evaluations; letting them in teaches the net to
    memorise a lookup the engine already has, and it drags the sigmoid target
    to saturation.
  * **Positions the tablebase already covers** (`--min-pieces`), for the same
    reason: the db probe runs before the net, so the net's opinion there is
    never used.

Duplicate suppression happens in the parent over the whole run, before the
shuffle, because a position appearing in both the train and the validation
split turns the validation number into a lie. The tablebase sampler cannot
produce duplicates by construction, so `train.py` never had to care; self-play
absolutely can (transpositions, and repeated opening lines).

Where the games start from matters as much as the filters. The 11-man ballot set
in `openings.py` is a few hundred positions, and drawing every game from it
makes the duplicate rate *rise* with dataset size - 5.7% at 56 games, 20% at
17,000 - because more games out of the same narrow mouth of the tree keep
landing on each other. `--seeds pdn` instead starts each game from a position
taken out of a real game in `CheckersGames.pdn`, of which there are ~480,000
distinct ones. Build that pool first; it is cached, so this is a one-off:

    python src/python/nnue/gen_pdn_data.py --pool-only

Output matches `gen_db_data.py` - an .npz of p1/p2/p1k/p2k/stm - except the
label is `eval_cp` (int32 centipawns) instead of `wld`.

Long runs write shards as they go (`--shard-size`). Nothing used to reach disk
until the run finished, so a several-hour run that was killed lost everything,
and the parent held every row in a Python list - about 400 MB at 1.2M rows, and
transiently double that during the numpy conversion.

Re-running the same command resumes - there is no flag for it, and no flag that
turns it off by throwing the shards away. The shards left behind are adopted,
the games already inside them are not replayed, and new shards are numbered
after them, so nothing is overwritten. Ctrl-C flushes the buffer to a shard on
the way out rather than dropping it. `--discard-shards` starts the games over,
and even that renames the old directory aside instead of deleting it: a shard
is hours of search time, so nothing here unlinks one except the successful
write of the .npz it went into. Do not point two concurrent runs at one shard
directory; resume assumes it owns the place.

Run:
    python src/python/nnue/gen_selfplay_data.py --games 200 --workers 8
    python src/python/nnue/gen_selfplay_data.py --games 40000 --workers 14 \
        --seeds pdn --shard-size 250000 --out data/selfplay_3m.npz
"""

import argparse
import json
import os
import random
import sys
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

NNUE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NNUE_DIR)
sys.path.insert(1, os.path.abspath(os.path.join(NNUE_DIR, '..')))

import Board_opperations as bo          # noqa: E402
import bitboard_converter as bc         # noqa: E402
import pdn                              # noqa: E402


MATE_BAND = 1200


# ---------------------------------------------------------------------------
# one game, run inside a worker process
# ---------------------------------------------------------------------------
#: one Kingsrow per worker process. The DLL keeps global state - hash table,
#: book, search threads - so two instances in one process would share it, and
#: constructing one is far too expensive to do per game.
_KINGSROW = None


def _kingsrow(seconds):
    global _KINGSROW
    if _KINGSROW is None:
        import kingsrow
        # note this chdirs the worker to the CheckerBoard directory; everything
        # here works in absolute paths, so nothing downstream notices
        _KINGSROW = kingsrow.Kingsrow(seconds=seconds)
    return _KINGSROW


#: what a worker hands back. The counters are here because a Kingsrow run whose
#: every ply had quietly fallen back to our own engine would look exactly like a
#: Kingsrow run, and be a self-play run. `ending` is the same idea for outcome
#: labels: a run where every game hit the piece floor would produce a file with
#: no outcomes in it and look no different from one that had them.
GameResult = namedtuple('GameResult',
                        'samples plies teacher_moves fallbacks ending')

# How a game finished, which decides the outcome label on every position in it.
# 'win' and 'tie' are resolved on the board. 'cap' is a game still going at
# --move-cap: adjudicated a draw, the same call engine match play makes, and
# sound here because a game neither side can finish in 200 plies is drawish
# almost by definition.
#
# The floor endings are the interesting ones. Games are stopped once both sides
# are down to almost nothing, because the tablebase owns the rest and playing it
# out would spend searches to rediscover what the database already knows - and
# measured on a 24-game smoke run, **22 of 24 games end this way**. Left
# unresolved that would be the whole feature: 16% of positions carrying an
# outcome. So the reached position is handed back to the engine, whose database
# probe answers it exactly. 'floor_won' is a proven result; 'floor_drawn' is
# everything else, which with a database loaded means a proven draw.
ENDINGS = ('win', 'tie', 'cap', 'floor_won', 'floor_drawn')

# board_search.c: |score| <= EVAL_MAX (1500) is a heuristic evaluation and
# |score| > WIN_MIN (2000) is a proven result. The gap between the two bands is
# what makes this test safe - there is no score a position can hold that is
# ambiguous between "the net likes it" and "the database says it is won".
PROVEN_MIN = 2000

# Outcome as integer half-points for the side to move: loss 0, draw 1, win 2.
# -1 means no outcome is known, which train.py reads as "use the search score
# alone" rather than as a draw.
RES_UNKNOWN, RES_LOSS, RES_DRAW, RES_WIN = -1, 0, 1, 2


def _xy(bit):
    """0..63 bit index -> the (x, y) the Board_opperations functions take."""
    return (bit % 8, bit // 8)


def _apply_move(board, hops):
    """Play a complete move onto `board`, one hop at a time."""
    for frm, to in hops:
        bo.update_board(_xy(frm), _xy(to), board)


def _engine_move(board, player, ei, seconds, depth):
    """Our engine's move: `(hops, eval)`. Does not touch `board`.

    The engine answers one hop at a time - a multi-jump continues with `forced`
    set to the square just landed on - so the chain is driven here rather than
    by the game loop, which is what keeps one iteration of that loop equal to
    one whole move. It plays onto a scratch copy because the caller may yet
    decide to play something else (`--eps`).

    The eval returned is the *first* search's, i.e. the score of the position
    handed in, never of some point in the middle of a capture sequence.
    """
    scratch = [row[:] for row in board]
    hops, forced, cp = [], -1, None
    while True:
        p1, p2, p1k, p2k = bc.convert_to_bitboard(scratch)
        res = ei.search(p1, p2, p1k, p2k, player, seconds, depth, forced)
        if cp is None:
            cp = int(res.eval)
        frm, to = res.move
        hops.append((frm, to))
        jumped = bo.update_board(_xy(frm), _xy(to), scratch)
        if jumped and bo.check_jump_required(scratch, player, _xy(to)):
            forced = to             # same player must continue the chain
            continue
        return hops, cp


def _kingsrow_move(board, player, kr):
    """Kingsrow's move and its score, off one search: `(hops, cp)`.

    Either half can be None - they are parsed out of the same status string and
    it does not always carry both, see `Kingsrow.bestmove`. `hops` is None when
    there is no move to be had, which is the caller's cue to fall back to our
    own engine for that ply.

    A capture arrives compressed (`6x29`, landing squares omitted), so the hops
    are recovered against our own move generator. Every hop is checked to be
    legal before it is played, and an unusable token yields no hops at all
    rather than a board that quietly stopped being a checkers position.
    """
    cp, squares, is_capture = kr.bestmove(board, player)
    if squares is None:
        return None, cp

    scratch = [row[:] for row in board]
    hops = []
    for a, b in zip(squares, squares[1:]):
        if _xy(b) in bo.generate_options(_xy(a), scratch, only_jump=is_capture):
            step = [(a, b)]
        elif is_capture:
            # a compressed chain: find the hops that connect the two squares
            step = pdn.find_capture_path(scratch, a, b)
            if step is None:
                return None, cp
        else:
            return None, cp
        for frm, to in step:
            hops.append((frm, to))
            bo.update_board(_xy(frm), _xy(to), scratch)
    return hops, cp


def _random_move(board, player, rng):
    """A complete random legal move as hops, or None if there is none.

    It finishes a multi-jump rather than handing a half-played chain back to the
    game loop, which the loop used to do. That matters now: a board in the
    middle of a capture is not a position an outside engine can be asked about,
    because Kingsrow would be free to continue the chain with a different piece.
    """
    scratch = [row[:] for row in board]
    hops, forced = [], None
    while True:
        if forced is None:
            jumpers = bo.check_jump_required(scratch, player)
            options = [(tuple(m[0]), tuple(m[1])) for m in
                       bo.generate_all_options(scratch, player, bool(jumpers))]
        else:
            options = [(forced, tuple(t)) for t in
                       bo.generate_options(forced, scratch, only_jump=True)]
        if not options:
            return hops or None
        frm, to = rng.choice(options)
        hops.append((frm[0] + frm[1] * 8, to[0] + to[1] * 8))
        jumped = bo.update_board(frm, to, scratch)
        if not (jumped and bo.check_jump_required(scratch, player, to)):
            return hops
        forced = to


def _resolve_from_db(p1, p2, p1k, p2k, player, ei):
    """Who wins the position the game stopped at, according to the tablebase.

    Returns `(winner, proven)`: the winning player or 0, and whether that came
    back as a proof rather than an opinion.

    A search rather than a direct probe because the C module exposes no probe -
    but at the piece floor every child of this position is inside the database,
    so even a shallow search returns a proven score off those probes. (The root
    itself is deliberately never probed; `board_search.c` gates the probe on
    `depth_abs > 0` so the root always produces a move.) Depth 4 is far more
    than needed and still instant on four pieces.

    A proven *draw* is indistinguishable from no answer at all: the database
    deliberately lets DRAW fall through to a normal search, so both come back as
    an ordinary evaluation. That is why `proven` is reported separately instead
    of a draw being inferred from a small score. With a database loaded, a
    not-won-not-lost position at the floor is drawn; with `db/` missing every
    position would look that way, and the caller's job is to notice that the
    proven fraction has collapsed rather than to quietly label a whole run as
    drawn.
    """
    res = ei.search(int(p1), int(p2), int(p1k), int(p2k), int(player), 1.0, 4)
    if res.eval > PROVEN_MIN:
        return player, True
    if res.eval < -PROVEN_MIN:
        return player ^ 3, True
    return 0, False


def play_one_game(args):
    """Play a game and return the labelled positions from it, as a GameResult.

    Samples are (p1, p2, p1k, p2k, stm, eval_cp, result) tuples, where `result`
    is the game's outcome in half-points from that position's side to move (see
    RES_*). One iteration of the loop below plays one *whole* move, multi-jumps
    included, so every position it looks at is one a player is genuinely on move
    in.

    The outcome is only known once the game is over, so positions are collected
    with the side to move recorded and the result is filled in at the end.
    """
    (board, start_player, seed, seconds, depth, skip_plies, random_plies, eps,
     min_pieces, move_cap, teacher, teacher_seconds) = args

    # imported here so the module is loaded once per worker, not at pickle time
    import engine_iface as ei

    rng = random.Random(seed)
    kr = _kingsrow(teacher_seconds) if teacher == 'kingsrow' else None
    player = start_player
    history = []
    samples = []
    ply = 0
    teacher_moves = fallbacks = 0
    winner, ending = 0, 'cap'      # 'cap' is what survives if the loop runs out

    while ply < move_cap:
        p1, p2, p1k, p2k = bc.convert_to_bitboard(board)

        # ---- decide whether this position is worth labelling ----
        # No "mid multi-jump" case to exclude any more: a partly played capture
        # chain never reaches the top of this loop.
        want = (ply >= skip_plies
                and ei.popcount(p1, p2, p1k, p2k) >= min_pieces
                and not ei.has_any_jump(p1, p2, p1k, p2k, player))

        # ---- pick the move, and take the label off the same search ----
        label, hops = None, None
        if kr is not None:
            hops, label = _kingsrow_move(board, player, kr)
            if hops is None:
                fallbacks += 1
            else:
                teacher_moves += 1
        if hops is None:
            # our engine: the teacher under --teacher self, and the fallback for
            # the odd position Kingsrow names no move for
            hops, own_eval = _engine_move(board, player, ei, seconds, depth)
            if kr is None:
                label = own_eval
        if not hops:
            # nothing legal for the side to move, so it has lost
            winner, ending = player ^ 3, 'win'
            break

        if eps > 0.0 and rng.random() < eps:
            # a random legal move now and then, so games do not collapse onto
            # one deterministic line per starting position. The label still
            # stands - it describes this position, not the move played from it.
            hops = _random_move(board, player, rng) or hops

        if want and label is not None and abs(label) < MATE_BAND:
            samples.append((p1, p2, p1k, p2k, player, int(label)))

        history.append([row[:] for row in board])
        _apply_move(board, hops)
        ply += 1
        player ^= 3

        won = bo.check_win(board, player)
        if won != 0:
            winner, ending = won, 'win'
            break
        if bo.check_tie(history):
            winner, ending = 0, 'tie'
            break
        # both sides reduced to almost nothing: the tablebase owns this, so ask
        # it who won instead of spending searches playing it out
        floor = bc.convert_to_bitboard(board)
        if ei.popcount(*floor) <= 4:
            winner, proven = _resolve_from_db(*floor, player, ei)
            ending = 'floor_won' if proven else 'floor_drawn'
            break

    # ---- attach the outcome, from each position's own point of view ----
    if winner == 0:            # 'tie', 'cap' adjudicated drawn, or a drawn floor
        result_for = lambda _stm: RES_DRAW
    else:
        result_for = lambda stm: RES_WIN if stm == winner else RES_LOSS
    samples = [s + (result_for(s[4]),) for s in samples]

    return GameResult(samples, ply, teacher_moves, fallbacks, ending)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def build_schedule(n_games, seed, random_plies, source='ballot',
                   pdn_frac=0.8, pool_cache=None, cut_min=6, cut_max=40):
    """Starting positions for each game, as (board, player_to_move, seed).

    Two sources, because they fail in opposite directions. The ballots are only
    a few hundred positions but they are all *sound* openings played from move
    one; the PDN pool is ~480,000 positions but each one is a snapshot out of
    somebody else's game, so the games played from it are shorter and start
    from whatever imbalance the humans had already created. `mix` takes both.
    """
    rng = random.Random(seed)
    n_pdn = int(round(n_games * pdn_frac)) if source == 'mix' else (
        n_games if source == 'pdn' else 0)

    schedule = []
    if n_pdn:
        schedule += _pdn_seeds(n_pdn, rng, pool_cache, cut_min, cut_max)
    if n_games - len(schedule) > 0:
        schedule += _ballot_seeds(n_games - len(schedule), rng, random_plies)

    rng.shuffle(schedule)
    return schedule


def _pdn_seeds(n, rng, pool_cache, cut_min, cut_max):
    """`n` positions sampled out of the human games, without replacement.

    Sampled from the *deduplicated* pool, so two games cannot start from the
    same board unless n exceeds the pool. The ply window matters: below
    `cut_min` every game is still in book and the seeds are near-identical,
    above `cut_max` most of them are already resolved endgames the tablebase
    owns.
    """
    # deferred: gen_pdn_data imports this module, so a top-level import here
    # would be circular
    import gen_pdn_data as gpd

    arr = gpd.load_pool(gpd.pdn.DEFAULT_PDN, pool_cache or gpd.DEFAULT_POOL)
    ply = arr[:, 5]
    arr = arr[(ply >= cut_min) & (ply <= cut_max)]
    if not len(arr):
        raise SystemExit(f"no PDN positions with ply in [{cut_min}, {cut_max}]")
    print(f"seeds: {len(arr):,} PDN positions in ply [{cut_min}, {cut_max}], "
          f"drawing {n:,}")

    idx = np.random.default_rng(rng.randrange(1 << 30)).permutation(len(arr))
    out = []
    while len(out) < n:
        for i in idx[:n - len(out)]:
            row = arr[i]
            board = bc.convert_to_matrix(int(row[0]), int(row[1]),
                                         int(row[2]), int(row[3]))
            out.append((board, int(row[4]), rng.randrange(1 << 30)))
    return out


def _ballot_seeds(n_games, rng, random_plies):
    """The original source: 11-man ballots walked a few random plies in."""
    import openings
    boards = openings.get_11man_boards()

    schedule = []
    while len(schedule) < n_games:
        board = [row[:] for row in rng.choice(boards)]
        player = 1
        # A few random plies for diversity: deterministic self-play from a
        # fixed opening set produces near-duplicate games. Always an even
        # number, so that player 1 is still to move and the game loop below can
        # assume it - rejecting the odd ones instead would throw away over half
        # the requested games.
        for _ in range(2 * rng.randint(0, random_plies // 2)):
            jumpers = bo.check_jump_required(board, player)
            legal = bo.generate_all_options(board, player, bool(jumpers))
            if not legal:
                break
            m = rng.choice(legal)
            jumped = bo.update_board(m[0], m[1], board)
            if jumped and bo.check_jump_required(board, player, m[1]):
                continue
            player ^= 3
        if player != 1:
            continue        # a multi-jump chain landed us off parity; resample
        schedule.append((board, 1, rng.randrange(1 << 30)))
    return schedule



#: midgame datasets, newest first. Several scripts want "whatever midgame set
#: exists" as a default, and the name keeps changing as the generator improves -
#: `selfplay_1m.npz` was ballot-seeded and is retired. Naming one file in each
#: script means every retirement silently breaks a gate somewhere.
MIDGAME_DATASETS = ('selfplay_3m.npz', 'pdn_positions.npz', 'selfplay_2m.npz')


def default_midgame_datasets():
    """The midgame .npz files that are actually present, newest first."""
    data_dir = os.path.join(NNUE_DIR, 'data')
    return [os.path.join(data_dir, n) for n in MIDGAME_DATASETS
            if os.path.exists(os.path.join(data_dir, n))]


def _key_view(arr):
    """A 1-D void view of the position columns, for set operations on rows."""
    keys = np.ascontiguousarray(arr[:, :5])
    return keys.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()


def deduplicate(rows):
    """Drop repeated positions, keyed on the full board + side to move."""
    arr = np.asarray(rows, dtype=np.int64)
    # np.unique over a structured view is the cheap way to dedupe wide rows
    _, idx = np.unique(_key_view(arr), return_index=True)
    return arr[np.sort(idx)]


def drop_positions_in(arr, paths):
    """Remove any position that also appears in one of the given .npz datasets.

    Needed once there is more than one dataset: `train.py` splits each file
    into its own train and validation halves, so a position present in two
    files can sit in one file's training set and another's validation set. That
    is the same leak deduplication exists to prevent, just spread across files
    instead of within one - and it is not hypothetical here, since the PDN pool
    seeds the self-play games.
    """
    if not paths:
        return arr
    for path in paths:
        d = np.load(path)
        other = np.stack([d['p1'], d['p2'], d['p1k'], d['p2k'], d['stm']],
                         axis=1).astype(np.int64)
        before = len(arr)
        arr = arr[~np.isin(_key_view(arr), _key_view(other))]
        print(f"  -{before - len(arr):,} positions already in "
              f"{os.path.basename(path)}")
    return arr


def write_dataset(arr, out_path, exclude=(), seed=0):
    """Dedupe, exclude, shuffle and write rows as an .npz. Returns the count.

    The last step of a run, factored out because it is also the whole of
    `shards_to_npz.py` - a killed run's shards are a dataset that has not had
    this done to it yet, and salvaging one should not mean a second
    implementation of the dedupe-and-shuffle that decides what the file means.

    None means nothing was left to write.
    """
    if not len(arr):
        print("no positions to write")
        return None

    raw_n = len(arr)
    arr = deduplicate(arr)
    print(f"\n{raw_n:,} labelled positions, {len(arr):,} unique "
          f"({raw_n - len(arr):,} duplicates dropped, "
          f"{(raw_n - len(arr)) / raw_n:.1%})")

    if exclude:
        arr = drop_positions_in(arr, exclude)
        print(f"{len(arr):,} after excluding other datasets")
    if not len(arr):
        print("nothing left after excluding")
        return None

    # shuffle so that a prefix split in train.py is a random split
    rng = np.random.default_rng(seed)
    arr = arr[rng.permutation(len(arr))]

    eval_cp = arr[:, 5].astype(np.int32)
    # half-points -> a win probability for the side to move. One game backs each
    # position here (unlike the PDN set, where the pool is shared), so result_n
    # is 1 or 0, and 0 is what tells train.py to ignore `result` entirely rather
    # than read the placeholder 0.5 as a draw.
    half = arr[:, 6]
    known = half != RES_UNKNOWN
    result = np.where(known, half / 2.0, 0.5)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        p1=arr[:, 0].astype(np.uint64), p2=arr[:, 1].astype(np.uint64),
        p1k=arr[:, 2].astype(np.uint64), p2k=arr[:, 3].astype(np.uint64),
        stm=arr[:, 4].astype(np.uint8), eval_cp=eval_cp,
        result=result.astype(np.float32),
        result_n=known.astype(np.uint16))

    print("\neval distribution (centipawns, side-to-move relative):")
    for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
        print(f"  p{q:<3} {np.percentile(eval_cp, q):>8.0f}")
    print(f"  mean {eval_cp.mean():.1f}  std {eval_cp.std():.1f}")
    print(f"  |eval| <= 25: {(np.abs(eval_cp) <= 25).mean():.1%}")

    print(f"\ngame outcome: {known.sum():,}/{len(arr):,} positions "
          f"({known.mean():.1%}) carry one")
    if known.any():
        for name, v in (('win', RES_WIN), ('draw', RES_DRAW), ('loss', RES_LOSS)):
            m = half == v
            print(f"  {name:<5} for the side to move: {m.sum():>9,} "
                  f"({m.mean():.1%})")
        print(f"  mean score for the side to move {result[known].mean():.3f}")
    if (~known).any():
        print(f"  {(~known).sum():,} positions carry none - these come from "
              f"shards written before outcomes were recorded, and train on the "
              f"search score alone")

    print(f"\nwrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return len(arr)


# ---------------------------------------------------------------------------
# shards on disk, and resuming from them
# ---------------------------------------------------------------------------
MANIFEST = 'manifest.jsonl'

#: the arguments that decide what game index N is, and what its labels mean.
#: Resuming across a change to any of these would splice two different
#: experiments into one file, so a mismatch is refused rather than merged.
#: Everything else (--workers, --shard-size, --exclude) is free to change.
RUN_KEYS = ('games', 'seed', 'seeds', 'pdn_frac', 'cut_min', 'cut_max',
            'random_plies', 'depth', 'seconds', 'skip_plies', 'eps',
            'min_pieces', 'move_cap', 'pool_cache', 'teacher',
            'teacher_seconds')


def _read_manifest(shard_dir):
    """(header, records) from the manifest, or (None, []) if there is none."""
    path = os.path.join(shard_dir, MANIFEST)
    if not os.path.exists(path):
        return None, []
    header, records = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # a kill during the append leaves a torn last line; everything
                # before it is still good
                break
            if header is None:
                header = rec
            else:
                records.append(rec)
    return header, records


#: columns in a shard: p1, p2, p1k, p2k, stm, eval_cp, result.
#: 6 is the pre-outcome layout, still readable - see `load_shard`.
SHARD_COLS = 7
SHARD_COLS_LEGACY = 6


def _shard_rows(path):
    """Row count of a shard, or None if it is not a readable shard.

    mmap so this reads the .npy header and not the 12 MB behind it. A shard
    truncated by a kill mid-write fails here instead of at the concatenate,
    which is the difference between skipping one shard and losing the run.
    """
    try:
        arr = np.load(path, mmap_mode='r')
    except (ValueError, OSError, EOFError):
        return None
    if arr.ndim != 2 or arr.shape[1] not in (SHARD_COLS, SHARD_COLS_LEGACY):
        return None
    return len(arr)


def load_shard(path):
    """One shard, widened to the current column count.

    Shards written before outcomes were recorded have 6 columns. They are
    perfectly good positions and labels and are worth hours of search time
    each, so they are read and marked "no outcome known" rather than refused -
    which is the same call `ShardStore.adopt` makes about a missing manifest.
    """
    arr = np.load(path)
    if arr.shape[1] == SHARD_COLS_LEGACY:
        pad = np.full((len(arr), SHARD_COLS - SHARD_COLS_LEGACY),
                      RES_UNKNOWN, dtype=arr.dtype)
        arr = np.hstack([arr, pad])
    return arr


class ShardStore:
    """The on-disk half of a run: shards, plus which games are inside them.

    A killed run leaves its shards behind, and with no record of *which* games
    produced them a resume cannot know what is left to play - so the old code
    ignored them, restarted the numbering at shard_0000, and overwrote hours of
    work with no warning. The manifest is what makes shards resumable: one JSON
    line per shard, appended and fsynced only after the .npy is safely in
    place, so the worst a kill can cost is the games still in the buffer.

    Shards that predate the manifest, or that a torn write left unrecorded, are
    adopted as data anyway - their games get replayed and the duplicate rows
    fall out in `deduplicate()`. Replaying a few games is cheap; deleting a
    shard because its bookkeeping is missing is not.
    """

    def __init__(self, shard_dir, config):
        self.dir = shard_dir
        self.config = config
        self.paths = []         # shards that will make up the final array
        self.done = set()       # game indices known to be inside those shards
        self.rows_on_disk = 0
        self.notes = []         # anything odd found while adopting

    # -- resume ------------------------------------------------------------
    def adopt(self):
        """Take over a previous run's shards. False if there are none."""
        if not os.path.isdir(self.dir):
            return False
        header, records = _read_manifest(self.dir)
        if header is not None:
            self._check_config(header.get('config'))

        adopted = []
        for rec in records:
            path = os.path.join(self.dir, rec['shard'])
            rows = _shard_rows(path) if os.path.exists(path) else None
            if rows is None:
                # the manifest is ahead of the disk. Stop trusting game
                # attributions here; anything after this is picked up below as
                # an unrecorded shard and its games are replayed.
                break
            adopted.append((path, rows, list(rec.get('games', ()))))

        recorded = {p for p, _, _ in adopted}
        for name in sorted(os.listdir(self.dir)):
            if not (name.startswith('shard_') and name.endswith('.npy')):
                continue
            path = os.path.join(self.dir, name)
            if path in recorded:
                continue
            rows = _shard_rows(path)
            if rows is None:
                # keep the bytes, just out of the way of the concatenate
                os.replace(path, path + '.corrupt')
                self.notes.append(f"{name} is unreadable, set aside as "
                                  f"{name}.corrupt")
                continue
            adopted.append((path, rows, []))

        if not adopted:
            return False

        for path, rows, games in adopted:
            self.paths.append(path)
            self.done.update(games)
            self.rows_on_disk += rows

        # rewrite the manifest to say exactly what we hold, so this is the
        # last resume that has to reason about torn lines or stray files
        self._rewrite(adopted)
        if header is None:
            self.notes.append("no manifest here, so the settings these shards "
                              "were built with cannot be checked against the "
                              "ones you just passed")
        untracked = sum(1 for _, _, g in adopted if not g)
        if untracked:
            self.notes.append(f"{untracked} of {len(adopted)} shard(s) carry no "
                              f"game record; those games replay and the "
                              f"duplicates drop out at the end")
        return True

    def _check_config(self, config):
        if config is None:
            raise SystemExit(
                f"{self.dir} has a manifest with no run settings in it.\n"
                f"Pass --discard-shards to set it aside and start over.")
        diff = {k: (v, self.config.get(k)) for k, v in config.items()
                if self.config.get(k) != v}
        if diff:
            lines = '\n'.join(f"  --{k.replace('_', '-')}: shards were built "
                              f"with {was!r}, now {now!r}"
                              for k, (was, now) in sorted(diff.items()))
            raise SystemExit(
                f"{self.dir} belongs to a different run:\n{lines}\n"
                f"Re-run with the original settings to resume it, or pass "
                f"--discard-shards to set those shards aside and start over.")

    def _rewrite(self, adopted):
        with open(os.path.join(self.dir, MANIFEST), 'w') as f:
            f.write(json.dumps({'config': self.config}) + '\n')
            for path, rows, games in adopted:
                f.write(json.dumps({'shard': os.path.basename(path),
                                    'rows': rows,
                                    'games': sorted(games)}) + '\n')
            f.flush()
            os.fsync(f.fileno())

    # -- writing -----------------------------------------------------------
    def _next_index(self):
        """One past the highest shard number present, so nothing is clobbered."""
        used = [-1]
        for name in os.listdir(self.dir):
            if name.startswith('shard_') and name.endswith('.npy'):
                try:
                    used.append(int(name[len('shard_'):-len('.npy')]))
                except ValueError:
                    pass
            # a .corrupt shard still owns its number
            elif name.startswith('shard_') and name.endswith('.npy.corrupt'):
                try:
                    used.append(int(name[len('shard_'):-len('.npy.corrupt')]))
                except ValueError:
                    pass
        return max(used) + 1

    def write(self, rows, games):
        """Flush rows (and the games that produced them) to a new shard."""
        os.makedirs(self.dir, exist_ok=True)
        manifest = os.path.join(self.dir, MANIFEST)
        if not os.path.exists(manifest):
            self._rewrite([])

        arr = np.asarray(rows, dtype=np.int64).reshape(-1, SHARD_COLS)
        path = os.path.join(self.dir, f'shard_{self._next_index():04d}.npy')
        tmp = path + '.writing'
        # write then rename, so a kill can only leave a .writing file behind
        # and never a half-written shard the next run has to guess about
        with open(tmp, 'wb') as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

        with open(manifest, 'a') as f:
            f.write(json.dumps({'shard': os.path.basename(path),
                                'rows': len(arr),
                                'games': sorted(games)}) + '\n')
            f.flush()
            os.fsync(f.fileno())

        self.paths.append(path)
        self.done.update(games)
        self.rows_on_disk += len(arr)
        return path

    def load(self):
        return np.concatenate([load_shard(p) for p in self.paths])

    def cleanup(self):
        """Drop the shards. Only ever called once the .npz is on disk."""
        for path in self.paths:
            os.remove(path)
        manifest = os.path.join(self.dir, MANIFEST)
        if os.path.exists(manifest):
            os.remove(manifest)
        # anything else in there (.corrupt, a stray .writing) is somebody's
        # evidence; leave it and leave the directory with it
        if os.path.isdir(self.dir) and not os.listdir(self.dir):
            os.rmdir(self.dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=200)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--depth', type=int, default=6,
                    help="search depth used to play and to label each "
                         "position. With --teacher kingsrow our engine does "
                         "neither, so this only reaches the odd ply Kingsrow "
                         "names no move for")
    ap.add_argument('--seconds', type=float, default=1.0,
                    help="per-move time cap; keep above what --depth needs so "
                         "the label depth is consistent")
    ap.add_argument('--skip-plies', type=int, default=1)
    ap.add_argument('--random-plies', type=int, default=1)
    ap.add_argument('--eps', type=float, default=0.025,
                    help="probability of playing a random move, for diversity")
    ap.add_argument('--min-pieces', type=int, default=5,
                    help="skip positions the endgame tablebase already covers")
    ap.add_argument('--move-cap', type=int, default=200)
    ap.add_argument('--teacher', choices=('self', 'kingsrow'), default='self',
                    help="who plays and labels. --teacher kingsrow replaces our "
                         "search entirely: it picks every move and scores every "
                         "position off the same one search per ply, so the "
                         "positions are the ones Kingsrow reaches. Both report "
                         "in the same units and the same side-to-move-relative "
                         "convention, but they are different oracles - do not "
                         "mix them within one training set")
    ap.add_argument('--teacher-seconds', type=float, default=0.1,
                    help="time per ply for --teacher kingsrow, which has no "
                         "depth control; 0.05 reaches depth 15-19. This buys "
                         "the move and the label together")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', choices=('ballot', 'pdn', 'mix'), default='mix',
                    help="where games start: the 11-man ballots, positions out "
                         "of CheckersGames.pdn, or both")
    ap.add_argument('--pdn-frac', type=float, default=0.8,
                    help="with --seeds mix, the share of games seeded from PDN")
    ap.add_argument('--pool-cache', default=None,
                    help="the PDN position pool built by gen_pdn_data.py")
    ap.add_argument('--cut-min', type=int, default=6)
    ap.add_argument('--cut-max', type=int, default=40,
                    help="ply window PDN seeds are drawn from")
    ap.add_argument('--exclude', nargs='*', default=[],
                    help=".npz datasets whose positions must not appear here, "
                         "so the same position cannot land in one file's train "
                         "split and another's validation split")
    ap.add_argument('--shard-size', type=int, default=10000,
                    help="flush to disk every N positions; 0 keeps everything "
                         "in memory until the end. Sharding is also what makes "
                         "a run resumable")
    ap.add_argument('--shard-dir', default=None)
    ap.add_argument('--discard-shards', action='store_true',
                    help="do NOT resume: set the previous run's shards aside "
                         "and start the games over from scratch")
    ap.add_argument('--out', default=os.path.join(NNUE_DIR, 'data',
                                                  'selfplay_positions.npz'))
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shard_dir = args.shard_dir or (out_path + '.shards')

    # before building the schedule: a settings mismatch should be reported now,
    # not after several minutes of loading the PDN pool
    store = ShardStore(shard_dir, {k: getattr(args, k) for k in RUN_KEYS})
    if args.discard_shards and os.path.isdir(shard_dir):
        # renamed, never deleted. Hours of search time go into these shards and
        # a mistyped flag should not be able to spend them.
        aside = f"{shard_dir}.discarded-{time.strftime('%Y%m%d-%H%M%S')}"
        os.rename(shard_dir, aside)
        print(f"--discard-shards: previous shards moved to {aside}\n"
              f"  delete that directory yourself once you are sure")
    elif store.adopt():
        print(f"resuming {shard_dir}: {len(store.paths)} shard(s), "
              f"{store.rows_on_disk:,} positions, "
              f"{len(store.done):,} games already played")
        for note in store.notes:
            print(f"  ! {note}")

    schedule = build_schedule(args.games, args.seed, args.random_plies,
                              source=args.seeds, pdn_frac=args.pdn_frac,
                              pool_cache=args.pool_cache,
                              cut_min=args.cut_min, cut_max=args.cut_max)
    oracle = (f"played and labelled by kingsrow at {args.teacher_seconds}s/ply"
              if args.teacher == 'kingsrow' else
              f"played and labelled by our own search at depth {args.depth}")
    print(f"{len(schedule)} games, {args.workers} workers, {oracle}")
    print(f"filters: ply >= {args.skip_plies}, pieces >= {args.min_pieces}, "
          f"no capture available, |eval| < {MATE_BAND}\n")

    jobs = [(board, player, seed, args.seconds, args.depth, args.skip_plies,
             args.random_plies, args.eps, args.min_pieces, args.move_cap,
             args.teacher, args.teacher_seconds)
            for board, player, seed in schedule]
    pending = [(i, j) for i, j in enumerate(jobs) if i not in store.done]
    if len(pending) < len(jobs):
        print(f"{len(pending):,} games left to play\n")

    rows = []
    buffered = set()            # games whose rows are in `rows` but not on disk
    base_rows = store.rows_on_disk
    t0 = time.time()
    done = 0
    teacher_moves = fallbacks = 0
    endings = {k: 0 for k in ENDINGS}
    interrupted = False
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(play_one_game, j): i for i, j in pending}
            try:
                for fut in as_completed(futures):
                    game = fut.result()
                    rows.extend(game.samples)
                    teacher_moves += game.teacher_moves
                    fallbacks += game.fallbacks
                    endings[game.ending] += 1
                    buffered.add(futures[fut])
                    done += 1
                    if args.shard_size and len(rows) >= args.shard_size:
                        store.write(rows, buffered)
                        rows, buffered = [], set()
                        print(f"  -> shard {len(store.paths)} written "
                              f"({store.rows_on_disk:,} so far)")
                    if done % 10 == 0 or done == len(pending):
                        el = time.time() - t0
                        n = store.rows_on_disk + len(rows)
                        eta = (el / done * (len(pending) - done)) / 60
                        # the fallback count is the only thing that says a
                        # kingsrow run really was played by kingsrow
                        played = teacher_moves + fallbacks
                        fb = (f"  {fallbacks / played:.1%} plies fell back to "
                              f"our engine" if args.teacher == 'kingsrow'
                              and played else "")
                        # a run where 'floor' dominates produces a file with
                        # almost no outcome labels in it, and nothing later
                        # would say so
                        ends = ' '.join(f"{k}:{v}" for k, v in endings.items() if v)
                        print(f"  {len(store.done) + len(buffered)}/"
                              f"{len(jobs)} games  {n:,} positions  "
                              f"{(n - base_rows) / el if el else 0:.0f} pos/s  "
                              f"{el:.0f}s elapsed  eta {eta:.0f}m{fb}"
                              f"  [{ends}]")
            except KeyboardInterrupt:
                # the buffer is finished work; a shard is the only way it
                # survives. The games still running are lost either way, so
                # cancel what has not started and take the checkpoint.
                interrupted = True
                for f in futures:
                    f.cancel()
                print("\ninterrupted - checkpointing finished games "
                      "(waiting for in-flight games to stop)")
    else:
        print("every game is already on disk; assembling the dataset")

    if buffered and (store.paths or args.shard_size or interrupted):
        store.write(rows, buffered)
        rows, buffered = [], set()
        print(f"  -> shard {len(store.paths)} written "
              f"({store.rows_on_disk:,} positions on disk)")

    if interrupted:
        if store.paths:
            print(f"\n{len(store.done):,}/{len(jobs)} games are recorded in "
                  f"{shard_dir}\nresume with the same command")
        else:
            print("\nnothing reached disk (no --shard-size, no finished games)")
        return 130

    # Most games end at the piece floor, and their outcome comes from the
    # engine's database probe. If `db/` were missing or unloadable every one of
    # them would come back "not proven" and be labelled a draw - a whole run of
    # outcome labels that are quietly worthless. Say so loudly instead.
    floor_total = endings['floor_won'] + endings['floor_drawn']
    if floor_total >= 20 and endings['floor_won'] / floor_total < 0.02:
        print(f"\n! {endings['floor_drawn']:,} of {floor_total:,} games ended at "
              f"the piece floor and NOT ONE resolved to a proven win.\n"
              f"! That is what a missing endgame database looks like: every one "
              f"of those games is being labelled a draw.\n"
              f"! Check that db/ is present (CHECKERS_DB_DIR overrides it) "
              f"before training on this file.")

    if store.paths:
        arr = store.load()
    elif rows:
        arr = np.asarray(rows, dtype=np.int64)
    else:
        print("no positions produced")
        return 1
    del rows

    if write_dataset(arr, out_path, args.exclude, args.seed) is None:
        return 1

    # only now that the real file exists, since the shards are the only copy
    store.cleanup()
    print(f"total time {time.time() - t0:.0f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
