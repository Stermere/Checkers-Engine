// Author: Collin Kees
// language: C
// Description: this program serves as the C library for move generation and evaluation of checkers boards

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "board_eval.c" // includes set, transposition table, and board evaler

// Define some constants
#define True 1
#define False 0
#define Null 0
#undef INFINITY
#define INFINITY 1000000000
#undef min
#undef max
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define PRINT_OUTPUT 1
#define TERMINATE_EARLY_THRESHOLD 20

// ---- score bands ----
// Every score the search handles is either an opinion or a proof, and the two
// must live in ranges that cannot overlap:
//
//   |score| <= EVAL_MAX   heuristic evaluation - what the net thinks
//   |score| >  WIN_MIN    a PROVEN result, encoded as WIN_SCORE - (plies to the win)
//
// The separation is what makes the comparisons in the search mean anything. A
// proven win outranks any heuristic score however good it looks, and among
// proven wins the shortest always scores highest - whether the proof came from a
// database probe or from a mate the search found itself, since both are measured
// in plies on the same scale. adjust_mate_score walks the distance up one ply per
// return, so a win N plies from the root scores WIN_SCORE - N there no matter how
// the N was split between search and database.
//
// The old constants (1000 for a win, a mate band starting at 900) had neither
// property. The static evaluation reaches +901 on a lopsided king ending, so it
// overlapped both the mate band and the whole database band, and database wins
// past 100 plies never decayed at all - the engine could not tell a win now from
// the same win ten plies later, and would take an unproven position over a
// proven long win or the other way round essentially at random.
#define EVAL_MAX   1500   // static evaluations are clamped to this
#define WIN_MIN    2000   // above this a score is a proven result and decays per ply
#define WIN_SCORE  4000   // a win available right now, ie. at a distance of 0 plies

// The transposition entry stores `eval` as int16 so a bucket fits one cache line
// (see struct hash_table_entry). Every score that reaches add_hash_entry is
// bounded by WIN_SCORE + 1 - the INFINITY timeout sentinel returns out of negmax
// before the store - so this is the bound that keeps that narrowing honest.
typedef char tt_eval_fits_int16[(WIN_SCORE + 1 <= 32767) ? 1 : -1];

// ---- search feature flags (for A/B testing individual features) ----
// every one of these is #ifndef-guarded so a single build can be ablated from
// the command line without editing this file, which is what makes it possible
// to measure features individually rather than as a bundle:
//
//   python src/python/Package_engine.py build --name search_engine_x
//          --define USE_HISTORY=1 --define USE_PVS=0
//
// (bundling is how the aspiration/history/root-reorder experiment ended up
// uninterpretable: three features moved together, the total was negative, and
// nothing said which one caused it.)
//
//
// ============================================================================
// WHAT IS DELIBERATELY ABSENT, AND WHY
// ============================================================================
// Each of these was implemented, measured against a build identical but for the
// one change, and then DELETED rather than left switched off. The measurement is
// the reason it is gone; re-adding one without a new measurement is going
// backwards. Node figures are at fixed depth 12 over 24 ballots
// (bench_eval_speed.py), Elo figures are bot_vs_bot.py.
//
//   Aspiration windows          +3.2% nodes. Retested 2026-08-05 alone, with PVS
//                               underneath and a proper widening ladder, so the
//                               implementation was not the problem: a failed
//                               window re-searches the whole root, and checkers
//                               scores swing too hard move-to-move for +-25.
//
//   Late move pruning           -15.9% nodes but -49 Elo, CI(-61,-36) at
//                               --depth 10. Loses ~37 Elo net.
//
//   Internal iterative reduction  -5.3% nodes, -5 Elo CI(-18,+7). The saving is
//                               worth about the accuracy it costs; a wash.
//
//   Logarithmic LMR curve       +2.9% nodes. ln(depth)*ln(index) only departs
//                               from the flat rule at high move indices, which a
//                               checkers node never reaches.
//
//   Counter-move table          +1.7% nodes.
//   History malus               +1.8% nodes.
//   History ageing              +7.1% nodes. Worst of the three, and standard
//                               practice everywhere else - it loses because the
//                               history table is allocated per search, so the
//                               early iterations are the only thing filling it.
//
//   "Improving" flag            +8.6% nodes for a marginal accuracy gain.
//
//   Capture ordering (MVV-LVA)  +0.03% nodes, null. The generator emits one jump
//                               SEGMENT at a time, so every capture takes exactly
//                               one piece and there is no victim to choose
//                               between. Worth revisiting ONLY if whole
//                               multi-jumps ever become single moves.
//
//   Root move reordering        by the previous iteration's scores. Never
//                               separable from the bundle it was measured in, and
//                               the transposition move already orders the root.
//
//   Single-reply extension      forced replies not consuming depth. Related
//                               capture extensions measured -30 Elo; too
//                               explosive at these time controls.
//                               RETESTED 2026-08-08, on its own this time and
//                               restricted to num_moves == 1 (a branching factor
//                               of one, so it adds a node per ply rather than a
//                               subtree - not the capture extension above).
//                               +103% nodes at fixed depth 12, and at equal time
//                               it reaches 11.6 nominal plies against 14.0.
//                               500 games @0.1s: +1 Elo, CI(-29,+32). A WASH, not
//                               a loss: the accuracy it buys on forced lines pays
//                               for the depth it costs, almost exactly. Left in
//                               behind USE_SINGLE_REPLY_EXTENSION (default 0)
//                               rather than deleted, because a wash at 0.1s may
//                               not be a wash at a long time control - that is the
//                               retest, and it is the only one worth running.
//
//   Raising MAX_DEPTH_MARGIN    from 10 to 24. Reports +1.7 plies of "extended
//                               depth" at equal time with the nominal depth and
//                               the fixed-depth tree both unchanged, because only
//                               ~0.1% of nodes ever reach the cap.
//                               500 games @0.1s: -3 Elo, CI(-34,+27). It makes the
//                               number the engine prints bigger without making the
//                               engine better, so the default stays 10.
//                               NOTE: bench_eval_speed.py cannot measure this at
//                               all - max_depth is min(i + margin, search_depth),
//                               so at a fixed depth the margin is clamped away and
//                               both builds are identical. Only a timed match sees
//                               it. Any future depth-cap tuning has the same blind
//                               spot.
//
// FOUR of the first six are ordering or move-count heuristics that are standard
// wins in chess and made the tree BIGGER here. The cause is one number: a
// checkers node has about eight moves against chess's ~35, and the transposition
// move plus two killers already fill the front of that list. There is almost no
// tail left to sort, so a new ordering heuristic displaces something the existing
// scheme ranked correctly, and move-count pruning cuts into the middle of the
// list rather than the dregs. Rank any future candidate by nodes FIRST - it costs
// seconds, and a feature that adds nodes needs no match at all.
// ============================================================================


#ifndef USE_HISTORY
#define USE_HISTORY 1         // order quiet moves by history heuristic (-2.1% nodes)
#endif
// The PV-node depth extension. This is flagged separately because USE_PVS
// changes how often it fires without touching a line of its code: a node
// searched with a null window can never be labelled PV_NODE (that needs
// board_eval strictly between alpha and beta, and beta == alpha + 1 leaves no
// room), so under PVS only the real principal variation carries the label and
// the extension stops firing all over the tree. Whether that is a gain or a
// loss is a separate question from whether PVS itself is, and it cannot be
// answered without being able to turn the extension off.
#ifndef USE_PV_EXTENSION
#define USE_PV_EXTENSION 1
#endif
#ifndef USE_FORCED_CONTINUATION
#define USE_FORCED_CONTINUATION 1 // multi-jumps must continue with the same piece (correct rules)
#endif
#ifndef USE_ENDGAME_DB
#define USE_ENDGAME_DB 1      // probe the endgame WLD tablebase (db/wld_*.bin) when present
#endif
// Keep the best root move of an iteration that ran out of time, instead of
// discarding the whole iteration and playing the previous depth's choice.
//
// Measured +2 Elo, CI(-20,+23), 1000 games @0.1s - i.e. nothing. That is worth
// recording because the reasoning said 10-20: with a branching factor of 3-5 the
// unfinished iteration is bigger than every finished one together, so throwing it
// away looked like throwing away most of the thinking time.
//
// The reason it is not is the persistent transposition table. The aborted
// iteration's work is not lost when the move is discarded - it is in the table,
// and the next search reads it straight back. What the old code actually discarded
// was one move choice, not the search behind it.
//
// Kept on anyway: the point estimate is positive, it cannot cost anything at a
// fixed depth (it only fires on a timeout), and the same change fixes a real
// defect - the INFINITY sentinel used to be written into search_results->evals[]
// before the abort check, leaving an aborted iteration's score array holding it
// next to stale values from the previous one.
#ifndef USE_PARTIAL_ITERATION
#define USE_PARTIAL_ITERATION 1
#endif

// ---- evaluation-guided search ----
// Everything below asks the evaluation what it thinks of a position the search
// has NOT looked into yet, and acts on the answer. That is only worth doing with
// an evaluation accurate enough to be believed: the handcrafted eval was not,
// which is why none of this existed before the network.
// Asking the network what it thinks of an interior node is FREE here, which is
// the measurement this whole section rests on: a build that computed a static
// eval at every interior node and used it for nothing visited a bit-identical
// tree (3,835,499 nodes both ways, fixed depth 12) at the same speed. The lazy
// accumulator in nnue.c is why - materialising an interior node shortens the
// walk-back for every leaf evaluation beneath it, so the extra evaluations very
// nearly pay for themselves. The eval is not the cost; only the decisions are.
#ifndef USE_STATIC_EVAL
#define USE_STATIC_EVAL 1
#endif
#ifndef USE_RFP
#define USE_RFP 1             // reverse futility pruning: way above beta, stop looking (-5.9% nodes)
#endif
#ifndef USE_FUTILITY
#define USE_FUTILITY 1        // skip quiet moves that cannot plausibly reach alpha (-14.6% nodes)
#endif
// Allow a transposition entry written by a PREVIOUS search (a previous move of the
// game) to produce a cutoff, not just a move for ordering.
//
// This matters because the table is persistent: most of what the engine knows at
// the start of a move was learned on the move before, and refusing to cut on it
// means re-searching it.
//
// Judge it on a fixed-TIME match. A bench_eval_speed run says -1.5% nodes and that
// number should be ignored - consecutive positions there are unrelated ballots, so
// the stale entries a probe can reach are mostly noise, whereas in a real game
// consecutive searches sit one move apart. The bench measures precisely the case
// this feature is worst at.
//
// MEASURED 2026-08-05, and turned ON:
//
//   0.1s/move, 1000 games:  +10 Elo, CI(-11,+32)
//   0.5s/move,  314 games:  +34 Elo, CI(-4,+73)   (read in flight, not to completion)
//
// The second run is what decided it, and not because of its point estimate - at
// 400 games the CI is far too wide to settle anything alone. It decided it because
// of the DIRECTION.
//
// This flag was previously held off on a specific piece of history: the bundle
// containing it measured -30 Elo at 0.1s and **-44 at 1.0s**, i.e. it got worse the
// longer the search. That is the pattern a longer time control should produce if
// stale cutoffs are the problem. The retest does the opposite - the effect grows
// with the time control, which is what a genuinely useful feature looks like here,
// since a longer search per move means the previous move's entries cover more of
// the tree this one wants.
//
// So the historical degradation belonged to the OTHER half of that bundle: the
// transposition probe running before the draw-by-repetition check, which is fixed
// and is now unconditional. Allowing a previous search's entries to cut is not
// what was costing the Elo.
//
// Still worth knowing: neither run's confidence interval excludes zero. The case
// rests on two independent positive point estimates, a mechanism that makes sense,
// and the disconfirmation of the reason it was off. A 1.0s run would settle it.
#ifndef USE_STALE_TT_CUTOFF
#define USE_STALE_TT_CUTOFF 1
#endif

// Margins are in centipawns on the network's scale (NNUE_EVAL_SCALE = 120, so a
// man is worth roughly 100), applied per ply of remaining depth.
//
// These were swept, and the sweep mattered more than the reasoning did. The
// first RFP setting tried was 140/depth<=4 - deliberately conservative, on the
// argument that checkers has no null move to fall back on and that zugzwang is
// not an exception here but the normal state of an endgame, so "I am so far
// ahead I could afford to do nothing" is exactly the assumption a checkers
// position is entitled to violate. That reasoning is still sound, but the
// conservative margin measured +2.4% nodes - it fired too rarely to pay for its
// own has_any_jump test, and looked like the structural failure it was
// reasonable to predict. Loosening to 80/depth<=6 turned it into -5.9%.
// A margin choice inverted the sign; do not conclude from one point.
//
// Pushing further (futility 4/100, RFP 60/8) bought only 4.3% more nodes over
// these values, so this is where the returns stop.
#ifndef RFP_MAX_DEPTH
#define RFP_MAX_DEPTH 6
#endif
#ifndef RFP_MARGIN
#define RFP_MARGIN 80
#endif
#ifndef FUTILITY_MAX_DEPTH
#define FUTILITY_MAX_DEPTH 3
#endif
#ifndef FUTILITY_MARGIN
#define FUTILITY_MARGIN 120
#endif

// ---- reduction / extension behavior ----
// building with /D OLD_ENGINE (build_old.bat) keeps the previous behavior so that
// bot_vs_bot.py can measure the elo delta of these fixes head to head.
#ifdef OLD_ENGINE
  #define USE_VERIFIED_LMR 0            // quiet-only late move reductions with a verification re-search
  #define USE_OLD_REDUCTION 1           // blanket late move reduction (reduces captures too, never re-searched)
  #define USE_LEGACY_MATERIAL_REDUCTION 1 // root-relative "-2 ply" reduction, fires inside every exchange
  #define USE_FREE_JUMP_CHAIN 0         // every segment of a multi jump burns a ply of depth
  #define USE_PVS 0                     // null-window search of every move after the first
#else
  #define USE_VERIFIED_LMR 1
  #define USE_OLD_REDUCTION 0
  #define USE_LEGACY_MATERIAL_REDUCTION 0
  #define USE_FREE_JUMP_CHAIN 1
  #ifndef USE_PVS
  #define USE_PVS 1
  #endif
#endif



// How many plies past the nominal iteration depth the search may run.
#ifndef MAX_DEPTH_MARGIN
#define MAX_DEPTH_MARGIN 10
#endif

// Forced replies do not consume depth. Off by default until measured; see the
// note in should_extend_or_reduce for why this is not the capture extension that
// was already tried and deleted.
#ifndef USE_SINGLE_REPLY_EXTENSION
#define USE_SINGLE_REPLY_EXTENSION 0
#endif

// late move reduction tuning.
// these can be aggressive because every reduced search that beats alpha is
// re-searched at full depth, so a reduction can only ever cost time - unlike the
// old unverified "-2 ply" material rule it can never hide a refutation.
#define LMR_MIN_MOVE_INDEX 3  // first move of a node that may be reduced
#define LMR_MIN_DEPTH 3       // minimum remaining depth for a reduction
#define LMR_MIN_PLY 2         // minimum distance from the root for a reduction
#define LMR_LATE_MOVE_INDEX 8 // from this move on, reduce one ply more
#define LMR_MATERIAL_DELTA 2  // piece lead (vs the root) that earns an extra ply of reduction





#ifdef PYTHON
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#endif


// Define some functions TODO use a header file for this
struct set* get_piece_locations(long long p1, long long p2, long long p1k, long long p2k);
void update_piece_locations(int piece_loc_initial, int piece_loc_after, struct set* piece_loc);
void undo_piece_locations_update(int piece_loc_initial, int piece_loc_after, struct set* piece_loc);
int get_next_board_state(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, int player, int piece_type, char* offsets);
int get_piece_at_location(long long p1, long long p2, long long p1k, long long p2k, int pos);
int update_board(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after);
int update_board_typed(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after, int piece_type);
void undo_board_update(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type, int initial_piece_type);
int generate_all_moves(long long p1, long long p2, long long p1k, long long p2k, int player, int* moves, struct set* piece_loc, char* offsets, int* jump);
int generate_moves(long long p1, long long p2, long long p1k, long long p2k, int pos, int* save_loc, char* offsets, int only_jump);
int negmax(long long* p1, long long* p2, long long* p1k, long long* p2k, int player,
    struct set* piece_loc, int depth, int alpha, int beta,
    struct board_evaler* evaler, unsigned long long int hash, int depth_abs, int forced_pos);
struct search_info* start_board_search(long long p1, long long p2, long long p1k, long long p2k, int player, float search_time, int search_depth, int forced_pos);
void human_readble_board(long long p1, long long p2, long long p1k, long long p2k);
long long n_ply_search(long long* p1, long long* p2, long long* p1k, long long* p2k, int player, struct set* piece_loc, char* offsets, int depth);
unsigned long long int update_hash(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, unsigned long long int hash, struct board_evaler* evaler);
unsigned long long int update_hash_typed(int pos_init, int pos_after, int piece_type, int jumped_piece_type, unsigned long long int hash, struct board_evaler* evaler);
void end_board_search(struct board_evaler* evaler);
void print_line(long long p1, long long p2, long long p1k, long long p2k, int player, unsigned long long hash, struct board_evaler* evaler);


// Holds the search result and the evaler struct
struct search_info {
    struct board_evaler* evaler;
    short best_move;
    int eval;
};

// positions already reached in the current game (root of every search we ran).
// seeded into the draw table each search so the engine avoids repeating positions
// when winning and steers into repetitions when losing. a new game is detected
// when the piece count increases (piece count only ever falls within a game).
#define GAME_HISTORY_MAX 1024
static unsigned long long GAME_HISTORY[GAME_HISTORY_MAX];
static int GAME_HISTORY_LEN = 0;
static int LAST_TOTAL_PIECES = 0;

// Python comunication code 
#ifdef PYTHON

// Python function to search the board
static PyObject* search_position(PyObject *self, PyObject *args){
    unsigned long long int p1, p2, p1k, p2k, player, search_depth;
    float search_time;
    int forced_pos = -1;

    // Get the arguments from python (p1, p2, p1k, p2k, player, search_time, search_depth[, forced_pos])
    // forced_pos: optional square (0-63) of a piece that is mid multi-jump and must
    // continue jumping (pass -1 or omit for a normal position)
    if (!PyArg_ParseTuple(args, "KKKKKfK|i", &p1, &p2, &p1k, &p2k, &player, &search_time, &search_depth, &forced_pos))
        return NULL;

    // Get the Result
    struct search_info* search_info = start_board_search(p1, p2, p1k, p2k, player, search_time, search_depth, forced_pos);

    // Package the relevant data into a python tuple
    PyObject* py_list = PyList_New(0);
    PyObject* py_tuple;
    py_tuple = Py_BuildValue("ii", (search_info->best_move >> 8) & 0xFF, search_info->best_move & 0xFF);
    PyList_Append(py_list, py_tuple);


    // Get some stats about the search.
    // `evals` is appended rather than inserted: callers index this tuple
    // positionally (bench_eval_speed.py reads [2] and [4]), so a new field may
    // only ever go on the end.
    // Field 1 is max_ply - the deepest ply the search actually reached, counting
    // the capture-only quiescence past the horizon - and NOT extended_depth,
    // which stops at the last node that still had depth budget left. Callers
    // display this as "how deep did it look", and in checkers, where captures are
    // mandatory and chains run a long way, extended_depth understates that badly.
    // extended_depth is still printed in the engine's own report next to both
    // other numbers; it is just not the one worth exporting.
    //
    // Format codes must match the C types exactly: this is a varargs call, so a
    // mismatch reads the wrong number of bytes off the argument list rather than
    // converting. `search_depth` and `max_ply` are int ("i"), `nodes`,
    // `num_entries` and `evals` are long long ("L"). They were all "K"
    // (unsigned long long), which for the two ints read four bytes of adjacent
    // stack as the high half of the value.
    py_tuple = Py_BuildValue("iiLLiL", search_info->evaler->search_depth, search_info->evaler->max_ply,
                        search_info->evaler->nodes, search_info->evaler->hash_table->num_entries,
                        search_info->eval, search_info->evaler->evals);
    PyList_Append(py_list, py_tuple);

    // Free the search tree the evaler and the transposition table
    end_board_search(search_info->evaler);
    free(search_info);

    // Return the python tuple
    return py_list;
}

// ---- NNUE test hooks -------------------------------------------------------
// These exist so src/python/nnue/verify_nnue_c.py can hold nnue.c to a bit-exact
// standard: the same positions go through the C code and through
// export.int_forward(), and every single one must agree. They are compiled into
// both builds, so the OLD_ENGINE build (which evaluates by hand) can still be
// used to verify the net.


// nnue_eval(p1, p2, p1k, p2k, stm) -> int centipawns, side-to-move relative
static PyObject* py_nnue_eval(PyObject *self, PyObject *args){
    unsigned long long p1, p2, p1k, p2k;
    int stm;
    if (!PyArg_ParseTuple(args, "KKKKi", &p1, &p2, &p1k, &p2k, &stm))
        return NULL;
    return PyLong_FromLong(nnue_eval_position((long long)p1, (long long)p2,
                                              (long long)p1k, (long long)p2k, stm));
}

// nnue_features(p1, p2, p1k, p2k, stm) -> list of active feature indices.
// Compared against features.encode_reference() so that an encoder bug shows up
// as an encoder bug rather than as a mysterious eval mismatch.
static PyObject* py_nnue_features(PyObject *self, PyObject *args){
    unsigned long long p1, p2, p1k, p2k;
    int stm, n, i;
    int idx[NNUE_MAX_ACTIVE];
    PyObject* list;

    if (!PyArg_ParseTuple(args, "KKKKi", &p1, &p2, &p1k, &p2k, &stm))
        return NULL;

    n = nnue_encode((long long)p1, (long long)p2, (long long)p1k, (long long)p2k, stm, idx);
    list = PyList_New(n);
    if (list == NULL) return NULL;
    for (i = 0; i < n; i++){
        PyList_SET_ITEM(list, i, PyLong_FromLong(idx[i]));
    }
    return list;
}

// nnue_eval_many(p1, p2, p1k, p2k, stm) -> bytes of int32 evals.
// Takes the raw buffers of five numpy arrays (four uint64, one uint8) so a
// hundred thousand positions cost one call instead of a hundred thousand.
static PyObject* py_nnue_eval_many(PyObject *self, PyObject *args){
    const char *b1, *b2, *b1k, *b2k, *bstm;
    Py_ssize_t n1, n2, n1k, n2k, nstm, i, count;
    const unsigned long long *ap1, *ap2, *ap1k, *ap2k;
    const unsigned char *astm;
    PyObject* out;
    int32_t* dst;

    if (!PyArg_ParseTuple(args, "y#y#y#y#y#", &b1, &n1, &b2, &n2, &b1k, &n1k,
                          &b2k, &n2k, &bstm, &nstm))
        return NULL;

    if (n1 != n2 || n1 != n1k || n1 != n2k || n1 % (Py_ssize_t)sizeof(unsigned long long) != 0){
        PyErr_SetString(PyExc_ValueError, "board buffers must be equal length uint64 arrays");
        return NULL;
    }
    count = n1 / (Py_ssize_t)sizeof(unsigned long long);
    if (nstm != count){
        PyErr_SetString(PyExc_ValueError, "stm buffer must be a uint8 array of the same length");
        return NULL;
    }

    ap1 = (const unsigned long long*)b1;
    ap2 = (const unsigned long long*)b2;
    ap1k = (const unsigned long long*)b1k;
    ap2k = (const unsigned long long*)b2k;
    astm = (const unsigned char*)bstm;

    out = PyBytes_FromStringAndSize(NULL, count * (Py_ssize_t)sizeof(int32_t));
    if (out == NULL) return NULL;
    dst = (int32_t*)PyBytes_AS_STRING(out);

    for (i = 0; i < count; i++){
        dst[i] = (int32_t)nnue_eval_position((long long)ap1[i], (long long)ap2[i],
                                             (long long)ap1k[i], (long long)ap2k[i],
                                             (int)astm[i]);
    }
    return out;
}

// perft(p1, p2, p1k, p2k, player, depth) -> total leaf nodes at that depth.
//
// A pure move generation counter: no evaluation, no transposition table, no
// pruning, so two builds agree on it if and only if they generate the same moves.
// That is the gate for a change to the generator, where equal search node counts
// would only prove the two agree given identical ordering as well - and it is the
// only way to reach n_ply_search, which had no caller at all after main() was
// commented out.
static PyObject* py_perft(PyObject *self, PyObject *args){
    unsigned long long p1, p2, p1k, p2k;
    int player, depth;
    if (!PyArg_ParseTuple(args, "KKKKii", &p1, &p2, &p1k, &p2k, &player, &depth))
        return NULL;

    long long lp1 = (long long)p1, lp2 = (long long)p2;
    long long lp1k = (long long)p1k, lp2k = (long long)p2k;
    struct set* piece_loc = get_piece_locations(lp1, lp2, lp1k, lp2k);
    char* offsets = compute_offsets();

    long long total = n_ply_search(&lp1, &lp2, &lp1k, &lp2k, player, piece_loc, offsets, depth);

    free(offsets);
    free(piece_loc);
    return PyLong_FromLongLong(total);
}

// nnue_info() -> dict describing the compiled-in net, so the test can fail loudly
// if the header and the checkpoint have drifted apart.
// `simd` is 2 for AVX2, 1 for SSE2, 0 for the scalar fallback, and
// `verify_incremental` says whether this build re-checks every incremental
// evaluation against a from-scratch one (it is far too slow for real play).
static PyObject* py_nnue_info(PyObject *self, PyObject *args){
    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i}",
                         "num_features", NNUE_NUM_FEATURES,
                         "max_active", NNUE_MAX_ACTIVE,
                         "l1", NNUE_L1,
                         "l2", NNUE_L2,
                         "quant_act", NNUE_QUANT_ACT,
                         "quant_w", NNUE_QUANT_W,
                         "use_nnue", USE_NNUE,
                         "simd", NNUE_SIMD,
                         "verify_incremental", NNUE_VERIFY_INCREMENTAL);
}

// nnue_debug_counters() -> (checks, mismatches, worst_cp) from the
// NNUE_VERIFY_INCREMENTAL build, or all zeros if it is off. This is how a real
// search - millions of make/unmake pairs, jumps, promotions, multi-jump chains -
// gets to be the test for the incremental accumulator, instead of a synthetic
// position list that would never reach the interesting cases.
static PyObject* py_nnue_debug_counters(PyObject *self, PyObject *args){
#if NNUE_VERIFY_INCREMENTAL
    return Py_BuildValue("LLi", nnue_incremental_checks,
                         nnue_incremental_mismatches, nnue_incremental_worst);
#else
    return Py_BuildValue("LLi", 0LL, 0LL, 0);
#endif
}

// Tell the pyhton interpreter about the functions we want to use
static PyMethodDef c_board_search_methods[] = {
    {"search_position", search_position, METH_VARARGS,
    "takes the 4 64bit integers for the board from convert_to_bitboard, the player,\
     the time and depth to search for, and optionally the square (0-63) of a piece\
     that is mid multi-jump and must continue jumping (-1 / omitted = none)"},
    {"perft", py_perft, METH_VARARGS,
     "perft(p1, p2, p1k, p2k, player, depth) -> leaf node count; pure move\
      generation, used to prove two builds generate the same moves"},
    {"nnue_eval", py_nnue_eval, METH_VARARGS,
     "nnue_eval(p1, p2, p1k, p2k, stm) -> centipawns from the side to move's perspective"},
    {"nnue_features", py_nnue_features, METH_VARARGS,
     "nnue_features(p1, p2, p1k, p2k, stm) -> list of active feature indices"},
    {"nnue_eval_many", py_nnue_eval_many, METH_VARARGS,
     "nnue_eval_many(p1, p2, p1k, p2k, stm) -> bytes of int32 evals; takes raw\
      numpy buffers (four uint64 arrays and one uint8 array)"},
    {"nnue_info", py_nnue_info, METH_NOARGS,
     "nnue_info() -> dict of the compiled network's dimensions and the USE_NNUE flag"},
    {"nnue_debug_counters", py_nnue_debug_counters, METH_NOARGS,
     "nnue_debug_counters() -> (checks, mismatches, worst_cp) for a build made\
      with /D NNUE_VERIFY_INCREMENTAL=1; all zeros otherwise"},
    {NULL, NULL, 0, NULL}
};

// Define the module
// When compiled with /D OLD_ENGINE the extension is exposed as "search_engine_old"
// so that both the current and a reference build can coexist in the same Python process.
// /D ENGINE_MODULE_NAME=name (Package_engine.py --name) overrides both, which is
// how a variant build - a differently optimized engine, or a slow self-checking
// one - can be imported next to the engine that is already in use.
#if defined(ENGINE_MODULE_NAME)
  #define _NNUE_STR2(x) #x
  #define _NNUE_STR(x)  _NNUE_STR2(x)
  #define _NNUE_CAT2(a, b) a##b
  #define _NNUE_CAT(a, b)  _NNUE_CAT2(a, b)
  #define _MODULE_NAME  _NNUE_STR(ENGINE_MODULE_NAME)
  #define _PYINIT_FUNC  _NNUE_CAT(PyInit_, ENGINE_MODULE_NAME)
#elif defined(OLD_ENGINE)
  #define _MODULE_NAME  "search_engine_old"
  #define _PYINIT_FUNC  PyInit_search_engine_old
#else
  #define _MODULE_NAME  "search_engine"

  #define _PYINIT_FUNC  PyInit_search_engine
#endif


static struct PyModuleDef search_engine = {
    PyModuleDef_HEAD_INIT,
    _MODULE_NAME,
    "C board search logic",
    -1,
    c_board_search_methods
};

PyMODINIT_FUNC
_PYINIT_FUNC(void){
    return PyModule_Create(&search_engine);
}

#endif
// End of python comunication code


// a function to round the float to 2 decimal places
float round_float(float num){
    return (float)((int)(num * 100 + 0.5)) / 100;
}

// get the location of all the piece's on the board to avoid looping over unused spots
// takes the board as arguments
// returns a set of all the pieces on the board
// note: assumes a valid board
struct set* get_piece_locations(long long p1, long long p2, long long p1k, long long p2k){
    struct set* piece_loc = create_set();
    for (int i = 0; i < 64; i++){
        if (get_piece_at_location(p1, p2, p1k, p2k, i) != 0){
            set_add(piece_loc, i);
        }
    }
    populate_set_array(piece_loc);

    return piece_loc;
}

// update the array of piece locations
void update_piece_locations(int piece_loc_initial, int piece_loc_after, struct set* piece_loc){
    // set the index of a potential jumped piece to -1
    int middle_piece_loc = -1;
    // check for a jump
    if (abs(piece_loc_initial - piece_loc_after) > 10){
        // get the location of the piece that was jumped
        middle_piece_loc = (piece_loc_initial + piece_loc_after) / 2;
        // remove the jumped piece from the set
        set_remove(piece_loc, middle_piece_loc);
    }
    // remove the piece that was moved from the set and add its new location to the set
    set_remove(piece_loc, piece_loc_initial);
    set_add(piece_loc, piece_loc_after);
}

// undo a update the the array of piece locations
void undo_piece_locations_update(int piece_loc_initial, int piece_loc_after, struct set* piece_loc){
    // set the index of a potential jumped piece to -1
    int middle_piece_loc = -1;
    // check for a jump
    if (abs(piece_loc_initial - piece_loc_after) > 10){
        // get the location of the piece that was jumped
        middle_piece_loc = (piece_loc_initial + piece_loc_after) / 2;
        // add the jumped piece back in to the set
        set_add(piece_loc, middle_piece_loc);
    }

    // add the piece that moved back to the set and remove its new location from the set (ie. undo the move)
    set_remove(piece_loc, piece_loc_after);
    set_add(piece_loc, piece_loc_initial);
}

// order the move list: transposition-table move first, then killer moves,
// then the remaining moves - quiet ones by history-heuristic score, captures by
// what they take
void order_moves(int* moves, int num_moves, struct hash_table_entry* entry, struct killer_entry* killer_entry, long long* history, int is_jump,
                 long long p1, long long p2, long long p1k, long long p2k){
    short best_move = (entry != NULL) ? entry->best_move : NO_MOVE;
    int sorted_index = 0;
    // move the transposition table moves to the start of the move list
    if (best_move != 0){
        char start = best_move >> 8;
        char end = best_move & 0xFF;
        // move the best move to the start of the move list
        for (int i = sorted_index; i < num_moves; i++){
            if (moves[i * 2] == start && moves[i * 2 + 1] == end){
                int temp = moves[i * 2];
                moves[i * 2] = moves[sorted_index * 2];
                moves[sorted_index * 2] = temp;
                temp = moves[i * 2 + 1];
                moves[i * 2 + 1] = moves[sorted_index * 2 + 1];
                moves[sorted_index * 2 + 1] = temp;
                sorted_index++;
                break;
            }
        }
    }

    // move the killer moves
    if (killer_entry->move1 != best_move && killer_entry->move1 != 0){
        char start = killer_entry->move1 >> 8;
        char end = killer_entry->move1 & 0xFF;
        // move the killer move to the start of the move list
        for (int i = sorted_index; i < num_moves; i++){
            if (moves[i * 2] == start && moves[i * 2 + 1] == end){
                int temp = moves[i * 2];
                moves[i * 2] = moves[sorted_index * 2];
                moves[sorted_index * 2] = temp;
                temp = moves[i * 2 + 1];
                moves[i * 2 + 1] = moves[sorted_index * 2 + 1];
                moves[sorted_index * 2 + 1] = temp;
                sorted_index++;
                break;   
            }
        }
    }
    if (killer_entry->move2 != best_move && killer_entry->move2 != 0){
        char start = killer_entry->move2 >> 8;
        char end = killer_entry->move2 & 0xFF;
        // move the killer move to the start of the move list
        for (int i = sorted_index; i < num_moves; i++){
            if (moves[i * 2] == start && moves[i * 2 + 1] == end){
                int temp = moves[i * 2];
                moves[i * 2] = moves[sorted_index * 2];
                moves[sorted_index * 2] = temp;
                temp = moves[i * 2 + 1];
                moves[i * 2 + 1] = moves[sorted_index * 2 + 1];
                moves[sorted_index * 2 + 1] = temp;
                sorted_index++;
                break;
            }
        }
    }

    // Sort the remaining quiet moves by history score (insertion sort, lists are
    // small). The scores are read once into a parallel array and moved along with
    // the moves, rather than being looked up again for every comparison: the inner
    // loop used to index `history` - a 64 KB table, so an L2 hit at best - on each
    // step of each shift. Same comparisons in the same order, so the resulting
    // list is identical; only the number of loads changes.
    if (!is_jump && history != NULL){
        long long score[96];
        for (int i = sorted_index; i < num_moves; i++){
            score[i] = history[(moves[i * 2] << 6) + moves[i * 2 + 1]];
        }
        for (int i = sorted_index + 1; i < num_moves; i++){
            int s = moves[i * 2];
            int e = moves[i * 2 + 1];
            long long h = score[i];
            int j = i - 1;
            while (j >= sorted_index && score[j] < h){
                moves[(j + 1) * 2] = moves[j * 2];
                moves[(j + 1) * 2 + 1] = moves[j * 2 + 1];
                score[j + 1] = score[j];
                j--;
            }
            moves[(j + 1) * 2] = s;
            moves[(j + 1) * 2 + 1] = e;
            score[j + 1] = h;
        }
    }

}

// Column masks for a single diagonal step, the one-square counterparts of
// board_eval.c's JUMP_COL_GE2 / JUMP_COL_LE5. A step of -9 or +7 moves one column
// left, so it needs column >= 1; -7 and +9 move one column right and need
// column <= 6. Rows are handled by the shift itself.
#define STEP_COL_GE1 0xFEFEFEFEFEFEFEFEull
#define STEP_COL_LE6 0x7F7F7F7F7F7F7F7Full

// find the state of the next board after a move
// returns 1 if the moving player is player 2 returns 0 if the player is player 1
// pos is the position the piece will be after the first jump and leading in to the second one
// convenion is that player 1 has internal state of 1 and player 2 has internal state of 2 (ie. 1 is red and 2 is black in a normal match)
int get_next_board_state(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, int player, int piece_type, char* offsets){
    // check if the last move was a non-promoting jump, if it wasn't invert the state
    if ((abs(pos_init - pos_after) < 10) || ((pos_after < 8 || pos_after > 55) && piece_type <= 2)){
        return player ^ 0x3;
    }
    
    // If the last move was a jump, see if it can jump again; if so do not change
    // the state. Only the yes/no is wanted, so this is a four test bitboard
    // predicate rather than a full generate_moves call into a throwaway buffer.
    //
    // Passing `piece_type` (the type at pos_init) is correct here rather than
    // looking up the type at pos_after: the early return above has already
    // excluded the only case where the two differ, a man that just promoted.
    (void)offsets;
    return can_jump_from(p1, p2, p1k, p2k, pos_after, piece_type) ? player : (player ^ 0x3);
}

// get the type of piece at the position specified
int get_piece_at_location(long long p1, long long p2, long long p1k, long long p2k, int pos){
    if (p1 >> pos & 1){
        return 1;
    } else if (p2 >> pos & 1){
        return 2;
    } else if (p1k >> pos & 1){
        return 3;
    } else if (p2k >> pos & 1){
        return 4;
    } else {
        return 0;
    }
}

// update the board with a move and return the type of piece that was captured
// returns 0 if no piece was captured.
//
// `piece_type` is what stands on piece_loc_initial. The search already knows it -
// it needs the same value for the undo and for the network's feature delta - and
// re-deriving it here cost a four branch scan of the bitboards on every single
// move made. update_board() below keeps the original signature for the standalone
// tools (db_gen.c, opening_book.c), which are not on any hot path.
int update_board_typed(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after, int piece_type){
    int return_value = 0;
    if (piece_type == 1){
        *p1 = *p1 ^ (1ll << piece_loc_initial);
        *p1 = *p1 ^ (1ll << piece_loc_after);
        // check if the piece should become a king
        if (piece_loc_after < 8){
            *p1k = *p1k ^ (1ll << piece_loc_after);
            *p1 = *p1 ^ (1ll << piece_loc_after);
        }
    } else if (piece_type == 2){
        *p2 = *p2 ^ (1ll << piece_loc_initial);
        *p2 = *p2 ^ (1ll << piece_loc_after);
        // check if the piece should become a king
        if (piece_loc_after > 55){
            *p2k = *p2k ^ (1ll << piece_loc_after);
            *p2 = *p2 ^ (1ll << piece_loc_after);
        }
    } else if (piece_type == 3){
        *p1k = *p1k ^ (1ll << piece_loc_initial);
        *p1k = *p1k ^ (1ll << piece_loc_after);
    } else if (piece_type == 4){
        *p2k = *p2k ^ (1ll << piece_loc_initial);
        *p2k = *p2k ^ (1ll << piece_loc_after);
    }
    // the endpoints are both non-negative, so >> 1 is exactly / 2 without the
    // sign correction the compiler must otherwise emit for a signed divide
    if (abs(piece_loc_initial - piece_loc_after) > 10){
        int mid = (piece_loc_initial + piece_loc_after) >> 1;
        return_value = get_piece_at_location(*p1, *p2, *p1k, *p2k, mid);
        if (return_value == 1){
            *p1 = *p1 ^ (1ll << mid);
        } else if (return_value == 2){
            *p2 = *p2 ^ (1ll << mid);
        } else if (return_value == 3){
            *p1k = *p1k ^ (1ll << mid);
        } else if (return_value == 4){
            *p2k = *p2k ^ (1ll << mid);
        }
    }

    return return_value;
}

// the original signature, for callers that do not already know the piece type
int update_board(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after){
    return update_board_typed(p1, p2, p1k, p2k, piece_loc_initial, piece_loc_after,
                              get_piece_at_location(*p1, *p2, *p1k, *p2k, piece_loc_initial));
}

// reverse the a board update
void undo_board_update(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type, int initial_piece_type){
    // what is standing on the destination square follows from what left the
    // origin square: a man that reached the far row is now a king, everything
    // else is unchanged. That is update_board_typed's promotion rule read
    // backwards, and it saves a four branch scan of the bitboards per unmake.
    int piece_type;
    if (initial_piece_type == 1){
        piece_type = (piece_loc_after < 8) ? 3 : 1;
    } else if (initial_piece_type == 2){
        piece_type = (piece_loc_after > 55) ? 4 : 2;
    } else {
        piece_type = initial_piece_type;
    }
    if (piece_type == 1){
        *p1 = *p1 ^ (1ll << piece_loc_initial);
        *p1 = *p1 ^ (1ll << piece_loc_after);
    } else if (piece_type == 2){
        *p2 = *p2 ^ (1ll << piece_loc_initial);
        *p2 = *p2 ^ (1ll << piece_loc_after);
    } else if (piece_type == 3){
        *p1k = *p1k ^ (1ll << piece_loc_initial);
        *p1k = *p1k ^ (1ll << piece_loc_after);
        // check if the piece should be unkinged
        if (initial_piece_type == 1){
            *p1k = *p1k ^ (1ll << piece_loc_initial);
            *p1 = *p1 ^ (1ll << piece_loc_initial);
        }
    } else if (piece_type == 4){
        *p2k = *p2k ^ (1ll << piece_loc_initial);
        *p2k = *p2k ^ (1ll << piece_loc_after);
        // check if the piece should be unkinged
        if (initial_piece_type == 2){
            *p2k = *p2k ^ (1ll << piece_loc_initial);
            *p2 = *p2 ^ (1ll << piece_loc_initial);
        }
    }
    if (jumped_piece_type != -1){
        if (jumped_piece_type == 1){
            *p1 = *p1 ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (jumped_piece_type == 2){
            *p2 = *p2 ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (jumped_piece_type == 3){
            *p1k = *p1k ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (jumped_piece_type == 4){
            *p2k = *p2k ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        }
    }
}

// updates the hash with the moves that will be made to get to the next move
// Both piece types are already known at the only call site that matters, so this
// takes them rather than rediscovering them: `piece_type` is what is moving and
// `jumped_piece_type` is update_board_typed()'s return value (0 when the move is
// quiet). With those in hand the function does not read the board at all - it is
// pure zobrist arithmetic over the two squares.
unsigned long long int update_hash_typed(int pos_init, int pos_after, int piece_type, int jumped_piece_type, unsigned long long int hash, struct board_evaler* evaler){
    if (jumped_piece_type != 0){
        int mid = (pos_init + pos_after) >> 1;
        if(jumped_piece_type == 1){
            hash ^= evaler->hash_table->piece_hash_diff[mid];
        } else if (jumped_piece_type == 2){
            hash ^= evaler->hash_table->piece_hash_diff[mid + 64];
        } else if (jumped_piece_type == 3){
            hash ^= evaler->hash_table->piece_hash_diff[mid + 128];
        } else if (jumped_piece_type == 4){
            hash ^= evaler->hash_table->piece_hash_diff[mid + 192];
        }
    }
    if (piece_type == 1){
        hash ^= evaler->hash_table->piece_hash_diff[pos_init];
        // a man reaching the back rank promotes, so hash the destination as a king
        if (pos_after < 8){
            hash ^= evaler->hash_table->piece_hash_diff[pos_after + 128];
        } else {
            hash ^= evaler->hash_table->piece_hash_diff[pos_after];
        }
    } else if (piece_type == 2){
        hash ^= evaler->hash_table->piece_hash_diff[pos_init + 64];
        if (pos_after > 55){
            hash ^= evaler->hash_table->piece_hash_diff[pos_after + 192];
        } else {
            hash ^= evaler->hash_table->piece_hash_diff[pos_after + 64];
        }
    } else if (piece_type == 3){
        hash ^= evaler->hash_table->piece_hash_diff[pos_init + 128];
        hash ^= evaler->hash_table->piece_hash_diff[pos_after + 128];
    } else if (piece_type == 4){
        hash ^= evaler->hash_table->piece_hash_diff[pos_init + 192];
        hash ^= evaler->hash_table->piece_hash_diff[pos_after + 192];
    }
    return hash;
}

// the original signature, for callers that do not already know the piece types.
// Must be called BEFORE the move is made on the board, as it was.
unsigned long long int update_hash(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, unsigned long long int hash, struct board_evaler* evaler){
    int piece_type = get_piece_at_location(p1, p2, p1k, p2k, pos_init);
    int jumped_piece_type = 0;
    if (abs(pos_init - pos_after) > 10){
        jumped_piece_type = get_piece_at_location(p1, p2, p1k, p2k, (pos_init + pos_after) >> 1);
    }
    return update_hash_typed(pos_init, pos_after, piece_type, jumped_piece_type, hash, evaler);
}

// generate all possible moves for a given board -
// takes the board and a memory location to save to as arguments
// returns the number of moves generated
// note: moves should have room for 96 elements as this is the maximum number of moves possible on a legal board
int generate_all_moves(long long p1, long long p2, long long p1k, long long p2k, int player, int* moves, struct set* piece_loc, char* offsets, int* jump){
    (void)piece_loc;   // the mover's pieces come straight off the bitboards now
    int num_moves = 0;

    // Whether this is a capture node is decided ONCE, up front.
    //
    // The old code discovered it half way through generating: generate_moves
    // returned -1 the moment it saw a jump, generate_all_moves threw away
    // everything it had already produced, set the flag and rescanned. That is two
    // passes over the piece list in exactly the case that matters most (captures
    // are mandatory, so a capture node is every tactical node in the game), and
    // has_any_jump answers the same question in four shift-and-mask operations.
    //
    // The resulting move list is identical, including its order: the old rescan
    // resumed from the piece that found the jump, and every piece before it had
    // already been proven jumpless by the very scan that returned -1.
    if (!*jump && has_any_jump(p1, p2, p1k, p2k, player)){
        *jump = 1;
    }
    int only_jump = *jump;

    // iterate the mover's pieces in ascending square order, which is the order
    // populate_set_array produced - so the generated list matches the old one
    // element for element
    unsigned long long own = (player == 1)
                                ? (unsigned long long)(p1 | p1k)
                                : (unsigned long long)(p2 | p2k);
    while (own){
        int pos = countTrailingZeros(own);
        own &= own - 1;
        num_moves += generate_moves(p1, p2, p1k, p2k, pos, moves + (num_moves * 2), offsets, only_jump);
    }
    return num_moves;
}

// generate the move's for a single piece
// takes the board, the piece position, memory location to save, and a jump flag as arguments
// returns the number of moves generated or -1 if a jump was found and the jump flag was not set
// note: save_loc should have room for 8 int
int generate_moves(long long p1, long long p2, long long p1k, long long p2k, int pos, int* save_loc, char* offsets, int only_jump){
    (void)offsets;   // board edges are column masks now, not a per-square table
    int num_moves = 0;
    int piece_type = get_piece_at_location(p1, p2, p1k, p2k, pos);
    if (piece_type == 0){
        return 0;
    }

    unsigned long long occupied = (unsigned long long)(p1 | p2 | p1k | p2k);
    unsigned long long empty = ~occupied;
    unsigned long long b = 1ull << pos;
    unsigned long long enemy = (piece_type == 1 || piece_type == 3)
                                  ? (unsigned long long)(p2 | p2k)
                                  : (unsigned long long)(p1 | p1k);

    // p1 men move toward row 0 only, p2 men toward row 7 only, kings both ways
    int can_up = (piece_type != 2);
    int can_down = (piece_type != 1);

    // Each direction is two tests instead of a table lookup plus up to two
    // get_piece_at_location scans. Rows need no masking: a shift that would leave
    // the board brings in zeros by itself, so only the columns can wrap and only
    // they are masked - which is exactly the reasoning has_any_jump already uses,
    // and these are its masks.
    //
    // The direction order (-9, -7, +7, +9) and the "jump beats quiet, and a jump
    // found while only_jump is false aborts the whole node" protocol are the old
    // function's, unchanged, because the search's move ordering is built on them.
#define GEN_STEP(step_mask, jump_mask, enemy_at, empty_at, empty_land, off)         \
    do {                                                                           \
        if (b & (jump_mask) & (enemy_at) & (empty_land)){                          \
            if (!only_jump){                                                       \
                return -1;                                                         \
            }                                                                      \
            save_loc[num_moves * 2] = pos;                                         \
            save_loc[(num_moves * 2) + 1] = pos + 2 * (off);                       \
            num_moves++;                                                           \
        } else if (!only_jump && (b & (step_mask) & (empty_at))){                  \
            save_loc[num_moves * 2] = pos;                                         \
            save_loc[(num_moves * 2) + 1] = pos + (off);                           \
            num_moves++;                                                           \
        }                                                                          \
    } while (0)

    if (can_up){
        // -9 (up-left): landing square is two columns left, so column >= 2
        GEN_STEP(STEP_COL_GE1, JUMP_COL_GE2, enemy << 9, empty << 9, empty << 18, -9);
        // -7 (up-right): landing square is two columns right, so column <= 5
        GEN_STEP(STEP_COL_LE6, JUMP_COL_LE5, enemy << 7, empty << 7, empty << 14, -7);
    }
    if (can_down){
        // +7 (down-left)
        GEN_STEP(STEP_COL_GE1, JUMP_COL_GE2, enemy >> 7, empty >> 7, empty >> 14, 7);
        // +9 (down-right)
        GEN_STEP(STEP_COL_LE6, JUMP_COL_LE5, enemy >> 9, empty >> 9, empty >> 18, 9);
    }
#undef GEN_STEP

    return num_moves;
}

// Move a proven win/loss one ply further from the root. Called on every value
// leaving a node, so a result proven d plies below the root arrives there having
// been walked back d times - which is what makes "shortest win" the highest score.
int adjust_mate_score(int eval) {
    return (eval > WIN_MIN) ?
         eval - 1 : ((eval < -WIN_MIN) ?
         eval + 1 : eval);
}

// plies to the win/loss encoded in a proven score
static inline int win_distance(int eval) {
    return WIN_SCORE - ((eval > 0) ? eval : -eval);
}

// a function that decides if a search should be extended at a certain node.
// reductions live in the search loop below, where the move that caused them is
// known and a fail high can be verified with a full depth re-search.
//
// in_quiescence: this node is past the nominal horizon (captures only)
// is_jump:       every move of this node is a capture (captures are mandatory)
// forced_pos:    >= 0 while a multi jump is still in progress
int should_extend_or_reduce(int depth, int depth_abs, int in_quiescence, int is_jump, int forced_pos,
                            int num_moves, int player, long long p1All, long long p2All,
                            struct hash_table_entry* table_entry,
                            struct board_evaler* evaler) {
    if (depth_abs >= evaler->max_depth){
        return -100;
    }

    if (in_quiescence){
        return depth;
    }

#if USE_SINGLE_REPLY_EXTENSION
    // ---- single reply extension ----
    // A node with exactly one legal move is not a decision. Spending a ply of the
    // depth budget to "choose" it buys no information, and it costs the line the
    // depth it would have had - which in checkers is expensive, because captures
    // are mandatory: 17% of main-search nodes here have exactly one legal move,
    // and 99.99% of those are forced captures.
    //
    // This is narrower than the capture extension that measured -30 Elo and was
    // deleted (see the list at the top of this file). That one extended every
    // capture node, which multiplies: a capture node still has siblings, so
    // extending it grows a subtree. A single-reply node has a branching factor of
    // one, so extending it adds one node per ply, never a subtree. The cost is
    // additive, not multiplicative - which is the whole reason to expect a
    // different answer this time. It is still a measurement, not an argument:
    // rank it by nodes first.
    //
    // The forced_pos case is already free under USE_FREE_JUMP_CHAIN, so excluding
    // it here keeps the two rules from stacking into a double extension.
    if (num_moves == 1 && forced_pos < 0){
        depth++;
    }
#else
    (void)num_moves;
#endif

    // Extract the node type from the table entry
    int node_type = UNKNOWN_NODE;
    if (table_entry != NULL){
        node_type = TT_NODE_TYPE(table_entry);
    }

    // PV-node extension (only for entries from the current search: the persistent
    // transposition table holds PV labels from older searches too, which would
    // otherwise trigger extensions all over the tree)
#if USE_PV_EXTENSION
    if (node_type == PV_NODE && depth_abs > 8 && table_entry->age == evaler->hash_table->age){
           depth++;
    }
#else
    (void)node_type;
#endif

#if USE_LEGACY_MATERIAL_REDUCTION
    // legacy behavior, kept only for the OLD_ENGINE reference build.
    //
    // this was the main weakness of the engine: the material swing is measured
    // against the ROOT piece counts, and every intermediate position of an
    // exchange transiently shows a >= 2 piece swing - that is what an exchange
    // is. worse, it reduces the side that is temporarily UP material, so the
    // moment the engine is +2 in the middle of a trade the line is truncated and
    // scores optimistically. offering two pieces to bait a long capture sequence
    // therefore pushes the recapture wave past the horizon.
    int p1_initial_piece_count = evaler->initial_piece_count_p1;
    int p2_initial_piece_count = evaler->initial_piece_count_p2;
    int p1_piece_count = get_bits_set(p1All);
    int p2_piece_count = get_bits_set(p2All);

    if (player == 1 && (p2_initial_piece_count - p2_piece_count) - (p1_initial_piece_count - p1_piece_count) >= 2){
        depth -= 2;
    } else if (player == 2 && (p1_initial_piece_count - p1_piece_count) - (p2_initial_piece_count - p2_piece_count) >= 2){
        depth -= 2;
    }
#else
    (void)is_jump;
    (void)forced_pos;
    (void)p1All;
    (void)p2All;
    (void)player;
#endif

    return depth;
}


// search the board for the best move recursivly and return the best eval that can be achived from that board position
// forced_pos >= 0 means the node is mid multi-jump: only the piece on that square may move, and it must jump
int negmax(long long* p1, long long* p2, long long* p1k, long long* p2k, int player,
    struct set* piece_loc, int depth, int alpha, int beta,
    struct board_evaler* evaler, unsigned long long int hash, int depth_abs, int forced_pos){
    // setup variables
    int player_next;
    int board_eval = -(WIN_SCORE + 1);  // worse than any real score, so move 1 always takes it
    int num_moves;
    int alpha_orig = alpha;
    int initial_piece_type;
    unsigned long long int next_hash;

    // update search stats
    evaler->nodes++;
    evaler->avg_depth += depth_abs;

    if (depth_abs > evaler->extended_depth && depth >= 0){
        evaler->extended_depth = depth_abs;
    }
    // no depth >= 0 test: this one counts quiescence, which is the point of it
    if (depth_abs > evaler->max_ply){
        evaler->max_ply = depth_abs;
    }

#if SEARCH_DIAG
    if (depth_abs >= evaler->max_depth){
        evaler->diag_capped_nodes++;
    }
#endif

    // if nodes is divisible by 10000, check the time
    // (never abort during the first iteration so a best move is always available)
    // 8192 rather than 10000: a power of two makes this a single AND instead of
    // the multiply-shift-multiply-subtract sequence a constant modulo compiles
    // to, and the interval is arbitrary anyway - it only has to be short enough
    // that the clock is read often enough to stop on time.
    if ((evaler->nodes & 8191) == 0 && evaler->nodes != 0 && evaler->search_depth > 1){
        clock_t current_time = clock();
        double cpu_time_used = ((double)(current_time - evaler->start_time)) / CLOCKS_PER_SEC;
        if (cpu_time_used > evaler->time_limit){
            return INFINITY;

        }
    }

    // check for a draw by repetition FIRST: this must never be masked by a
    // cached transposition entry (cached evals know nothing about repetitions
    // along the current line or in the game so far)
    // (>= rather than ==: the count cannot currently skip 2, since every entry is
    // checked on the way in and only ever incremented by one, but that makes the
    // test correct by an argument about the caller rather than by construction)
    if (get_draw_entry(evaler->draw_table, hash) >= 2){
        return 0;
    }

#if USE_ENDGAME_DB
    // exact endgame database probe (not at the root, and not mid multi-jump).
    // WIN/LOSS return proven scores on the same plies-to-the-win scale the search
    // uses for its own mates, so the engine converts a won ending by strictly
    // decreasing the distance every move (and resists longest when lost), and a
    // database win reached sooner always beats the same win reached later. DRAW
    // deliberately falls through to a normal search: a theoretical draw still has
    // to be played out, and a flat 0 would erase all practical guidance - child
    // probes still guard against ever entering a theoretically lost line.
    //
    // dtw is capped at 254 by the generator and 255 means the distance file was
    // missing, so clamping keeps an unknown distance inside the proven band
    // rather than letting it wrap into the heuristic range.
    if (depth_abs > 0 && forced_pos < 0 && endgame_db_max_pieces() > 0
        && get_bits_set(*p1 | *p2 | *p1k | *p2k) <= endgame_db_max_pieces()){
        int wld, dtw;
        if (probe_endgame_db(*p1, *p2, *p1k, *p2k, player, &wld, &dtw)){
            if (wld == DB_WIN){
                return WIN_SCORE - min(dtw, 254);
            }
            if (wld == DB_LOSS){
                return -WIN_SCORE + min(dtw, 254);
            }
        }
    }
#endif

    // check if this board has been searched to depth before and if so return the eval from the hash table
    // if the value is not a PV-node then use the info that can be used to prune the search
    // notes: no hash cutoffs at the root (the root must always search so the move list gets scored).
    // Entries from PREVIOUS searches may cut as well, under USE_STALE_TT_CUTOFF - with a
    // persistent table that is most of what the engine knows at the start of a move.
    // What makes that safe is the repetition check above running unconditionally first,
    // not the age test: a cached score knows nothing about repetitions along the current
    // line, so the defence has to be at the node, not at the entry.
    struct hash_table_entry* table_entry = get_hash_entry(evaler->hash_table, hash);
    if (table_entry != NULL && table_entry->depth >= depth && TT_PLAYER(table_entry) == player && depth_abs > 0
        && (USE_STALE_TT_CUTOFF || table_entry->age == evaler->hash_table->age)) {
        if (TT_NODE_TYPE(table_entry) == PV_NODE) {
            return adjust_mate_score(table_entry->eval);
        }
        else if (TT_NODE_TYPE(table_entry) == LOWER_BOUND) {
            alpha = max(alpha, table_entry->eval);
        }
        else if (TT_NODE_TYPE(table_entry) == UPPER_BOUND) {
            beta = min(beta, table_entry->eval);
        }
        if (alpha >= beta) {
            return adjust_mate_score(table_entry->eval);
        }
    }

    // get the moves for this board and player combo
    int moves[96];
    int force_captures = (depth <= 0) ? 1 : 0;
    int is_jump = force_captures;

    if (forced_pos >= 0){
        // mid multi-jump: the same piece must continue jumping
        is_jump = 1;
        num_moves = generate_moves(*p1, *p2, *p1k, *p2k, forced_pos, &moves[0], evaler->piece_offsets, True);
    } else {
        num_moves = generate_all_moves(*p1, *p2, *p1k, *p2k, player, &moves[0], piece_loc, evaler->piece_offsets, &is_jump);
    }

    // order the moves: at the root reuse the previous iteration's scores when possible,
    // otherwise use the hash move, killers and history
    int root_preordered = 0;
    if (num_moves > 1 && !root_preordered){
#if USE_HISTORY
        long long* history = evaler->history + (player - 1) * 4096;
#else
        long long* history = NULL;
#endif
        order_moves(&moves[0], num_moves, table_entry, killer_slot(evaler->killer_table, depth_abs),
                    history, is_jump, *p1, *p2, *p1k, *p2k);
    }

    // if there are no move then a player must have won, or there are no captures so end this branch (only count as a win if captures_only is false)
    // this is not a perfect check but it should be good enough
    if (num_moves == 0){
        // if there are no moves and captures only is false then a win has occured eval who won and return
        if (!force_captures){
            return -WIN_SCORE;
        }
        // No captures were available. Distinguish a quiet position (evaluate it)
        // from a loss (the side to move has nothing at all). This used to build a
        // whole second move list over every friendly piece just to compare its
        // length to zero, at what is the most common node in the tree; has_any_move
        // answers the same question with eight shift-and-mask tests.
        if (!has_any_move(*p1, *p2, *p1k, *p2k, player)){
            return -WIN_SCORE;
        }

        // if there are no moves and captures only is true then we found the end of a catures only search evaluate the position and return.
        // clamped so a heuristic score can never reach into the proven band and be
        // mistaken for a win - see the score band comment at the top of the file
        int q_eval = get_eval(*p1, *p2, *p1k, *p2k, player, piece_loc, evaler, depth_abs);
        return max(-EVAL_MAX, min(EVAL_MAX, q_eval));
    }

#if SEARCH_DIAG
    if (num_moves == 1){
        evaler->diag_single_reply++;
        // the subset that is NOT already free: mid multi-jump continuations
        // (forced_pos >= 0) cost no ply under USE_FREE_JUMP_CHAIN, and
        // quiescence nodes are past the nominal horizon anyway
        if (forced_pos < 0 && depth > 0){
            evaler->diag_single_reply_payable++;
            if (is_jump){
                evaler->diag_single_reply_jump++;
            }
        }
    }
#endif

    depth = should_extend_or_reduce(depth, depth_abs, force_captures, is_jump, forced_pos, num_moves, player, *p1 | *p1k, *p2 | *p2k, table_entry, evaler);


    // ---- what the evaluation thinks of THIS node, before searching it ----
    // Computed only where it is both meaningful and affordable:
    //   * depth > 0        - quiescence already evaluates at its own leaves
    //   * !is_jump         - captures are forced, so at a jump node the side to
    //                        move has no choice at all and a static score taken
    //                        mid-exchange describes a position nobody can hold
    //   * forced_pos < 0   - likewise mid multi-jump
    //   * beta > alpha + 1 is NOT required: null-window nodes are exactly where
    //                        the pruning below pays off most
    // The result is cached in the transposition entry, so the second visit to a
    // position (very common across iterative deepening) does not pay again.
    int static_eval = NO_EVAL;
#if USE_STATIC_EVAL
    if (depth > 0 && !is_jump && forced_pos < 0){
        if (table_entry != NULL && table_entry->static_eval != NO_EVAL
            && TT_PLAYER(table_entry) == player){
            static_eval = table_entry->static_eval;
        } else {
            int raw = get_eval(*p1, *p2, *p1k, *p2k, player, piece_loc, evaler, depth_abs);
            static_eval = max(-EVAL_MAX, min(EVAL_MAX, raw));
        }
    }


#if USE_RFP
    // ---- reverse futility pruning ----
    // The side to move is so far above beta that even conceding RFP_MARGIN per
    // remaining ply it still fails high, so return without searching.
    //
    // The assumption is "a position this good does not collapse in the next few
    // plies". In chess that is underwritten by the null move - the side to move
    // can nearly always pass and stay winning. Checkers has no such backstop:
    // the obligation to move IS the losing mechanism in a large class of
    // endings. Three guards keep the bet honest:
    //   * only at non-PV (null-window) nodes, where a wrong bound costs a
    //     re-search rather than the principal variation
    //   * only in quiet positions - neither side has a capture available, so
    //     nothing is hanging that a single reply could take
    //   * never against a proven score, where "close to beta" is meaningless
    if (static_eval != NO_EVAL && beta == alpha + 1 && depth <= RFP_MAX_DEPTH
        && beta < WIN_MIN && beta > -WIN_MIN
        && !has_any_jump(*p1, *p2, *p1k, *p2k, player == 1 ? 2 : 1)
        && static_eval - RFP_MARGIN * depth >= beta){
        return static_eval;
    }
#endif
#endif // USE_STATIC_EVAL


    // the the next boards are ready to be searched so begin the search!
    short best_move = NO_MOVE;
    int eval, flip, next_alpha, next_beta;

    // a fresh root pass has nothing to report yet
    if (depth_abs == 0){
        evaler->search_results->iter_best_move = NO_MOVE;
        evaler->search_results->iter_best_eval = 0;
        evaler->search_results->iter_scored = 0;
    }

    for (int i = 0; i < num_moves; i++){
        // prepare moves
        int move_start = moves[i * 2];
        int move_end = moves[(i * 2) + 1];

#if USE_STATIC_EVAL && USE_FUTILITY
        // ---- futility pruning ----
        // This node is already so far below alpha that a quiet move - which by
        // definition wins no material - cannot plausibly close the gap, so skip
        // it without making it on the board at all. static_eval is only ever set
        // at a node where no capture exists, so every move here is quiet.
        //
        // Never the first move (the ordering has put the best candidate there and
        // it has to be searched for the node to return anything), never at a PV
        // node, and never a move that promotes: a new king is a material-sized
        // swing that this margin does not describe.
        //
        // Like all forward pruning this makes the stored upper bound conditional
        // on the guess being right - the node returns a score that ignores moves
        // it never looked at. That is the trade every futility scheme makes; the
        // margin is what buys it back.
        // The mover's type is wanted by the futility test, by the board update, by
        // the hash update, by the network's feature delta and by the undo. Look it
        // up once, here, and pass it to all of them: it used to be rediscovered
        // four to five times per move by a four branch scan of the bitboards.
        initial_piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, move_start);

        if (static_eval != NO_EVAL && i > 0 && beta == alpha + 1
            && depth > 0 && depth <= FUTILITY_MAX_DEPTH
            && alpha > -WIN_MIN && alpha < WIN_MIN
            && static_eval + FUTILITY_MARGIN * depth <= alpha){
            if (!((initial_piece_type == 1 && move_end < 8) || (initial_piece_type == 2 && move_end > 55))){
                continue;
            }
        }
#else
        initial_piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, move_start);
#endif

        // Make the move first, then hash it. update_hash_typed needs the two piece
        // types rather than the board, and update_board_typed hands back the
        // captured one, so this order lets a single lookup serve both - the old
        // order forced the hash update to re-scan the board for the victim.
        int jumped_piece_type = update_board_typed(p1, p2, p1k, p2k, move_start, move_end, initial_piece_type);
        next_hash = update_hash_typed(move_start, move_end, initial_piece_type, jumped_piece_type, hash, evaler);
        if (forced_pos >= 0){
            // leaving this node clears its forced-continuation marker from the hash
            next_hash ^= evaler->hash_table->piece_hash_diff[FORCED_KEY_OFFSET + forced_pos];
        }

#if !USE_NNUE
        // piece_loc is read only by calculate_eval, which is the OLD_ENGINE build's
        // evaluation. Under USE_NNUE nothing reads it - generate_all_moves takes the
        // mover's pieces straight off the bitboards and get_eval ignores it - so
        // maintaining it on every make and unmake was pure cost.
        update_piece_locations(move_start, move_end, piece_loc);
#endif
        player_next = get_next_board_state(*p1, *p2, *p1k, *p2k, move_start, move_end, player, initial_piece_type, evaler->piece_offsets);
#if USE_NNUE && NNUE_INCREMENTAL
        // record what this move changed so the child's accumulator can be the
        // parent's minus two feature rows plus one, instead of a fresh sum over
        // every piece on the board. Nothing is computed yet - most children
        // return from the hash table or the endgame database without ever being
        // evaluated, and those must not pay for an update they never read.
        nnue_stack_push(evaler->nnue, depth_abs, initial_piece_type, move_start, move_end, jumped_piece_type);
#endif

        // same player again means a forced continuation of this jump; otherwise the side to move flips
        int next_forced = -1;
        if (player_next == player){
#if USE_FORCED_CONTINUATION
            next_forced = move_end;
            next_hash ^= evaler->hash_table->piece_hash_diff[FORCED_KEY_OFFSET + move_end];
#endif
        } else {
            next_hash ^= evaler->hash_table->piece_hash_diff[SIDE_KEY_INDEX];
        }
        // next_hash is final here, and the child will not probe it until after the
        // draw bookkeeping and the reduction rules below have run - enough work to
        // hide most of the cache miss
        prefetch_hash_entry(evaler->hash_table, next_hash);
        add_draw_entry(evaler->draw_table, next_hash);

        // only flip the eval if the player changed
        flip = (player != player_next) ? -1 : 1;
        next_alpha = (flip == 1) ? alpha : -beta;
        next_beta = (flip == 1) ? beta : -alpha;

        // ---- late move reductions ----
        int reduction = 0;
#if USE_VERIFIED_LMR
        // only quiet moves that leave a quiet position may be reduced:
        //  - is_jump      : this move is a capture, captures are forced and tactical
        //  - next_forced  : the move continues a multi jump
        //  - has_any_jump : the move leaves the opponent a capture, ie. it offers a
        //                   trade. reducing these is exactly what let a "sacrifice
        //                   two pieces to start a long exchange" line hide the
        //                   recapture wave behind the horizon
        // every reduced search that beats alpha is re-searched at full depth, so a
        // reduction can cost time but can never change the result of the node.
        if (!is_jump && next_forced < 0 && depth >= LMR_MIN_DEPTH
            && i >= LMR_MIN_MOVE_INDEX && depth_abs >= LMR_MIN_PLY
            && !has_any_jump(*p1, *p2, *p1k, *p2k, player_next)){
            reduction = 1;

            // the deeper into a well ordered move list we are, the less likely the
            // move is to be best
            if (i >= LMR_LATE_MOVE_INDEX){
                reduction++;
            }

            // a line where the side to move is well up on material compared to the
            // root is unlikely to be allowed by the opponent, so look at it last
            // and shallowest (the position is quiet, and a fail high is verified)
            int p1_lost = evaler->initial_piece_count_p1 - get_bits_set(*p1 | *p1k);
            int p2_lost = evaler->initial_piece_count_p2 - get_bits_set(*p2 | *p2k);
            int material_lead = (player_next == 1) ? (p2_lost - p1_lost) : (p1_lost - p2_lost);
            if (material_lead >= LMR_MATERIAL_DELTA){
                reduction++;
            }

            // never reduce straight into quiescence
            if (reduction > depth - 2){
                reduction = depth - 2;
            }
        }
#elif USE_OLD_REDUCTION
        // baseline behavior: reduce late moves deep in the tree (no re-search)
        if (i > 5 && depth_abs + 1 > 8){
            reduction = 1;
        }
#endif

        // a multi jump is a single move, so continuing the chain must not consume a
        // ply: otherwise an n-jump capture eats n plies of depth and long exchanges
        // get cut off half way through
        int full_depth = depth - 1;
#if USE_FREE_JUMP_CHAIN
        if (player_next == player){
            full_depth = depth;
        }
#endif
        int child_depth = full_depth - reduction;

        // ---- principal variation search ----
        // the first move of a node is searched with the real window to establish a
        // score. Every later move only has to prove it is NOT better than that, and
        // a null window ([alpha, alpha+1]) settles that question over a much smaller
        // tree than the real window does, because every child of it can cut off
        // immediately. Only a move that beats alpha anyway is re-searched with the
        // real window to find out what it is actually worth.
        //
        // This is exact: it cannot change the value the node returns, only how many
        // nodes it takes to get there. What it costs is the occasional re-search,
        // and what it saves grows with how well the moves are ordered - the better
        // the first move, the more often the null window is right. That is why it
        // belongs with a strong evaluation rather than before one.
        //
        // Not applied at the root: the root's per-move scores are recorded in
        // search_results->evals and read back by only_viable_move() and the root
        // reorder, and a null window would turn those scores into bounds. Nor in a
        // node that is already searching a null window (beta == alpha + 1), where
        // it would be a no-op.
        int use_pvs = 0;
#if USE_PVS
        use_pvs = (depth_abs > 0 && i > 0 && beta > alpha + 1);
#endif
        int nw_alpha = next_alpha;
        int nw_beta = next_beta;
        if (use_pvs){
            // the null window is [alpha, alpha+1] in THIS node's frame; mirror it
            // for the child only when the side to move flips
            nw_alpha = (flip == 1) ? alpha : -(alpha + 1);
            nw_beta  = (flip == 1) ? alpha + 1 : -alpha;
        }

        eval = negmax(p1, p2, p1k, p2k, player_next, piece_loc, child_depth,
                    nw_alpha, nw_beta, evaler, next_hash,
                    depth_abs + 1, next_forced) * flip;
#if USE_VERIFIED_LMR
        // the reduced search came back above alpha, so it may be a real improvement:
        // verify it at full depth before it is allowed to change the node's result
        // (still on whatever window the first pass used - proving the move is good
        // is a separate question from finding out how good, which the PVS re-search
        // below handles)
        if (reduction > 0 && eval > alpha && eval != INFINITY && eval != -INFINITY){
            eval = negmax(p1, p2, p1k, p2k, player_next, piece_loc, full_depth,
                        nw_alpha, nw_beta, evaler, next_hash,
                        depth_abs + 1, next_forced) * flip;
        }
#endif
        // the null window only ever returns a bound. This move beat alpha, so it is
        // a new principal variation and the node needs its true score - re-search it
        // with the real window. (eval >= beta needs no re-search: the node is about
        // to fail high and the bound is all that leaves it.)
        if (use_pvs && eval > alpha && eval < beta && eval != INFINITY && eval != -INFINITY){
            eval = negmax(p1, p2, p1k, p2k, player_next, piece_loc, full_depth,
                        next_alpha, next_beta, evaler, next_hash,
                        depth_abs + 1, next_forced) * flip;
        }


        // undo the update to the board and piece locations
#if !USE_NNUE
        undo_piece_locations_update(move_start, move_end, piece_loc);
#endif
        undo_board_update(p1, p2, p1k, p2k, move_start, move_end, jumped_piece_type, initial_piece_type);
        remove_draw_entry(evaler->draw_table, next_hash);

        // if the eval is infinity the search is trying to end so return.
        // This is checked BEFORE the root bookkeeping below: the sentinel is not a
        // score, and letting it into evals[] would leave an aborted iteration's
        // array holding INFINITY next to stale values from the previous one.
        if (eval == INFINITY || eval == -INFINITY){
            return INFINITY;
        }

        // store results of moves from the root node
        if (depth_abs == 0) {
            evaler->search_results->evals[i] = eval;
            evaler->search_results->moves[i] = (short)((move_start << 8) | move_end);
            evaler->search_results->iter_scored = i + 1;
        }

        // alpha beta prunning
        if (board_eval < eval){
            board_eval = eval;
            best_move = (move_start << 8) | move_end;

            // publish the root best move the moment it is known rather than at the
            // end of the iteration, so a search that runs out of time part way
            // down the root move list still improves on the previous depth.
            // alpha_orig is the bottom of the window this node was called with, so
            // this is the same test the completed iteration is committed under
            // below: above it the score is a value, at or below it the move only
            // ever produced a fail-soft bound.
            if (depth_abs == 0 && board_eval > alpha_orig){
                evaler->search_results->iter_best_move = best_move;
                evaler->search_results->iter_best_eval = board_eval;
            }
        }
        if (board_eval > alpha){
            alpha = board_eval;
        }
        // prune if a cut off occurs
        if (alpha >= beta){
            if (!is_jump){
                update_killer_table(evaler->killer_table, depth_abs, move_start, move_end);
                if (depth > 0){
                    long long bonus = (long long)depth * depth;
                    evaler->history[(player - 1) * 4096 + (move_start << 6) + move_end] += bonus;
                }
            }
            break;
        }
    }

    if (depth_abs == 0) {
        evaler->search_results->num_moves = num_moves;
        // only trust the best move when it was not a fail-low (aspiration windows can fail low)
        if (board_eval > alpha_orig && best_move != NO_MOVE){
            evaler->search_results->best_move = best_move;
            evaler->search_results->best_eval = board_eval;
        }
    }
    // if alpha and beta have improved then this is a PV node
    char node_type;
    if (board_eval <= alpha_orig){
        node_type = UPPER_BOUND;
    }
    else if (board_eval >= beta){
        node_type = LOWER_BOUND;
    }
    else {
        node_type = PV_NODE;
    }

    // store the eval in the hash table, along with the static evaluation of this
    // node if one was computed - it is independent of the search and stays valid
    // for every later visit, whatever depth or window that visit arrives with
    add_hash_entry(evaler->hash_table, hash, board_eval, depth, player, best_move, node_type,
                   (short)static_eval);

    return adjust_mate_score(board_eval);
}

int only_viable_move(struct search_results* search_results) {
    int m = -9999;
    for (int i = 0; i < search_results->num_moves; i++) {
        m = max(m, search_results->evals[i]);
    }

    int moves_near_m = 0;
    for (int i = 0; i < search_results->num_moves; i++) {
        moves_near_m += (search_results->evals[i] >= (m - TERMINATE_EARLY_THRESHOLD));
    }

    return moves_near_m <= 1;
}


// prepare needed memory for a search and call the search function to find the best move and return a pointer to the memory
// location that holds the moves for the board ordered in the order best to worst.
// forced_pos >= 0 means the position is mid multi-jump: only the piece on that square may move (and must jump)
struct search_info* start_board_search(long long p1, long long p2, long long p1k, long long p2k, int player, float search_time, int search_depth, int forced_pos){
    struct search_info* return_struct = malloc(sizeof(struct search_info));

    // set for the piece locations
    struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);
    int moves[96];

    clock_t start, end;
    start = clock();
    struct board_evaler* evaler = board_evaler_constructor(p1 | p1k, p2 | p2k, search_depth, search_time, start);

#if USE_ENDGAME_DB
    // load the endgame tablebase once per process (no-op if the files are absent)
    {
        const char* db_dir = getenv("CHECKERS_DB_DIR");
        endgame_db_init(db_dir != NULL ? db_dir : "db", DB_MAX_TOTAL);
    }
#endif

    unsigned long long int hash = get_hash(p1, p2, p1k, p2k, evaler->hash_table);
    if (player == 2){
        hash ^= evaler->hash_table->piece_hash_diff[SIDE_KEY_INDEX];
    }
    if (forced_pos >= 0){
        hash ^= evaler->hash_table->piece_hash_diff[FORCED_KEY_OFFSET + forced_pos];
    }

    // update the game history and seed the draw table with it. the harness (and
    // real play) adjudicates a draw on the THIRD occurrence of a position, and the
    // in-search draw check fires when the table count reaches 2, so each position
    // is seeded with (occurrences - 1): played once -> the line must repeat it
    // twice more; played twice -> once more completes the 3-fold.
    int total_pieces = get_bits_set(p1 | p2 | p1k | p2k);
    if (total_pieces > LAST_TOTAL_PIECES){
        GAME_HISTORY_LEN = 0;
    }
    LAST_TOTAL_PIECES = total_pieces;
    int repeat_of_previous = (GAME_HISTORY_LEN > 0
                              && GAME_HISTORY[GAME_HISTORY_LEN - 1] == hash);
    if (forced_pos < 0 && !repeat_of_previous && GAME_HISTORY_LEN < GAME_HISTORY_MAX){
        GAME_HISTORY[GAME_HISTORY_LEN++] = hash;
    }
    for (int gi = 0; gi < GAME_HISTORY_LEN; gi++){
        for (int gj = 0; gj < gi; gj++){
            if (GAME_HISTORY[gj] == GAME_HISTORY[gi]){
                add_draw_entry(evaler->draw_table, GAME_HISTORY[gi]);
                break;
            }
        }
    }

#if USE_NNUE
    // seed ply 0 with the root position; every accumulator below it is rebuilt
    // from here by the deltas the search records as it descends
    nnue_stack_reset(evaler->nnue, p1, p2, p1k, p2k);
#endif

    double cpu_time_used = 0.0;
    int depth = 0;
    int extended_depth = 0;
    // snapshotted alongside extended_depth so the two always describe the same
    // thing - the last COMPLETED iteration - rather than one of them carrying a
    // high water mark from an iteration that was abandoned on the clock
    int max_ply = 0;
    int eval_ = 0;
    int last_eval = 0;
    int have_result = 0;
    int jump = 0;

    // iterative deepening with an aspiration window around the previous score
    for (int i = 1; i <= search_depth; i++){
        // update the evalers search depth
        evaler->search_depth = i;
        // The hard ceiling on how far past the nominal iteration depth anything -
        // extensions, and the capture-only quiescence that jump chains produce -
        // may run. Measured 2026-08-08: only ~0.1% of nodes ever reach it, so it
        // is not shaping the search, but it does set the ceiling on the depth the
        // engine REPORTS, because extended_depth is exactly this value at longer
        // time controls.
        evaler->max_depth = min(max(i + MAX_DEPTH_MARGIN, 5), search_depth);

        eval_ = negmax(&p1, &p2, &p1k, &p2k, player, piece_loc, i, -INFINITY, INFINITY, evaler, hash, 0, forced_pos);

        // get the end time
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        // the search ran out of time part way through this iteration
        if (eval_ == INFINITY || eval_ == -INFINITY){
#if USE_PARTIAL_ITERATION
            // Whatever the root did finish is still worth having. The move
            // ordering searches the previous iteration's best move first, so
            // iter_best_move IS that move until something beats it at the current
            // depth - which makes adopting it a fixed-depth comparison, not a
            // comparison of a deep score against a shallow one. It is NO_MOVE when
            // no root move raised alpha (nothing completed, or the window failed
            // low), and then the previous iteration's choice stands.
            //
            // This matters more than its size suggests: the branching factor makes
            // the unfinished iteration bigger than every completed one put
            // together, and it is the iteration most likely to have found the
            // refutation of the move the previous depth liked.
            if (evaler->search_results->iter_best_move != NO_MOVE){
                evaler->search_results->best_move = evaler->search_results->iter_best_move;
                evaler->search_results->best_eval = evaler->search_results->iter_best_eval;
            }
#endif
            evaler->search_depth = max(depth, 1);
            evaler->extended_depth = extended_depth;
            evaler->max_ply = max_ply;
            break;
        }

        last_eval = eval_;
        have_result = 1;
        depth = i;
        extended_depth = evaler->extended_depth;
        if (depth > extended_depth){
            extended_depth = depth;
        }
        max_ply = evaler->max_ply;
        if (extended_depth > max_ply){
            max_ply = extended_depth;
        }

        if (PRINT_OUTPUT){
            printf("\rPLY: %d\t PLYEX: %d\t Eval: %d   ", depth, evaler->extended_depth, eval_);
        }

        // stop starting new iterations once the time budget is used up
        // (with the persistent transposition table, a partially completed deeper
        // iteration still contributes, so keep searching until the budget is spent)
        if (cpu_time_used > search_time){
            break;
        }

        // terminate once the result is proven AND fully inside the depth already
        // searched, so no shorter win can be hiding past the horizon.
        //
        // The distance test is the point. A database probe proves a win that can
        // be hundreds of plies long from a node only a few plies down, and
        // stopping there locks in whatever entry into the database the shallow
        // search happened to find first - which is how the engine ended up
        // simplifying into won endings that take far longer than they need to.
        // When the whole win fits inside depth i the search has already seen
        // every shorter one, so there is genuinely nothing left to find.
        else if ((eval_ > WIN_MIN || eval_ < -WIN_MIN) && win_distance(eval_) <= i){
            break;
        }

        // if there is only one move left then terminate
        else if ((forced_pos >= 0
                    ? generate_moves(p1, p2, p1k, p2k, forced_pos, &moves[0], evaler->piece_offsets, True)
                    : generate_all_moves(p1, p2, p1k, p2k, player, &moves[0], piece_loc, evaler->piece_offsets, &jump)) == 1){
            break;
        }

        // If there has only been one viable move for the last 4 ply terminate the search
        if (i > 8 && only_viable_move(evaler->search_results)) {
            break;
        }
    }

    // the best move and eval were recorded at the root of the last completed iteration
    short best_move = evaler->search_results->best_move;
    int best_eval = evaler->search_results->best_eval;

    // absolute fallback (should not happen): play the first legal move
    if (best_move == NO_MOVE){
        int n = (forced_pos >= 0)
                    ? generate_moves(p1, p2, p1k, p2k, forced_pos, &moves[0], evaler->piece_offsets, True)
                    : generate_all_moves(p1, p2, p1k, p2k, player, &moves[0], piece_loc, evaler->piece_offsets, &jump);
        if (n > 0){
            best_move = (short)((moves[0] << 8) | moves[1]);
        }
    }

    // print the output of the engine
    if (PRINT_OUTPUT) {
        printf("\r                                                                   \r");
        printf("Search Results:\n");
#if TT_STATS
        printf("HashTable Hit ratio: %lld\n", (evaler->hash_table->hit_count * 100) / (evaler->hash_table->hit_count + evaler->hash_table->miss_count));
#endif
        printf("HashTable Usage: %lld\n", (evaler->hash_table->num_entries * 100llu) / evaler->hash_table->total_size);
        printf("Nodes/s: %fM\n", round_float(((double)evaler->nodes / (cpu_time_used + 0.01)) / 1000000.0));
        printf("Time: %fs\n", cpu_time_used);
        // three different questions, so three numbers rather than one labelled
        // "Depth" that answers whichever the reader assumes:
        //   depth   - nominal, the last iterative deepening iteration that finished
        //   ext     - deepest ply that still had depth budget left (extensions included)
        //   max ply - deepest ply reached at all, counting the capture-only
        //             quiescence past the horizon, which in checkers runs a long way
        printf("Depth: %d  ext: %d  max ply: %d\n",
               evaler->search_depth, evaler->extended_depth, evaler->max_ply);
#if SEARCH_DIAG
        printf("DIAG nodes at max_depth cap: %lld (%.1f%%)\n", evaler->diag_capped_nodes,
               100.0 * (double)evaler->diag_capped_nodes / (double)evaler->nodes);
        printf("DIAG single-reply nodes: %lld (%.1f%%)\n", evaler->diag_single_reply,
               100.0 * (double)evaler->diag_single_reply / (double)evaler->nodes);
        printf("DIAG   payable (not already free): %lld (%.1f%%)\n", evaler->diag_single_reply_payable,
               100.0 * (double)evaler->diag_single_reply_payable / (double)evaler->nodes);
        printf("DIAG   of those, capture nodes: %lld\n", evaler->diag_single_reply_jump);
#endif
        printf("Avg depth: %lld\n", evaler->avg_depth / evaler->nodes);
        printf("Eval: %d\n\n", best_eval);
    }

    // print the line of best moves to the terminal (deguggigng)
    print_line(p1, p2, p1k, p2k, player, hash, evaler);

    free(piece_loc);

    return_struct->evaler = evaler;
    return_struct->best_move = best_move;
    return_struct->eval = best_eval;


    // return the best moves and evaler
    return return_struct;
}

// clean up after the search has been concluded and the thread is about to exit
// note: the transposition table is persistent across searches, so it is not freed here
void end_board_search(struct board_evaler* evaler){
    // free the memory used by the evaler
    free(evaler->killer_table->table);
    free(evaler->killer_table);
    free(evaler->piece_offsets);
    free(evaler->search_results);
    free(evaler->history);
    free(evaler->cone_p1);
    free(evaler->cone_p2);
    nnue_stack_destroy(evaler->nnue);
    free(evaler->piece_pos_map_p1);
    free(evaler->piece_pos_map_p2);
    free(evaler->king_pos_map);
    free(evaler->dist_arr);
    free_draw_table(evaler->draw_table);
    free(evaler);
}

// search to the depth specified and count to total amount of boards for a certain depth
// this is used to test if the move generation is working as expected and to see how much time is spent
long long n_ply_search(long long* p1, long long* p2, long long* p1k, long long* p2k, int player, struct set* piece_loc, char* offsets, int depth){
    // make a int array of size 96 elements to hold the moves
    int moves[96];
    long long total_boards = 0;
    int player_next;
    int pos_init;
    int pos_final;

     // if there are no moves or the depth is 0 return 1
    if (depth == 0){
        total_boards += 1;
        return total_boards;
    }

    // generate the moves for the current board.
    // `jump` is an in/out parameter: in it says "captures only", out it says
    // whether the list that came back is captures. It used to be passed the
    // literal False, which is a null POINTER here - so this function segfaulted on
    // its first call. Nothing had called it since main() was commented out, which
    // is exactly the kind of rot exposing it to Python is meant to stop.
    int jump = 0;
    int num_moves = generate_all_moves(*p1, *p2, *p1k, *p2k, player, &moves[0], piece_loc, offsets, &jump);

    // otherwise loop over all the moves and call the function recursivly
    for (int i = 0; i < num_moves; i++){
        // get the moves being made this iteration of the search
        pos_init = moves[i * 2];
        pos_final = moves[(i * 2) + 1];

        // update the board and set values
        int intitial_piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, pos_init);
        int jumped_piece_type = update_board(p1, p2, p1k, p2k, pos_init, pos_final);
        update_piece_locations(pos_init, pos_final, piece_loc);

        // get the player that is going to move after this move
        player_next = get_next_board_state(*p1, *p2, *p1k, *p2k, pos_init, pos_final, player, intitial_piece_type, offsets);

        // call the function recursivly
        total_boards += n_ply_search(p1, p2, p1k, p2k, player_next, piece_loc, offsets, depth - 1);

        // undo the update to the board and piece locations
        undo_piece_locations_update(pos_init, pos_final, piece_loc);
        undo_board_update(p1, p2, p1k, p2k, pos_init, pos_final, jumped_piece_type, intitial_piece_type);
    }
    
    return total_boards;

}

// print the board to standard out in a human readable format (for debugging)
void human_readble_board(long long p1, long long p2, long long p1k, long long p2k){
    printf("  0 1 2 3 4 5 6 7 \n");
    for (int row = 0; row < 8; row++){
        printf("%d ", row);
        for (int col = 0; col < 8; col++){
            int piece = get_piece_at_location(p1, p2, p1k, p2k, (row * 8) + col);
            if (piece == 0){
                printf("  ");
            }
            else if (piece == 1){
                printf("o ");
            }
            else if (piece == 2){
                printf("x ");
            }
            else if (piece == 3){
                printf("O ");
            }
            else if (piece == 4){
                printf("X ");
            }
        }
        printf("\n");
    }
}

// print the expected line to standard out in a human readable format (for debugging)
void print_line(long long p1, long long p2, long long p1k, long long p2k, int player, unsigned long long hash, struct board_evaler* evaler){
    // use the hash table to make the line of moves
    printf("PV ");
    int depth = 0;
    int forced_pos = -1;
    while (depth < evaler->search_depth) {
        struct hash_table_entry* table_entry = get_hash_entry(evaler->hash_table, hash);
        if (table_entry == NULL || table_entry->best_move == NO_MOVE || TT_NODE_TYPE(table_entry) != PV_NODE || TT_PLAYER(table_entry) != player){
            break;
        }

        int move_start = (table_entry->best_move >> 8) & 0xFF;
        int move_end = table_entry->best_move & 0xFF;

        // convert from 64 bit board to 32 square representation
        int start = (65 - move_start) / 2;
        int end = (65 - move_end) / 2;


        printf("- %d %d ", start, end);

        hash = update_hash(p1, p2, p1k, p2k, move_start, move_end, hash, evaler);
        if (forced_pos >= 0){
            hash ^= evaler->hash_table->piece_hash_diff[FORCED_KEY_OFFSET + forced_pos];
        }
        int initial_piece_type = get_piece_at_location(p1, p2, p1k, p2k, move_start);
        update_board(&p1, &p2, &p1k, &p2k, move_start, move_end);
        int player_next = get_next_board_state(p1, p2, p1k, p2k, move_start, move_end, player, initial_piece_type, evaler->piece_offsets);
        if (player_next == player){
#if USE_FORCED_CONTINUATION
            forced_pos = move_end;
            hash ^= evaler->hash_table->piece_hash_diff[FORCED_KEY_OFFSET + move_end];
#endif
        } else {
            forced_pos = -1;
            hash ^= evaler->hash_table->piece_hash_diff[SIDE_KEY_INDEX];
        }
        player = player_next;
        depth++;
    }
    printf("\n");
    //human_readble_board(p1, p2, p1k, p2k);
}

// main function
// runs a n_ply search to verify the move generation is working as expected
// also benchmarks the time it takes to generate the moves
// int main(){

//     printf("beginning tests\n");

//     // setup initial board and search structures
//     // starting bit values
//     long long p1 = 6172839697753047040;
//     long long p2 = 11163050;
//     long long p1k = 0;
//     long long p2k = 0;

//     // end game player 2 winning endgame
//     //long long p1 = 5838922414443986944;
//     //long long p2 = 8409224;
//     //long long p1k = 0;
//     //long long p2k = 288230376154333184;


//     int player = 1;

//     struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);

//     int* offsets = malloc(sizeof(int) * 64 * 4);
//     compute_offsets(offsets);

//     int depth = 9;

//     clock_t start, end;
//     double cpu_time_used;

//     for (int i = 2; i < depth; i++){
//         // get the start time
//         start = clock();

//         long long n_ply_search_result = n_ply_search(&p1, &p2, &p1k, &p2k, player, piece_loc, offsets, i);

//         // get the end time
//         end = clock();
//         cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

//         // print the results
//         printf("%d ply search result: %lld\n", i, n_ply_search_result);
//         printf("%d ply search time: %f\n\n", i, cpu_time_used);

//     }

//     // now do a search to depth 20 and print the results
//     start_board_search(p1, p2, p1k, p2k, player, 10, 40);

//     return 0;
// }