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

// ---- search feature flags (for A/B testing individual features) ----
#define USE_ASPIRATION 0      // aspiration windows around the previous iteration's score
#define USE_SINGULAR_EXT 0    // forced (single-reply) moves do not consume depth
#define USE_HISTORY 0         // order quiet moves by history heuristic
#define USE_ROOT_REORDER 0    // order root moves by previous iteration's scores
#define USE_FORCED_CONTINUATION 1 // multi-jumps must continue with the same piece (correct rules)
#define USE_ENDGAME_DB 1      // probe the endgame WLD tablebase (db/wld_*.bin) when present

// ---- reduction / extension behavior ----
// building with /D OLD_ENGINE (build_old.bat) keeps the previous behavior so that
// bot_vs_bot.py can measure the elo delta of these fixes head to head.
#ifdef OLD_ENGINE
  #define USE_VERIFIED_LMR 0            // quiet-only late move reductions with a verification re-search
  #define USE_OLD_REDUCTION 1           // blanket late move reduction (reduces captures too, never re-searched)
  #define USE_LEGACY_MATERIAL_REDUCTION 1 // root-relative "-2 ply" reduction, fires inside every exchange
  #define USE_FREE_JUMP_CHAIN 0         // every segment of a multi jump burns a ply of depth
#else
  #define USE_VERIFIED_LMR 1
  #define USE_OLD_REDUCTION 0
  #define USE_LEGACY_MATERIAL_REDUCTION 0
  #define USE_FREE_JUMP_CHAIN 1
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
struct board_data;
struct board_data *board_data_constructor(int player, int move_start, int move_end);
struct set* get_piece_locations(long long p1, long long p2, long long p1k, long long p2k);
void update_piece_locations(int piece_loc_initial, int piece_loc_after, struct set* piece_loc);
void undo_piece_locations_update(int piece_loc_initial, int piece_loc_after, struct set* piece_loc);
void sort_moves(struct board_data* ptr, int player);
int get_next_board_state(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, int player, int piece_type, char* offsets);
int get_piece_at_location(long long p1, long long p2, long long p1k, long long p2k, int pos);
int update_board(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after);
void undo_board_update(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type, int initial_piece_type);
int generate_all_moves(long long p1, long long p2, long long p1k, long long p2k, int player, int* moves, struct set* piece_loc, char* offsets, int* jump);
int generate_moves(long long p1, long long p2, long long p1k, long long p2k, int pos, int* save_loc, char* offsets, int only_jump);
int negmax(long long* p1, long long* p2, long long* p1k, long long* p2k, int player,
    struct set* piece_loc, int depth, int alpha, int beta,
    struct board_evaler* evaler, unsigned long long int hash, int depth_abs, int forced_pos);
struct search_info* start_board_search(long long p1, long long p2, long long p1k, long long p2k, int player, float search_time, int search_depth, int forced_pos);
void human_readble_board(long long p1, long long p2, long long p1k, long long p2k);
long long n_ply_search(long long* p1, long long* p2, long long* p1k, long long* p2k, int player, struct set* piece_loc, char* offsets, int depth);
void quick_sort(struct board_data* ptr, int low, int high);
int partition(struct board_data* ptr, int low, int high);
unsigned long long int update_hash(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, unsigned long long int hash, struct board_evaler* evaler);
struct board_data* get_best_move(struct board_data *head, int player);
void end_board_search(struct board_evaler* evaler);
int free_board_data(struct board_data* data);
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


    // Get some stats about the search
    py_tuple = Py_BuildValue("KKKKi", search_info->evaler->search_depth, search_info->evaler->extended_depth,
                        search_info->evaler->nodes, search_info->evaler->hash_table->num_entries,
                        search_info->eval);
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
// then the remaining (quiet) moves sorted by history-heuristic score
void order_moves(int* moves, int num_moves, struct hash_table_entry* entry, struct killer_entry* killer_entry, long long* history, int is_jump){
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

    // sort the remaining quiet moves by history score (insertion sort, lists are small)
    if (!is_jump && history != NULL){
        for (int i = sorted_index + 1; i < num_moves; i++){
            int s = moves[i * 2];
            int e = moves[i * 2 + 1];
            long long h = history[(s << 6) + e];
            int j = i - 1;
            while (j >= sorted_index && history[(moves[j * 2] << 6) + moves[j * 2 + 1]] < h){
                moves[(j + 1) * 2] = moves[j * 2];
                moves[(j + 1) * 2 + 1] = moves[j * 2 + 1];
                j--;
            }
            moves[(j + 1) * 2] = s;
            moves[(j + 1) * 2 + 1] = e;
        }
    }
}

// reorder the root move list to match the previous iteration's scores (descending).
// returns 1 if every stored move matched the freshly generated list, 0 to fall back to normal ordering
int reorder_root_moves(int* moves, int num_moves, struct search_results* prev){
    int idx[96];
    for (int i = 0; i < num_moves; i++){
        idx[i] = i;
    }
    // insertion sort of the previous iteration's move indices by eval, descending
    for (int i = 1; i < num_moves; i++){
        int id = idx[i];
        int j = i - 1;
        while (j >= 0 && prev->evals[idx[j]] < prev->evals[id]){
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = id;
    }

    int placed = 0;
    for (int r = 0; r < num_moves; r++){
        short mv = prev->moves[idx[r]];
        int start = (mv >> 8) & 0xFF;
        int end = mv & 0xFF;
        for (int j = placed; j < num_moves; j++){
            if (moves[j * 2] == start && moves[j * 2 + 1] == end){
                int ts = moves[placed * 2];
                int te = moves[placed * 2 + 1];
                moves[placed * 2] = moves[j * 2];
                moves[placed * 2 + 1] = moves[j * 2 + 1];
                moves[j * 2] = ts;
                moves[j * 2 + 1] = te;
                placed++;
                break;
            }
        }
    }
    return placed == num_moves;
}

// find the state of the next board after a move
// returns 1 if the moving player is player 2 returns 0 if the player is player 1
// pos is the position the piece will be after the first jump and leading in to the second one
// convenion is that player 1 has internal state of 1 and player 2 has internal state of 2 (ie. 1 is red and 2 is black in a normal match)
int get_next_board_state(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, int player, int piece_type, char* offsets){
    // check if the last move was a non-promoting jump, if it wasn't invert the state
    if ((abs(pos_init - pos_after) < 10) || ((pos_after < 8 || pos_after > 55) && piece_type <= 2)){
        return player ^ 0x3;
    }
    
    // if the last move was a jump see if it can jump again if so do not change the state
    int move_bucket[8];
    int num_moves = generate_moves(p1, p2, p1k, p2k, pos_after, &move_bucket[0], offsets, True);
    if (num_moves != 0){
        return player;
    } else {
        return player ^ 0x3;
    }
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
// returns 0 if no piece was captured
int update_board(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after){
    int piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, piece_loc_initial);
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
    if (abs(piece_loc_initial - piece_loc_after) > 10){
        return_value = get_piece_at_location(*p1, *p2, *p1k, *p2k, (piece_loc_initial + piece_loc_after) / 2);
        if (return_value == 1){
            *p1 = *p1 ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (return_value == 2){
            *p2 = *p2 ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (return_value == 3){
            *p1k = *p1k ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (return_value == 4){
            *p2k = *p2k ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        }
    }

    return return_value;
}

// reverse the a board update
void undo_board_update(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type, int initial_piece_type){
    int piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, piece_loc_after);
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
unsigned long long int update_hash(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, unsigned long long int hash, struct board_evaler* evaler){
    int piece_type = get_piece_at_location(p1, p2, p1k, p2k, pos_init);
    int jumped_piece_type = -1;
    if (abs(pos_init - pos_after) > 10){
        jumped_piece_type = get_piece_at_location(p1, p2, p1k, p2k, (pos_init + pos_after) / 2);
        if(jumped_piece_type == 1){
            hash ^= evaler->hash_table->piece_hash_diff[((pos_init + pos_after)/2)];
        } else if (jumped_piece_type == 2){
            hash ^= evaler->hash_table->piece_hash_diff[((pos_init + pos_after)/2) + 64];
        } else if (jumped_piece_type == 3){
            hash ^= evaler->hash_table->piece_hash_diff[((pos_init + pos_after)/2) + 128];
        } else if (jumped_piece_type == 4){
            hash ^= evaler->hash_table->piece_hash_diff[((pos_init + pos_after)/2) + 192];
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

// generate all possible moves for a given board - 
// takes the board and a memory location to save to as arguments
// returns the number of moves generated
// note: moves should have room for 96 elements as this is the maximum number of moves possible on a legal board
int generate_all_moves(long long p1, long long p2, long long p1k, long long p2k, int player, int* moves, struct set* piece_loc, char* offsets, int* jump){
    // setup variables
    int num_moves = 0;
    int move_from_pos = 0;
    int type;
    int friendly_pieces = 2 + player;
    int num_pieces = populate_set_array(piece_loc);

    // loop over all the locations with a piece and generate the moves for each one
    for (int i = 0; i < num_pieces; i++){
        // check if it is a valid piece
        type = get_piece_at_location(p1, p2, p1k, p2k, piece_loc->array[i]);
        if (type == player || type == friendly_pieces){
            // generate the moves for the piece
            move_from_pos = generate_moves(p1, p2, p1k, p2k, piece_loc->array[i], moves + (num_moves * 2), offsets, *jump);
            if (move_from_pos == -1){
                num_moves = 0;
                *jump = 1;
                move_from_pos = generate_moves(p1, p2, p1k, p2k, piece_loc->array[i], moves + (num_moves * 2), offsets, *jump);
                num_moves += move_from_pos;
            }
            // if no special cases occured add the moves to the counter
            else {
                num_moves += move_from_pos;
            }
        }    
    }
    return num_moves;
}

// generate the move's for a single piece
// takes the board, the piece position, memory location to save, and a jump flag as arguments
// returns the number of moves generated or -1 if a jump was found and the jump flag was not set
// note: save_loc should have room for 8 int
int generate_moves(long long p1, long long p2, long long p1k, long long p2k, int pos, int* save_loc, char* offsets, int only_jump){
    // get the offset index for the position, then initialized some variables
    int offset_index = pos * 4;
    int num_moves = 0;
    int friendly_type = 0;
    int piece_type = get_piece_at_location(p1, p2, p1k, p2k, pos);
    // load offsets into a local variable
    int number_offsets[] = {-9, -7, 7, 9};
    
    // set the friendly piece type
    if (piece_type == 1){
        friendly_type = 3;
    }
    else if (piece_type == 2){
        friendly_type = 4;
    }
    else if (piece_type == 3){
        friendly_type = 1;
    }
    else if (piece_type == 4){
        friendly_type = 2;
    }
    // loop over all the offsets
    int new_pos = pos;
    int temp_pos_type;
    for (int i = 0; i < 4; i++){
        // if it is not a valid offset for that piece continue
        if (offsets[offset_index + i] == 0 || ((piece_type == 2) & (i==0 || i == 1)) || ((piece_type == 1) & (i==2 || i==3))){
            continue;
        }
        // get the data of the potential jump location
        new_pos = pos + number_offsets[i];
        temp_pos_type = get_piece_at_location(p1, p2, p1k, p2k, new_pos);
        // if the location is empty, add it to the list of moves - also check for only jump flag
        if (temp_pos_type == 0 && !only_jump){
            save_loc[num_moves * 2] = pos;
            save_loc[(num_moves * 2) + 1] = new_pos;
            num_moves++;
        // if the location is a friendly piece, continue
        } else if (temp_pos_type == friendly_type || temp_pos_type == piece_type){
            continue;
        // if it is an enemy piece see if the location one further is empty and in that case it is a capture move!
        } else if (temp_pos_type != 0){
            // make sure the offset is valid at the new location
            if (offsets[(new_pos * 4) + i] == 0){
                continue;
            }
            new_pos = new_pos + number_offsets[i];
            temp_pos_type = get_piece_at_location(p1, p2, p1k, p2k, new_pos);

            if (temp_pos_type == 0){
                // if a jump was found and the jump flag was not set, return -1
                if (!only_jump){
                    return -1;
                }
                save_loc[num_moves * 2] = pos;
                save_loc[(num_moves * 2) + 1] = new_pos;
                num_moves++;
            }
        }
    }
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

#if USE_SINGULAR_EXT
    // a singular (forced) reply does not consume depth: the node has exactly one
    // child so the extension is free and lets forced sequences resolve fully
    if (num_moves == 1){
        return depth + 1;
    }
#endif

    // Extract the node type from the table entry
    int node_type = UNKNOWN_NODE;
    if (table_entry != NULL){
        node_type = table_entry->node_type;
    }

    // PV-node extension (only for entries from the current search: the persistent
    // transposition table holds PV labels from older searches too, which would
    // otherwise trigger extensions all over the tree)
    if (node_type == PV_NODE && depth_abs > 8 && table_entry->age == evaler->hash_table->age){
           depth++;
    }

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

    // if nodes is divisible by 10000, check the time
    // (never abort during the first iteration so a best move is always available)
    if (evaler->nodes % 10000 == 0 && evaler->nodes != 0 && evaler->search_depth > 1){
        clock_t current_time = clock();
        double cpu_time_used = ((double)(current_time - evaler->start_time)) / CLOCKS_PER_SEC;
        if (cpu_time_used > evaler->time_limit){
            return INFINITY;

        }
    }

    // check for a draw by repetition FIRST: this must never be masked by a
    // cached transposition entry (cached evals know nothing about repetitions
    // along the current line or in the game so far)
    if (get_draw_entry(evaler->draw_table, hash) == 2){
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
    // notes: no hash cutoffs at the root (the root must always search so the move list gets scored),
    // and no cutoffs from entries of previous searches (their repetition context is stale) -
    // stale entries still provide the hash move for ordering
    struct hash_table_entry* table_entry = get_hash_entry(evaler->hash_table, hash, evaler->search_depth, depth);
    if (table_entry != NULL && table_entry->depth >= depth && table_entry->player == player && depth_abs > 0
        && table_entry->age == evaler->hash_table->age) {
        if (table_entry->node_type == PV_NODE) {
            return adjust_mate_score(table_entry->eval);
        }
        else if (table_entry->node_type == LOWER_BOUND) {
            alpha = max(alpha, table_entry->eval);
        }
        else if (table_entry->node_type == UPPER_BOUND) {
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
#if USE_ROOT_REORDER
    if (depth_abs == 0 && num_moves > 1 && evaler->search_results->num_moves == num_moves){
        root_preordered = reorder_root_moves(&moves[0], num_moves, evaler->search_results);
    }
#endif
    if (num_moves > 1 && !root_preordered){
#if USE_HISTORY
        long long* history = evaler->history + (player - 1) * 4096;
#else
        long long* history = NULL;
#endif
        order_moves(&moves[0], num_moves, table_entry, evaler->killer_table->table + depth_abs,
                    history, is_jump);
    }

    // if there are no move then a player must have won, or there are no captures so end this branch (only count as a win if captures_only is false)
    // this is not a perfect check but it should be good enough
    if (num_moves == 0){
        // if there are no moves and captures only is false then a win has occured eval who won and return
        if (!force_captures){
            return -WIN_SCORE;
        }
        // if not already generating all moves then generate all moves regardless of depth to check for a win
        int captures = 0;
        num_moves = generate_all_moves(*p1, *p2, *p1k, *p2k, player, &moves[0], piece_loc, evaler->piece_offsets, &captures);
        if (num_moves == 0){
            return -WIN_SCORE;
        }

        // if there are no moves and captures only is true then we found the end of a catures only search evaluate the position and return.
        // clamped so a heuristic score can never reach into the proven band and be
        // mistaken for a win - see the score band comment at the top of the file
        int static_eval = get_eval(*p1, *p2, *p1k, *p2k, player, piece_loc, evaler, depth_abs);
        return max(-EVAL_MAX, min(EVAL_MAX, static_eval));
    }

    depth = should_extend_or_reduce(depth, depth_abs, force_captures, is_jump, forced_pos, num_moves, player, *p1 | *p1k, *p2 | *p2k, table_entry, evaler);


    // the the next boards are ready to be searched so begin the search!
    short best_move = NO_MOVE;
    int eval, flip, next_alpha, next_beta;
    for (int i = 0; i < num_moves; i++){
        // prepare moves
        int move_start = moves[i * 2];
        int move_end = moves[(i * 2) + 1];

        // get the hash for this next board (always do this before the move is made on the board)
        next_hash = update_hash(*p1, *p2, *p1k, *p2k, move_start, move_end, hash, evaler);
        if (forced_pos >= 0){
            // leaving this node clears its forced-continuation marker from the hash
            next_hash ^= evaler->hash_table->piece_hash_diff[FORCED_KEY_OFFSET + forced_pos];
        }

        // update the board and set values
        initial_piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, move_start);
        int jumped_piece_type = update_board(p1, p2, p1k, p2k, move_start, move_end);
        update_piece_locations(move_start, move_end, piece_loc);
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
        int child_depth = depth - 1 - reduction;
#if USE_FREE_JUMP_CHAIN
        if (player_next == player){
            child_depth = depth - reduction;
        }
#endif

        eval = negmax(p1, p2, p1k, p2k, player_next, piece_loc, child_depth,
                    next_alpha, next_beta, evaler, next_hash,
                    depth_abs + 1, next_forced) * flip;
#if USE_VERIFIED_LMR
        // the reduced search came back above alpha, so it may be a real improvement:
        // verify it at full depth before it is allowed to change the node's result
        if (reduction > 0 && eval > alpha && eval != INFINITY && eval != -INFINITY){
            eval = negmax(p1, p2, p1k, p2k, player_next, piece_loc, depth - 1,
                        next_alpha, next_beta, evaler, next_hash,
                        depth_abs + 1, next_forced) * flip;
        }
#endif


        // undo the update to the board and piece locations
        undo_piece_locations_update(move_start, move_end, piece_loc);
        undo_board_update(p1, p2, p1k, p2k, move_start, move_end, jumped_piece_type, initial_piece_type);
        remove_draw_entry(evaler->draw_table, next_hash);

        // store results of moves from the root node
        if (depth_abs == 0) {
            evaler->search_results->evals[i] = eval;
            evaler->search_results->moves[i] = (short)((move_start << 8) | move_end);
        }

        // if the eval is infinity the search is trying to end so return
        if (eval == INFINITY || eval == -INFINITY){
            return INFINITY;
        }

        // alpha beta prunning
        if (board_eval < eval){
            board_eval = eval;
            best_move = (move_start << 8) | move_end;
        }
        if (board_eval > alpha){
            alpha = board_eval;
        }
        // prune if a cut off occurs
        if (alpha >= beta){
            if (!is_jump){
                update_killer_table(evaler->killer_table, depth_abs, move_start, move_end);
                if (depth > 0){
                    evaler->history[(player - 1) * 4096 + (move_start << 6) + move_end] += (long long)depth * depth;
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

    // store the eval in the hash table
    add_hash_entry(evaler->hash_table, hash, board_eval, depth, evaler->search_depth, player, best_move, node_type);

    return adjust_mate_score(board_eval);
}

// marks pv nodes in the hash table
void PV_labler(long long* p1, long long* p2, long long* p1k, long long* p2k, int player,
    struct set* piece_loc, int depth, unsigned long long int hash, struct board_evaler* evaler, int forced_pos) {

    struct hash_table_entry* table_entry = get_hash_entry(evaler->hash_table, hash, evaler->search_depth, depth);
    if (table_entry == NULL || table_entry->best_move == NO_MOVE || depth <= 0 || table_entry->player != player) {
        return;
    }

    table_entry->node_type = PV_NODE;

    short move_start = (table_entry->best_move >> 8) & 0xFF;
    short move_end = table_entry->best_move & 0xFF;

    hash = update_hash(*p1, *p2, *p1k, *p2k, move_start, move_end, hash, evaler);
    if (forced_pos >= 0){
        hash ^= evaler->hash_table->piece_hash_diff[FORCED_KEY_OFFSET + forced_pos];
    }
    int initial_piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, move_start);
    int jumped_piece_type = update_board(p1, p2, p1k, p2k, move_start, move_end);
    update_piece_locations(move_start, move_end, piece_loc);
    int player_next = get_next_board_state(*p1, *p2, *p1k, *p2k, move_start, move_end, player, initial_piece_type, evaler->piece_offsets);
    int next_forced = -1;
    if (player_next == player){
#if USE_FORCED_CONTINUATION
        next_forced = move_end;
        hash ^= evaler->hash_table->piece_hash_diff[FORCED_KEY_OFFSET + move_end];
#endif
    } else {
        hash ^= evaler->hash_table->piece_hash_diff[SIDE_KEY_INDEX];
    }

    PV_labler(p1, p2, p1k, p2k, player_next, piece_loc, depth - 1, hash, evaler, next_forced);

    undo_piece_locations_update(move_start, move_end, piece_loc);
    undo_board_update(p1, p2, p1k, p2k, move_start, move_end, jumped_piece_type, initial_piece_type);
}

int MTDF(long long* p1, long long* p2, long long* p1k, long long* p2k, int player,
    struct set* piece_loc, int depth, int f, struct board_evaler* evaler,
    unsigned long long int hash) {

    int g = f;
    int upper_bound = INFINITY;
    int lower_bound = -INFINITY;



    while (lower_bound < upper_bound) {
        int beta = (g == lower_bound) ? g + 1 : g;

        g = negmax(p1, p2, p1k, p2k, player, piece_loc, depth, beta - 1, beta,
            evaler, hash, 0, -1);

        if (g < beta) {
            upper_bound = g;
        } else {
            lower_bound = g;
        }
    }

    // follow the pv line labeling each node as a PV node
    PV_labler(p1, p2, p1k, p2k, player, piece_loc, depth, hash, evaler, -1);

    return g;
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
    // mid multi-jump roots are transient states, not repeatable positions
    if (forced_pos < 0 && GAME_HISTORY_LEN < GAME_HISTORY_MAX){
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
    int eval_ = 0;
    int last_eval = 0;
    int have_result = 0;
    int jump = 0;

    // iterative deepening with an aspiration window around the previous score
    for (int i = 1; i <= search_depth; i++){
        // update the evalers search depth
        evaler->search_depth = i;
        evaler->max_depth = min(max(i + 10, 5), search_depth);

        int asp_alpha = -INFINITY;
        int asp_beta = INFINITY;
#if USE_ASPIRATION
        if (i >= 4 && have_result && last_eval < WIN_MIN && last_eval > -WIN_MIN){
            asp_alpha = last_eval - 40;
            asp_beta = last_eval + 40;
        }
#endif

        eval_ = negmax(&p1, &p2, &p1k, &p2k, player, piece_loc, i, asp_alpha, asp_beta, evaler, hash, 0, forced_pos);

        // the window failed: re-search with the full window
        if (eval_ != INFINITY && eval_ != -INFINITY && (eval_ <= asp_alpha || eval_ >= asp_beta)){
            eval_ = negmax(&p1, &p2, &p1k, &p2k, player, piece_loc, i, -INFINITY, INFINITY, evaler, hash, 0, forced_pos);
        }

        // get the end time
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        // if the eval is infinity the search ran out of time; keep the last completed iteration's results
        if (eval_ == INFINITY || eval_ == -INFINITY){
            evaler->search_depth = max(depth, 1);
            evaler->extended_depth = extended_depth;
            break;
        }

        last_eval = eval_;
        have_result = 1;
        depth = i;
        extended_depth = evaler->extended_depth;
        if (depth > extended_depth){
            extended_depth = depth;
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
        printf("HashTable Hit ratio: %lld\n", (evaler->hash_table->hit_count * 100) / (evaler->hash_table->hit_count + evaler->hash_table->miss_count));
        printf("HashTable Usage: %lld\n", (evaler->hash_table->num_entries * 100llu) / evaler->hash_table->total_size);
        printf("Nodes/s: %fM\n", round_float(((double)evaler->nodes / (cpu_time_used + 0.01)) / 1000000.0));
        printf("Time: %fs\n", cpu_time_used);
        printf("Depth: %d\n", evaler->extended_depth);
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

    // generate the moves for the current board
    int num_moves = generate_all_moves(*p1, *p2, *p1k, *p2k, player, &moves[0], piece_loc, offsets, False);

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
        struct hash_table_entry* table_entry = get_hash_entry(evaler->hash_table, hash, evaler->search_depth, depth);
        if (table_entry == NULL || table_entry->best_move == NO_MOVE || table_entry->node_type != PV_NODE || table_entry->player != player){
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