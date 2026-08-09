// Browser binding for the search.
//
// Structurally the twin of the CPython block in board_search.c: it is the only
// place that knows about the host, it lives entirely inside one #ifdef, and it
// puts not a single instruction into the .pyd. The two can be read side by side
// and should stay that way - if search_position grows a field, so does this.
//
// This file is #included from board_search.c (it is not a translation unit of
// its own, like every other .c in this engine). Package_engine.py globs
// src/engine/*.c for its dependency list but compiles only board_search.c, so
// editing this file correctly triggers a .pyd rebuild while never being
// compiled into one.

#include <emscripten/emscripten.h>
#include <stdint.h>

// One search's results, in the layout the worker reads out of HEAP32.
//
// The two 64 bit counters are split into halves rather than exported as i64
// because JS only ever formats them for display, and a BigInt at the boundary
// for a number that ends up as text is friction with no payoff. Fields are
// append-only for the same reason the Python stats tuple is: the reader indexes
// positionally.
//
//   [0] move from 0..63    [4] max ply          [8]  tt entries high
//   [1] move to   0..63    [5] nodes low        [9]  evals low
//   [2] eval (stm view)    [6] nodes high       [10] evals high
//   [3] nominal depth      [7] tt entries low   [11] extended depth
#define WASM_RESULT_FIELDS 12
static int32_t WASM_RESULT[WASM_RESULT_FIELDS];

EMSCRIPTEN_KEEPALIVE
int32_t* wasm_result_ptr(void){
    return WASM_RESULT;
}

EMSCRIPTEN_KEEPALIVE
int wasm_result_fields(void){
    return WASM_RESULT_FIELDS;
}

// The whole engine surface the GUI needs, in one call. Mirrors search_position
// argument for argument and field for field, including the teardown, so that
// the two host bindings can be diffed by eye.
//
// The packed move is also the return value, so the common case needs no heap
// access at all. That matters more than it looks: under ALLOW_MEMORY_GROWTH a
// cached HEAP32 view is detached by any allocation, and a search allocates.
//
// forced_pos is the square 0..63 of a piece mid multi-jump that must keep
// jumping, or -1 for a normal position - same contract as the Python entry.
EMSCRIPTEN_KEEPALIVE
int wasm_search(unsigned long long p1, unsigned long long p2,
                unsigned long long p1k, unsigned long long p2k,
                int player, double search_time, int search_depth,
                int forced_pos){
    struct search_info* si = start_board_search((long long)p1, (long long)p2,
                                                (long long)p1k, (long long)p2k,
                                                player, (float)search_time,
                                                search_depth, forced_pos);

    const int packed = si->best_move;
    const unsigned long long nodes = (unsigned long long)si->evaler->nodes;
    const unsigned long long entries = (unsigned long long)si->evaler->hash_table->num_entries;
    const unsigned long long evals = (unsigned long long)si->evaler->evals;

    WASM_RESULT[0]  = (packed >> 8) & 0xFF;
    WASM_RESULT[1]  = packed & 0xFF;
    WASM_RESULT[2]  = si->eval;
    WASM_RESULT[3]  = si->evaler->search_depth;
    WASM_RESULT[4]  = si->evaler->max_ply;
    WASM_RESULT[5]  = (int32_t)(nodes & 0xFFFFFFFFu);
    WASM_RESULT[6]  = (int32_t)(nodes >> 32);
    WASM_RESULT[7]  = (int32_t)(entries & 0xFFFFFFFFu);
    WASM_RESULT[8]  = (int32_t)(entries >> 32);
    WASM_RESULT[9]  = (int32_t)(evals & 0xFFFFFFFFu);
    WASM_RESULT[10] = (int32_t)(evals >> 32);
    WASM_RESULT[11] = si->evaler->extended_depth;

    end_board_search(si->evaler);
    free(si);

    return packed & 0xFFFF;
}

// perft, so the browser build is held to exactly the same move generation
// standard as the .pyd rather than being taken on trust.
//
// Returns double: exact to 2^53, which is far past any perft that finishes
// while someone is waiting, and it keeps i64 out of the boundary.
EMSCRIPTEN_KEEPALIVE
double wasm_perft(unsigned long long p1, unsigned long long p2,
                  unsigned long long p1k, unsigned long long p2k,
                  int player, int depth){
    long long lp1 = (long long)p1, lp2 = (long long)p2;
    long long lp1k = (long long)p1k, lp2k = (long long)p2k;
    struct set* piece_loc = get_piece_locations(lp1, lp2, lp1k, lp2k);
    char* offsets = compute_offsets();

    long long total = n_ply_search(&lp1, &lp2, &lp1k, &lp2k, player,
                                   piece_loc, offsets, depth);

    free(offsets);
    free(piece_loc);
    return (double)total;
}

// static evaluation, the counterpart of nnue_eval - the browser half of the
// bit-exactness check against export.int_forward()
EMSCRIPTEN_KEEPALIVE
int wasm_nnue_eval(unsigned long long p1, unsigned long long p2,
                   unsigned long long p1k, unsigned long long p2k, int stm){
    return nnue_eval_position((long long)p1, (long long)p2,
                              (long long)p1k, (long long)p2k, stm);
}

// which vector path this artifact was actually built with: 2 AVX2, 1 SSE2 (the
// one wasm SIMD128 selects), 0 scalar. The browser equivalent of
// nnue_info()["simd"], and the only way to tell from outside whether the SIMD
// build or the fallback is the one that got loaded.
EMSCRIPTEN_KEEPALIVE
int wasm_nnue_simd(void){
    return NNUE_SIMD;
}

// largest total piece count the tablebase currently in memory can answer for.
// Grows as slices arrive, so the host can tell whether a tier finished landing.
EMSCRIPTEN_KEEPALIVE
int wasm_db_max_pieces(void){
#if USE_ENDGAME_DB
    return endgame_db_max_pieces();
#else
    return 0;
#endif
}

// Progress out of the iterative deepening loop: what replaces the "\rPLY: ..."
// line PRINT_OUTPUT wrote to a terminal that, in a browser, does not exist.
// Posted straight out of the worker so the UI can show the search getting
// deeper rather than a spinner that means nothing.
//
// Declared with EM_JS rather than called through a function pointer because the
// call site in board_search.c is inside the search's hot outer loop and this
// keeps the whole mechanism to one line there.
EM_JS(void, wasm_on_iteration, (int depth, int ext, int eval), {
    if (typeof postMessage === 'function'){
        postMessage({ type: 'progress', depth: depth, ext: ext, eval: eval });
    }
});
