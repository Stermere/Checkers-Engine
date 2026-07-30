// Author: Collin Kees
// Fixed point NNUE evaluation network.
//
// This is the C side of src/python/nnue/. It must reproduce
// `export.int_forward()` **bit exactly** - not approximately - because the
// whole point of the quantized export is that the engine and the training
// pipeline agree on what the net says. src/python/nnue/verify_nnue_c.py is the
// test that holds this file to that standard.
//
// The contract, copied from export.py:
//
//   accumulator / activations : int16, units of 1/127   (QUANT_ACT)
//   layer weights             : int16, units of 1/64    (QUANT_W)
//
//   acc = sum of active feature rows + ft_bias        units 1/127
//   acc = clamp(acc, 0, 127)
//   h   = acc . h_weight + h_bias                     units 1/(127*64)
//   h   = (h + 32) >> 6                               back to units 1/127
//   h   = clamp(h, 0, 127)
//   o   = h . o_weight + o_bias                       units 1/(127*64)
//   cp  = o * 120 / (127*64)                          truncated toward zero
//
// Everything below is integer arithmetic, so vectorising it does not change a
// single result: integer addition is associative and exact, and the bounds
// checked in `nnue_static_checks` below guarantee nothing overflows the width
// it is summed in. That is what lets this file be fast and still be tested by
// exact equality.
//
// Three details are easy to get wrong and all three are deliberate here:
//
//   * `(h + 32) >> 6`, never `/ 64`. The numpy reference uses floor division;
//     C integer division truncates toward zero, which differs for negative
//     values. An arithmetic right shift floors, so it matches. Hidden
//     pre-activations are negative often enough that `/ 64` would corrupt a
//     large fraction of evaluations.
//
//   * the final centipawn conversion is done in int64, not in floating point.
//     numpy computes `o * 120.0 / 8128.0` in float64 and then truncates. The
//     exact rational o*120/8128 and its float64 rounding can only disagree if
//     the true value sits within ~1e-12 of an integer without being one, which
//     is impossible for a rational with denominator 8128 (the nearest
//     non-integer is 1/8128 away). So integer arithmetic is exactly equal to
//     the reference and, unlike float math, is immune to /fp:fast.
//
//   * the accumulator is int16, not int32, so that a feature row is added a
//     whole vector at a time and twice as many lanes fit per instruction.
//     Safe only because export.py emits NNUE_ACC_ABS_MAX, the worst case any
//     lane can reach over every subset of features; the compile time check
//     below fails loudly if a retrained net ever breaks that.
//
// ---------------------------------------------------------------------------
// Why there are two ways to evaluate a position
// ---------------------------------------------------------------------------
// `nnue_eval_position()` is the stateless one: encode the board, sum the rows
// of its <=24 active features, propagate. It is what the Python test hooks call
// and what the engine falls back to, and it is the definition of the function.
//
// The search does not use it. A position and its child differ by one moved
// piece and at most one captured piece, so the child's accumulator is the
// parent's minus two rows plus one - three row operations instead of twenty
// four. `struct nnue_stack` keeps one accumulator per ply and does exactly
// that. This is the "incremental update" that the NN in NNUE is named after,
// and it is what makes a wider first layer affordable: its cost per node is
// independent of how many pieces are on the board.
//
// Two things make it more subtle here than in a chess engine:
//
//   * the encoding is side-to-move relative (there is no side-to-move input;
//     the eval is antisymmetric by construction), so "the accumulator" is not
//     one vector but two - the position seen from player 1's chair and from
//     player 2's. Both are updated on every move and the eval reads whichever
//     one matches the side to move. This is the standard NNUE two-perspective
//     arrangement; it just falls out of a different property here.
//
//   * updates are lazy. A node is entered far more often than it is evaluated
//     (a transposition hit, a database hit or a repetition draw all return
//     before reaching the eval), so a push only records *what* changed. The
//     accumulator itself is materialised on the first eval that needs it, by
//     walking back to the deepest ply that already has one. Nodes that never
//     evaluate never pay.

#ifndef NNUE_C
#define NNUE_C

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Build with /D NNUE_WEIGHTS_CANDIDATE=1 to link against
// nnue_weights_candidate.h instead of the committed header. That file is
// gitignored and is what src/python/nnue/match_net.py writes, so a search that
// exports and plays hundreds of nets never touches the tracked champion
// weights - the working tree stays clean and a killed run leaves nothing to
// restore. Deploying a winner is then an explicit copy, not a side effect.
#ifdef NNUE_WEIGHTS_CANDIDATE
#include "nnue_weights_candidate.h"
#else
#include "nnue_weights.h"
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifndef NNUE_ACC_ABS_MAX
#error "nnue_weights.h predates the incremental accumulator - regenerate it: python src/python/nnue/export.py --out src/engine/nnue_weights.h"
#endif
#if NNUE_ACC_ABS_MAX > 32767
#error "the accumulator no longer fits in int16 - retrain with smaller feature weights, or widen it to int32 here and in export.py"
#endif
#if NNUE_QUANT_W != 64
#error "the >> 6 that rescales the hidden layer assumes QUANT_W == 64"
#endif
#define NNUE_QUANT_W_SHIFT 6

// Build with /D NNUE_VERIFY_INCREMENTAL=1 to have every incremental evaluation
// re-checked against the from-scratch one. That is the gate for the delta
// logic: verify_nnue_c.py can only reach the stateless path, so without this
// nothing would notice a wrong feature index in nnue_stack_push until the
// engine quietly played worse. Counters are read from Python by
// nnue_debug_counters(); see src/python/nnue/verify_nnue_incremental.py.
#ifndef NNUE_VERIFY_INCREMENTAL
#define NNUE_VERIFY_INCREMENTAL 0
#endif

// Build with /D NNUE_INCREMENTAL=0 to make the search evaluate every leaf from
// scratch instead. Whether incremental updates pay for themselves depends on
// how large a fraction of nodes are actually evaluated - the update happens at
// every node, the saving only at evaluated ones - so this is here to be
// measured rather than assumed, and it is the honest way to attribute a speedup
// between vectorising the arithmetic and not repeating it.
#ifndef NNUE_INCREMENTAL
#define NNUE_INCREMENTAL 1
#endif

// ---------------------------------------------------------------------------
// SIMD selection
// ---------------------------------------------------------------------------
// Everything is written once against a tiny vector vocabulary and instantiated
// for whichever width the compiler was told it may use. SSE2 is unconditional
// on x86-64, so the scalar path exists only for other architectures.
// /arch:AVX2 (see src/python/Package_engine.py) is what turns on the 256 bit
// version; without it the code still builds and still passes the bit-exact
// test, just at half the lanes.
// /D NNUE_NO_SIMD forces the scalar path. It is the only way to reach it on
// x86-64, where SSE2 is part of the architecture and cannot be switched off, so
// without it the scalar fallback would be code that nothing has ever compiled -
// which is the same as code that does not work. It is also the switch to use on
// a target with no vector support at all.
#if defined(NNUE_NO_SIMD)
  #define NNUE_SIMD 0
#elif defined(__AVX2__)
  #define NNUE_SIMD 2
  #include <immintrin.h>
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
  #define NNUE_SIMD 1
  #include <emmintrin.h>
#else
  #define NNUE_SIMD 0
#endif

#if NNUE_SIMD == 2
typedef __m256i nnue_vec;
typedef __m256i nnue_vec32;
#define NNUE_LANES 16
#define nv_load(p)     _mm256_loadu_si256((const __m256i*)(const void*)(p))
#define nv_store(p, v) _mm256_storeu_si256((__m256i*)(void*)(p), (v))
#define nv_add(a, b)   _mm256_add_epi16((a), (b))
#define nv_sub(a, b)   _mm256_sub_epi16((a), (b))
#define nv_min(a, b)   _mm256_min_epi16((a), (b))
#define nv_max(a, b)   _mm256_max_epi16((a), (b))
#define nv_set1(x)     _mm256_set1_epi16((short)(x))
#define nv_madd(a, b)  _mm256_madd_epi16((a), (b))
#define nv32_zero()    _mm256_setzero_si256()
#define nv32_add(a, b) _mm256_add_epi32((a), (b))
static inline int32_t nv32_hsum(nnue_vec32 v){
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}
#elif NNUE_SIMD == 1
typedef __m128i nnue_vec;
typedef __m128i nnue_vec32;
#define NNUE_LANES 8
#define nv_load(p)     _mm_loadu_si128((const __m128i*)(const void*)(p))
#define nv_store(p, v) _mm_storeu_si128((__m128i*)(void*)(p), (v))
#define nv_add(a, b)   _mm_add_epi16((a), (b))
#define nv_sub(a, b)   _mm_sub_epi16((a), (b))
#define nv_min(a, b)   _mm_min_epi16((a), (b))
#define nv_max(a, b)   _mm_max_epi16((a), (b))
#define nv_set1(x)     _mm_set1_epi16((short)(x))
#define nv_madd(a, b)  _mm_madd_epi16((a), (b))
#define nv32_zero()    _mm_setzero_si128()
#define nv32_add(a, b) _mm_add_epi32((a), (b))
static inline int32_t nv32_hsum(nnue_vec32 v){
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(v);
}
#endif

// How much of the accumulator is held in registers at once while feature rows
// are streamed past it. 8 vectors is 128 int16 under AVX2 and leaves half the
// register file free; the tiling only applies when L1 divides evenly, which
// every power-of-two width does.
#define NNUE_REGS 8

#if NNUE_SIMD
  #if (NNUE_L1 % NNUE_LANES) != 0
    // a first layer that is not a whole number of vectors is not worth a masked
    // tail path; drop to scalar rather than be subtly wrong
    #undef NNUE_SIMD
    #define NNUE_SIMD 0
    #define NNUE_TILED 0
  #else
    #define NNUE_TILE (NNUE_REGS * NNUE_LANES)
    #define NNUE_TILED ((NNUE_L1 % NNUE_TILE) == 0)
  #endif
#else
  #define NNUE_TILED 0
#endif

// ---------------------------------------------------------------------------
// feature layout - mirrors src/python/nnue/features.py exactly
// ---------------------------------------------------------------------------
#define NNUE_NUM_SQ32 32
#define NNUE_NUM_MAN_SQ 28

#define NNUE_OWN_MEN_OFFSET 0
#define NNUE_OWN_KING_OFFSET (NNUE_OWN_MEN_OFFSET + NNUE_NUM_MAN_SQ)   // 28
#define NNUE_ENEMY_MEN_OFFSET (NNUE_OWN_KING_OFFSET + NNUE_NUM_SQ32)   // 60
#define NNUE_ENEMY_KING_OFFSET (NNUE_ENEMY_MEN_OFFSET + NNUE_NUM_MAN_SQ) // 88

// the accumulator sums whole rows of the transformer, so a feature's row is
// nnue_ft_weight[index * NNUE_L1 .. index * NNUE_L1 + NNUE_L1)
#define NNUE_FT_ROW(i) (nnue_ft_weight + (size_t)(i) * NNUE_L1)

// The generated header states the scale as a float (120.0f). Taking it as an
// integer here keeps the final conversion in integer arithmetic; it is derived
// from the header rather than hardcoded so a retrained net cannot leave this
// file behind. If EVAL_SCALE ever becomes non-integral in model.py, the
// conversion below has to change with it.
#define NNUE_EVAL_SCALE_INT ((int64_t)(NNUE_EVAL_SCALE))
#define NNUE_CP_DIVISOR ((int64_t)NNUE_QUANT_ACT * (int64_t)NNUE_QUANT_W)


// index of the lowest set bit
static inline int nnue_ctz64(unsigned long long x){
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (int)idx;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int n = 0;
    while (!(x & 1ull)){ x >>= 1; n++; }
    return n;
#endif
}

// bit64 -> sq32, dark squares only: sq32 = (bit >> 3) * 4 + ((bit & 7) >> 1)
static inline int nnue_bit_to_sq32(int bit){
    return (bit >> 3) * 4 + ((bit & 7) >> 1);
}

// The feature a single piece contributes, seen from `persp` (1 or 2).
// Piece type codes are the engine's: 1 = p1 man, 2 = p2 man, 3 = p1 king,
// 4 = p2 king. This is the same mapping nnue_encode() performs in bulk, and the
// incremental path is the only caller - keeping one definition of it is what
// stops the two paths from drifting.
static inline int nnue_feature_index(int type, int bit, int persp){
    int sq = nnue_bit_to_sq32(bit);
    int is_king = (type >= 3);
    int color = (type == 1 || type == 3) ? 1 : 2;
    if (persp == 2) sq = 31 - sq;
    if (color == persp){
        // own men live on sq32 4..31 (a man on row 0 of its own side promoted)
        assert((is_king || sq >= 4) && "own man on its own promotion row");
        return is_king ? (NNUE_OWN_KING_OFFSET + sq) : (NNUE_OWN_MEN_OFFSET + sq - 4);
    }
    assert((is_king || sq < NNUE_NUM_MAN_SQ) && "enemy man on its own promotion row");
    return is_king ? (NNUE_ENEMY_KING_OFFSET + sq) : (NNUE_ENEMY_MEN_OFFSET + sq);
}

// ---------------------------------------------------------------------------
// encoding
// ---------------------------------------------------------------------------
// Fill `out` with the active feature indices for the position, from the side to
// move's point of view, and return how many there are (at most NNUE_MAX_ACTIVE).
//
// When it is player 2's turn the board is rotated 180 degrees and the colors
// swapped, which on dark squares is exactly sq32 -> 31 - sq32. There is no
// side-to-move input; the eval is antisymmetric by construction.
//
// Men on their own promotion row are dropped rather than encoded. That is not a
// silent failure mode invented here - it is what the numpy encoder does (it
// slices the man planes to 28 squares), so dropping keeps C and Python in
// agreement. Such a piece is illegal anyway (it would have promoted), and
// gen_db_data.py did find malformed tablebase entries, so the case is asserted
// in debug builds and handled defensively in release ones.
static int nnue_encode(long long p1, long long p2, long long p1k, long long p2k,
                       int stm, int* out){
    unsigned long long own_m, own_k, enemy_m, enemy_k;
    int flip = (stm == 2);
    int n = 0;

    if (flip){
        own_m = (unsigned long long)p2;
        own_k = (unsigned long long)p2k;
        enemy_m = (unsigned long long)p1;
        enemy_k = (unsigned long long)p1k;
    } else {
        own_m = (unsigned long long)p1;
        own_k = (unsigned long long)p1k;
        enemy_m = (unsigned long long)p2;
        enemy_k = (unsigned long long)p2k;
    }

    // own men: sq32 4..31, index sq32 - 4
    while (own_m){
        int sq = nnue_bit_to_sq32(nnue_ctz64(own_m));
        own_m &= own_m - 1;
        if (flip) sq = 31 - sq;
        assert(sq >= 4 && "own man on its own promotion row");
        if (sq >= 4 && n < NNUE_MAX_ACTIVE) out[n++] = NNUE_OWN_MEN_OFFSET + sq - 4;
    }
    // own kings: sq32 0..31
    while (own_k){
        int sq = nnue_bit_to_sq32(nnue_ctz64(own_k));
        own_k &= own_k - 1;
        if (flip) sq = 31 - sq;
        if (n < NNUE_MAX_ACTIVE) out[n++] = NNUE_OWN_KING_OFFSET + sq;
    }
    // enemy men: sq32 0..27, index sq32
    while (enemy_m){
        int sq = nnue_bit_to_sq32(nnue_ctz64(enemy_m));
        enemy_m &= enemy_m - 1;
        if (flip) sq = 31 - sq;
        assert(sq < NNUE_NUM_MAN_SQ && "enemy man on its own promotion row");
        if (sq < NNUE_NUM_MAN_SQ && n < NNUE_MAX_ACTIVE) out[n++] = NNUE_ENEMY_MEN_OFFSET + sq;
    }
    // enemy kings: sq32 0..31
    while (enemy_k){
        int sq = nnue_bit_to_sq32(nnue_ctz64(enemy_k));
        enemy_k &= enemy_k - 1;
        if (flip) sq = 31 - sq;
        if (n < NNUE_MAX_ACTIVE) out[n++] = NNUE_ENEMY_KING_OFFSET + sq;
    }

    return n;
}

// ---------------------------------------------------------------------------
// the accumulator
// ---------------------------------------------------------------------------
// One vector of int16 per NNUE_LANES columns of the first layer. Both kernels
// below hold a tile of the accumulator in registers and stream feature rows
// past it, so the accumulator is written to memory exactly once.

// acc = ft_bias + sum of the rows of `count` features. O(pieces).
static void nnue_acc_refresh(int16_t* acc, const int* idx, int count){
#if NNUE_SIMD && NNUE_TILED
    int t, j, k;
    for (t = 0; t < NNUE_L1; t += NNUE_TILE){
        nnue_vec r[NNUE_REGS];
        for (k = 0; k < NNUE_REGS; k++) r[k] = nv_load(nnue_ft_bias + t + k * NNUE_LANES);
        for (j = 0; j < count; j++){
            const int16_t* row = NNUE_FT_ROW(idx[j]) + t;
            for (k = 0; k < NNUE_REGS; k++) r[k] = nv_add(r[k], nv_load(row + k * NNUE_LANES));
        }
        for (k = 0; k < NNUE_REGS; k++) nv_store(acc + t + k * NNUE_LANES, r[k]);
    }
#elif NNUE_SIMD
    int i, j;
    for (i = 0; i < NNUE_L1; i += NNUE_LANES) nv_store(acc + i, nv_load(nnue_ft_bias + i));
    for (j = 0; j < count; j++){
        const int16_t* row = NNUE_FT_ROW(idx[j]);
        for (i = 0; i < NNUE_L1; i += NNUE_LANES)
            nv_store(acc + i, nv_add(nv_load(acc + i), nv_load(row + i)));
    }
#else
    int i, j;
    for (i = 0; i < NNUE_L1; i++) acc[i] = nnue_ft_bias[i];
    for (j = 0; j < count; j++){
        const int16_t* row = NNUE_FT_ROW(idx[j]);
        for (i = 0; i < NNUE_L1; i++) acc[i] = (int16_t)(acc[i] + row[i]);
    }
#endif
}

// dst = src - rows(sub) + rows(add). O(pieces that moved), which is the whole
// point. `dst` and `src` must not overlap.
static void nnue_acc_update(int16_t* dst, const int16_t* src,
                            const int16_t* sub, int n_sub,
                            const int16_t* add, int n_add){
#if NNUE_SIMD && NNUE_TILED
    int t, j, k;
    for (t = 0; t < NNUE_L1; t += NNUE_TILE){
        nnue_vec r[NNUE_REGS];
        for (k = 0; k < NNUE_REGS; k++) r[k] = nv_load(src + t + k * NNUE_LANES);
        for (j = 0; j < n_sub; j++){
            const int16_t* row = NNUE_FT_ROW(sub[j]) + t;
            for (k = 0; k < NNUE_REGS; k++) r[k] = nv_sub(r[k], nv_load(row + k * NNUE_LANES));
        }
        for (j = 0; j < n_add; j++){
            const int16_t* row = NNUE_FT_ROW(add[j]) + t;
            for (k = 0; k < NNUE_REGS; k++) r[k] = nv_add(r[k], nv_load(row + k * NNUE_LANES));
        }
        for (k = 0; k < NNUE_REGS; k++) nv_store(dst + t + k * NNUE_LANES, r[k]);
    }
#elif NNUE_SIMD
    int i, j;
    for (i = 0; i < NNUE_L1; i += NNUE_LANES) nv_store(dst + i, nv_load(src + i));
    for (j = 0; j < n_sub; j++){
        const int16_t* row = NNUE_FT_ROW(sub[j]);
        for (i = 0; i < NNUE_L1; i += NNUE_LANES)
            nv_store(dst + i, nv_sub(nv_load(dst + i), nv_load(row + i)));
    }
    for (j = 0; j < n_add; j++){
        const int16_t* row = NNUE_FT_ROW(add[j]);
        for (i = 0; i < NNUE_L1; i += NNUE_LANES)
            nv_store(dst + i, nv_add(nv_load(dst + i), nv_load(row + i)));
    }
#else
    int i, j;
    memcpy(dst, src, sizeof(int16_t) * NNUE_L1);
    for (j = 0; j < n_sub; j++){
        const int16_t* row = NNUE_FT_ROW(sub[j]);
        for (i = 0; i < NNUE_L1; i++) dst[i] = (int16_t)(dst[i] - row[i]);
    }
    for (j = 0; j < n_add; j++){
        const int16_t* row = NNUE_FT_ROW(add[j]);
        for (i = 0; i < NNUE_L1; i++) dst[i] = (int16_t)(dst[i] + row[i]);
    }
#endif
}

// ---------------------------------------------------------------------------
// the rest of the forward pass
// ---------------------------------------------------------------------------
// accumulator -> centipawns from the side to move's point of view.
// This is the part whose cost does not shrink with incremental updates, so it
// is the one that sets how wide L1 can usefully get.
static int nnue_propagate(const int16_t* acc){
    NNUE_ALIGN64 int16_t clipped[NNUE_L1];
    int16_t hidden[NNUE_L2];
    int32_t o;
    int i;

#if NNUE_SIMD
    {
        const nnue_vec lo = nv_set1(0);
        const nnue_vec hi = nv_set1(NNUE_QUANT_ACT);
        for (i = 0; i < NNUE_L1; i += NNUE_LANES)
            nv_store(clipped + i, nv_min(nv_max(nv_load(acc + i), lo), hi));
    }
#else
    for (i = 0; i < NNUE_L1; i++){
        int32_t v = acc[i];
        if (v < 0) v = 0;
        else if (v > NNUE_QUANT_ACT) v = NNUE_QUANT_ACT;
        clipped[i] = (int16_t)v;
    }
#endif

    // hidden layer. Activations are in [0,127] and weights fit int16, so a
    // 128-term int32 dot product cannot come close to overflowing - which is
    // why pairwise madd accumulation gives exactly the reference's total.
    for (i = 0; i < NNUE_L2; i++){
        const int16_t* w = nnue_h_weight + (size_t)i * NNUE_L1;
        int32_t sum;
#if NNUE_SIMD
        {
            nnue_vec32 s = nv32_zero();
            int j;
            for (j = 0; j < NNUE_L1; j += NNUE_LANES)
                s = nv32_add(s, nv_madd(nv_load(clipped + j), nv_load(w + j)));
            sum = nnue_h_bias[i] + nv32_hsum(s);
        }
#else
        {
            int j;
            sum = nnue_h_bias[i];
            for (j = 0; j < NNUE_L1; j++) sum += (int32_t)clipped[j] * (int32_t)w[j];
        }
#endif
        // units 1/(127*64) -> 1/127, rounding half up via floor((x+32)/64).
        // >> 6 and not / 64: the shift floors for negative sums, / 64 does not.
        sum = (sum + (NNUE_QUANT_W / 2)) >> NNUE_QUANT_W_SHIFT;
        if (sum < 0) sum = 0;
        else if (sum > NNUE_QUANT_ACT) sum = NNUE_QUANT_ACT;
        hidden[i] = (int16_t)sum;
    }

    // output layer - a single neuron over L2 terms, not worth vectorising
    o = nnue_o_bias[0];
    for (i = 0; i < NNUE_L2; i++) o += (int32_t)hidden[i] * (int32_t)nnue_o_weight[i];

    // logit -> centipawns, truncated toward zero (numpy's .astype(np.int32)).
    // int64 keeps this exact and independent of the FP model.
    return (int)(((int64_t)o * NNUE_EVAL_SCALE_INT) / NNUE_CP_DIVISOR);
}

// evaluate a position from scratch. The result is **side-to-move relative**
// (positive means the player about to move is better), which is exactly what
// negamax wants and is also the convention the search's own eval uses after
// get_eval's sign flip.
static int nnue_eval_position(long long p1, long long p2, long long p1k, long long p2k,
                              int stm){
    NNUE_ALIGN64 int16_t acc[NNUE_L1];
    int indices[NNUE_MAX_ACTIVE];
    int n = nnue_encode(p1, p2, p1k, p2k, stm, indices);
    nnue_acc_refresh(acc, indices, n);
    return nnue_propagate(acc);
}

// ---------------------------------------------------------------------------
// the incremental accumulator stack
// ---------------------------------------------------------------------------
// One entry per search ply, holding both perspectives. `delta[d]` describes how
// the board at ply d differs from the board at ply d-1; `computed[d]` says
// whether `acc[d]` has actually been brought up to date yet.

// A move removes the piece from its origin square, puts it (possibly promoted)
// on its destination square, and may remove one captured piece. Three feature
// changes per perspective is the true maximum; 4 is slack.
#define NNUE_MAX_DIRTY 4

struct nnue_delta {
    int n_sub;
    int n_add;
    int16_t sub[2][NNUE_MAX_DIRTY];  // [perspective 0 = p1, 1 = p2][k]
    int16_t add[2][NNUE_MAX_DIRTY];
};

#if NNUE_VERIFY_INCREMENTAL
static long long nnue_incremental_checks = 0;
static long long nnue_incremental_mismatches = 0;
static int nnue_incremental_worst = 0;
#endif

struct nnue_stack {
    int max_ply;
    int16_t* acc;                // max_ply * 2 * NNUE_L1, 64 byte aligned
    // bit 0 = perspective 0 (player 1) is up to date at this ply, bit 1 = same
    // for player 2. The two are tracked separately because they are independent
    // chains: reaching acc[d][p] only ever needs acc[d-1][p]. A leaf therefore
    // materialises one perspective instead of both, and since leaves are most of
    // the tree that is most of the saving.
    unsigned char* computed;     // max_ply
    struct nnue_delta* delta;    // max_ply
    void* acc_raw;               // the un-aligned allocation, for free()
};
#define NNUE_PERSP_BIT(p) ((unsigned char)(1u << (p)))

#define NNUE_ACC_AT(st, ply, persp) \
    ((st)->acc + ((size_t)(ply) * 2 + (size_t)(persp)) * (size_t)NNUE_L1)

static struct nnue_stack* nnue_stack_create(int max_ply){
    struct nnue_stack* st;
    unsigned char* raw;
    size_t bytes;

    if (max_ply < 2) max_ply = 2;
    st = (struct nnue_stack*)malloc(sizeof(struct nnue_stack));
    if (st == NULL) return NULL;

    // over-allocate and align by hand rather than reach for _aligned_malloc /
    // posix_memalign, which are spelled differently on every toolchain
    bytes = (size_t)max_ply * 2u * (size_t)NNUE_L1 * sizeof(int16_t);
    raw = (unsigned char*)malloc(bytes + 63);
    st->acc_raw = raw;
    st->acc = raw ? (int16_t*)(void*)(((uintptr_t)raw + 63) & ~(uintptr_t)63) : NULL;
    st->computed = (unsigned char*)malloc((size_t)max_ply);
    st->delta = (struct nnue_delta*)malloc((size_t)max_ply * sizeof(struct nnue_delta));
    st->max_ply = max_ply;

    if (raw == NULL || st->computed == NULL || st->delta == NULL){
        free(raw);
        free(st->computed);
        free(st->delta);
        free(st);
        return NULL;
    }
    memset(st->computed, 0, (size_t)max_ply);
    return st;
}

static void nnue_stack_destroy(struct nnue_stack* st){
    if (st == NULL) return;
    free(st->acc_raw);
    free(st->computed);
    free(st->delta);
    free(st);
}

// Seed ply 0 with the root position and drop every accumulator below it.
// Clearing the whole `computed` array is what makes a new search safe: a stale
// 1 left over from the previous iteration would stop the walk-back early and
// hand out an accumulator belonging to a different position.
static void nnue_stack_reset(struct nnue_stack* st,
                             long long p1, long long p2, long long p1k, long long p2k){
    int indices[NNUE_MAX_ACTIVE];
    int n;
    if (st == NULL) return;
    memset(st->computed, 0, (size_t)st->max_ply);
    n = nnue_encode(p1, p2, p1k, p2k, 1, indices);
    nnue_acc_refresh(NNUE_ACC_AT(st, 0, 0), indices, n);
    n = nnue_encode(p1, p2, p1k, p2k, 2, indices);
    nnue_acc_refresh(NNUE_ACC_AT(st, 0, 1), indices, n);
    st->computed[0] = NNUE_PERSP_BIT(0) | NNUE_PERSP_BIT(1);
}

// Record how the child at ply+1 differs from the node at ply. Called right
// after the move is played on the board and before the recursive call.
//
// `moved_type` is the type the piece had *before* the move (the search already
// has it, for undo), `captured_type` is update_board()'s return value: 0 for a
// quiet move, otherwise the type of the jumped piece, which always sits on the
// square between the endpoints. The promotion rule is update_board()'s own.
//
// No accumulator arithmetic happens here. Most nodes return before they are
// ever evaluated, and a delta that is never needed should cost a few stores.
static void nnue_stack_push(struct nnue_stack* st, int ply,
                            int moved_type, int from, int to, int captured_type){
    struct nnue_delta* d;
    int to_type, p;

    if (st == NULL || ply < 0 || ply + 1 >= st->max_ply) return;

    to_type = moved_type;
    if (moved_type == 1 && to < 8) to_type = 3;
    else if (moved_type == 2 && to > 55) to_type = 4;

    d = &st->delta[ply + 1];
    for (p = 0; p < 2; p++){
        d->sub[p][0] = (int16_t)nnue_feature_index(moved_type, from, p + 1);
        d->add[p][0] = (int16_t)nnue_feature_index(to_type, to, p + 1);
    }
    d->n_sub = 1;
    d->n_add = 1;

    if (captured_type){
        int mid = (from + to) / 2;
        for (p = 0; p < 2; p++)
            d->sub[p][1] = (int16_t)nnue_feature_index(captured_type, mid, p + 1);
        d->n_sub = 2;
    }

    st->computed[ply + 1] = 0;
}

// Evaluate the board at `ply`, updating only as much of the accumulator stack
// as this actually requires. The board is passed as well so that the two cases
// this cannot serve - a ply past the end of the stack, and a stack that failed
// to allocate - fall back to the from-scratch path instead of being wrong.
static int nnue_stack_eval(struct nnue_stack* st, int ply, int player,
                           long long p1, long long p2, long long p1k, long long p2k){
#if !NNUE_INCREMENTAL
    (void)st;
    (void)ply;
    return nnue_eval_position(p1, p2, p1k, p2k, player);
#else
    int base, d;
    // perspective 0 is player 1's chair, perspective 1 is player 2's
    const int p = player - 1;
    const unsigned char bit = NNUE_PERSP_BIT(p);

    if (st == NULL || ply < 0 || ply >= st->max_ply || st->computed[0] != 3)
        return nnue_eval_position(p1, p2, p1k, p2k, player);

    if (!(st->computed[ply] & bit)){
        // the deepest ancestor that is already up to date *for this
        // perspective*. Usually ply-1 or a little above it: an accumulator
        // built for one leaf serves every sibling below the same parent.
        base = ply;
        while (base > 0 && !(st->computed[base] & bit)) base--;
        for (d = base + 1; d <= ply; d++){
            const struct nnue_delta* dl = &st->delta[d];
            nnue_acc_update(NNUE_ACC_AT(st, d, p), NNUE_ACC_AT(st, d - 1, p),
                            dl->sub[p], dl->n_sub, dl->add[p], dl->n_add);
            st->computed[d] |= bit;
        }
    }

#if NNUE_VERIFY_INCREMENTAL
    {
        int got = nnue_propagate(NNUE_ACC_AT(st, ply, p));
        int want = nnue_eval_position(p1, p2, p1k, p2k, player);
        nnue_incremental_checks++;
        if (got != want){
            nnue_incremental_mismatches++;
            if (nnue_incremental_worst < (got > want ? got - want : want - got))
                nnue_incremental_worst = (got > want ? got - want : want - got);
        }
        return got;
    }
#else
    return nnue_propagate(NNUE_ACC_AT(st, ply, p));
#endif
#endif /* NNUE_INCREMENTAL */
}

#endif /* NNUE_C */
