// Author: Collin kees
// Description: Hash table for storing and updating checkers hashes and relevant data

// includes
#include <stdlib.h>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__SSE__) || defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

#define PV_NODE 1
#define LOWER_BOUND 2
#define UPPER_BOUND 3
#define UNKNOWN_NODE 5
#define NO_MOVE 0 // no move was found

// "no static evaluation is cached for this entry". Deliberately outside the
// heuristic band (board_search.c clamps static evals to +-EVAL_MAX = 1500), so
// it can never collide with a real score.
#define NO_EVAL 32000

#define MT_STATE_SIZE 624
#define NUMBER_OF_BUCKETS 4

// hit/miss/pv counters feed the search report and nothing else, but they are
// three read-modify-writes on shared struct fields on the hottest path in the
// engine - every node probes, and a probe that hits touches two of them.
//
// Off by default; build with /D TT_STATS=1 to get the hit ratio back. It cannot
// key off PRINT_OUTPUT because this file is included at board_search.c:10, forty
// lines before PRINT_OUTPUT is defined - the reference would quietly evaluate to
// 0 and the report would print a ratio computed from the two initialisation
// values, which is worse than not printing it.
#ifndef TT_STATS
#define TT_STATS 0
#endif

// zobrist key layout: [0..255] piece-square keys (4 piece types x 64 squares),
// [256..319] forced-continuation-square keys (mid multi-jump state),
// [320] side-to-move key (xor'ed in when player 2 is to move)
#define ZOBRIST_KEYS 321
#define FORCED_KEY_OFFSET 256
#define SIDE_KEY_INDEX 320


// define functions
struct hash_table_entry;
struct hash_table;
unsigned long long int* compute_piece_hash_diffs();
struct hash_table_entry* get_storage_index(struct hash_table *table, unsigned long long int hash);
struct hash_table_entry* get_hash_entry(struct hash_table *table, unsigned long long int hash);

// stores the data related to the hashed value
// ie. the hash, its evaluation, what is the type of fail (fail high or fail low, or true value), the best move, etc.
//
// **This struct is exactly 16 bytes and that is the point.** At
// NUMBER_OF_BUCKETS = 4 a bucket is then exactly one 64 byte cache line, so a
// probe touches one line and the prefetch below covers all of it. The previous
// layout was 24 bytes: a 96 byte bucket straddling two lines, which at ~96 MB
// (where every probe misses last level cache anyway) meant two memory stalls per
// node where one would do, and a prefetch that only ever pulled in the first.
//
// What each field had to give up to get there:
//
//   * `key` is the TOP 32 bits of the hash, not the whole thing. The bucket index
//     is taken from the bottom bits, so an entry found in bucket b with key k
//     agrees with the probe on 32 + log2(bucket count) = 52 bits. A false match
//     therefore happens about once in 2^32 probes per slot, which over a whole
//     search is a fraction of one - and this is the same trade every serious
//     engine makes (Stockfish verifies 16 bits).
//   * `eval` is int16 rather than int. Every score written here is bounded by
//     WIN_SCORE + 1 = 4001, because the INFINITY (1e9) timeout sentinel returns
//     out of negmax before it ever reaches add_hash_entry. Checked below.
//   * `node_type` and `player` share `flags` with the occupancy bit, since one
//     needs 2 bits and the other 1.
//
// `flags` also replaces `hash == 0` as the "never written" test, which a 32 bit
// key can no longer serve: a real key of 0 is perfectly possible.
struct hash_table_entry {
    unsigned int key;
    short eval;
    // the STATIC evaluation of this position (what the net says without searching),
    // as opposed to `eval` which is the result of a search. Cached here because the
    // search now consults it at interior nodes for its pruning margins, and a
    // transposition hit should not have to pay for the network again. NO_EVAL when
    // this entry has never been evaluated statically.
    short static_eval;
    // moves are stored in a format of: top byte is the start square, bottom byte is the end square
    short best_move;
    signed char depth;
    unsigned char age;
    unsigned char flags;
    unsigned char pad;
};

#define TT_TYPE_MASK   0x03u   // node_type is 1..3, so two bits
#define TT_PLAYER_BIT  0x04u   // set when the side to move is player 2
#define TT_OCCUPIED    0x80u   // this slot has been written at least once

#define TT_KEY(h)        ((unsigned int)((h) >> 32))
#define TT_NODE_TYPE(e)  ((char)((e)->flags & TT_TYPE_MASK))
#define TT_PLAYER(e)     ((int)(((e)->flags & TT_PLAYER_BIT) >> 2) + 1)

// the cache line claim above is load bearing, so make the build fail if the
// layout ever drifts off it rather than quietly losing the speedup
typedef char tt_entry_is_16_bytes[(sizeof(struct hash_table_entry) == 16) ? 1 : -1];
typedef char tt_bucket_is_one_cache_line[(sizeof(struct hash_table_entry) * NUMBER_OF_BUCKETS == 64) ? 1 : -1];
// (the matching check that `eval` still fits int16 lives in board_search.c,
// next to WIN_SCORE - this file is included before that is defined)

// holds the tabel of hash_table_entry's and the size of the table
struct hash_table {
    struct hash_table_entry *table;
    long long int size;
    // size - 1. The bucket index is computed on every probe AND every store, so
    // it runs at least twice per node; `hash % size` there is a 64-bit division,
    // which is 20-40 cycles of latency each time. size is a power of two by
    // construction (see init_hash_table), so the mask is exactly equivalent.
    unsigned long long int size_mask;
    long long int total_size;
    long long int num_entries;
    long long int hit_count;
    long long int miss_count;
    int pv_retrival_count;
    unsigned char age;
    unsigned long long int* piece_hash_diff;
};

// the transposition table persists for the lifetime of the process so knowledge
// carries over from move to move (zobrist keys use a fixed seed so hashes stay valid)
static struct hash_table* GLOBAL_HASH_TABLE = NULL;

// Define the structure for the Mersenne Twister rng
struct mt_state {
    unsigned long long mt[MT_STATE_SIZE];
    int index;
};

// Initialize the Mersenne Twister state with a seed
void mt_init(struct mt_state *state, unsigned long long seed) {
    state->mt[0] = seed;
    for (int i = 1; i < MT_STATE_SIZE; ++i) {
        state->mt[i] = 0xFFFFFFFFFFFFFFFFull & (6364136223846793005ull * (state->mt[i - 1] ^ (state->mt[i - 1] >> 62)) + i);
    }
    state->index = MT_STATE_SIZE;
}

// Generate a 64-bit random number using the Mersenne Twister algorithm
unsigned long long mt_rand(struct mt_state *state) {
    if (state->index >= MT_STATE_SIZE) {
        for (int i = 0; i < MT_STATE_SIZE; ++i) {
            unsigned long long y = (state->mt[i] & 0x8000000000000000ull) + (state->mt[(i + 1) % MT_STATE_SIZE] & 0x7FFFFFFFFFFFFFFFull);
            state->mt[i] = state->mt[(i + 397) % MT_STATE_SIZE] ^ (y >> 1);
            if (y % 2 != 0) {
                state->mt[i] ^= 0x9D2C5680u;
            }
        }
        state->index = 0;
    }

    unsigned long long y = state->mt[state->index];
    y ^= (y >> 29) & 0x5555555555555555ull;
    y ^= (y << 17) & 0x71D67FFFEDA60000ull;
    y ^= (y << 37) & 0xFFF7EEE000000000ull;
    y ^= y >> 43;

    state->index++;

    return y;
}

// initializes the hash table (or reuses the persistent one, bumping its age)
struct hash_table* init_hash_table(long long int size){
    if (GLOBAL_HASH_TABLE != NULL){
        GLOBAL_HASH_TABLE->age++;
        GLOBAL_HASH_TABLE->hit_count = 1;
        GLOBAL_HASH_TABLE->miss_count = 1;
        GLOBAL_HASH_TABLE->pv_retrival_count = 0;
        return GLOBAL_HASH_TABLE;
    }

    struct hash_table *table = (struct hash_table*)malloc(sizeof(struct hash_table));

    // 64 byte aligned, not merely allocated: a bucket is NUMBER_OF_BUCKETS
    // entries of 16 bytes = exactly one cache line, and that only buys anything
    // if the base is on a line boundary too. calloc usually hands back a page for
    // an allocation this size, but "usually" is not a guarantee and a misaligned
    // base would silently put every single bucket back across two lines - the
    // exact cost this layout exists to remove.
    size_t bytes = (size_t)size * sizeof(struct hash_table_entry);
#if defined(_MSC_VER)
    table->table = (struct hash_table_entry*)_aligned_malloc(bytes, 64);
#else
    table->table = (struct hash_table_entry*)aligned_alloc(64, bytes);
#endif
    // check for allocation failiure and exit if it does
    if (table->table == NULL){
        printf("Error: failed to allocate memory for hash table\n");
        exit(1);
    }
    // aligned allocators do not zero, and an all zero entry is what "never
    // written" means (flags has no TT_OCCUPIED bit)
    memset(table->table, 0, bytes);

    table->size = size / NUMBER_OF_BUCKETS;
    // the mask below is only equivalent to a modulo for a power of two, and a
    // silently wrong index would show up as a mysterious strength loss rather
    // than as a crash, so refuse to run instead
    if (table->size <= 0 || (table->size & (table->size - 1)) != 0){
        printf("Error: hash table bucket count %lld is not a power of two\n", table->size);
        exit(1);
    }
    table->size_mask = (unsigned long long int)(table->size - 1);
    table->total_size = size;
    table->piece_hash_diff = compute_piece_hash_diffs();
    table->num_entries = 1;
    table->hit_count = 1;
    table->miss_count = 1;
    table->pv_retrival_count = 0;
    table->age = 0;

    GLOBAL_HASH_TABLE = table;
    return table;
}

// adds a new entry to the hash table (depth is the depth remaining at the node).
// static_eval may be NO_EVAL, meaning "the caller did not compute one"; an
// already cached static evaluation for the same position is kept in that case,
// since it stays valid however the search score changes.
void add_hash_entry(struct hash_table *table, unsigned long long int hash, int eval, int depth, int player,
                    short best_move, char node_type, short static_eval){

    struct hash_table_entry* entry = get_storage_index(table, hash);
    if (entry == NULL) {
        return;
    }

    const unsigned int key = TT_KEY(hash);
    if (entry->key == key && (entry->flags & TT_OCCUPIED)) {
        // same position: the static evaluation does not depend on the search, so
        // a freshly computed one is never worse than what is already there and a
        // missing one must not erase it
        if (static_eval == NO_EVAL) {
            static_eval = entry->static_eval;
        }
        // keep a deeper entry from this search unless the new one is exact
        int stale = (entry->age != table->age);
        entry->age = table->age;
        if (!stale && entry->depth > depth && node_type != PV_NODE) {
            // the search result is not being replaced, but a static evaluation the
            // entry did not have before still is - it is strictly new information
            entry->static_eval = static_eval;
            return;
        }
    } else {
        if (!(entry->flags & TT_OCCUPIED)) {
            table->num_entries++;
        }
    }

    entry->depth = (signed char)depth;
    entry->flags = (unsigned char)(TT_OCCUPIED
                                   | ((unsigned)node_type & TT_TYPE_MASK)
                                   | ((player == 2) ? TT_PLAYER_BIT : 0u));
    entry->best_move = best_move;
    entry->age = table->age;
    entry->key = key;
    entry->eval = (short)eval;
    entry->static_eval = static_eval;
}

// start pulling a bucket into cache without waiting for it.
// The table is ~96 MB, so essentially every probe is a last-level cache miss and
// the search stalls on it. The child's hash is known well before the child runs
// (the move is made, the reductions are computed, has_any_jump is evaluated), so
// asking for the line at that point hides most of the latency behind work that
// was going to happen anyway. Purely a hint: it cannot change what is read.
static inline void prefetch_hash_entry(struct hash_table *table, unsigned long long int hash){
    const char* addr = (const char*)(table->table + ((hash & table->size_mask) * NUMBER_OF_BUCKETS));
#if defined(_MSC_VER)
    _mm_prefetch(addr, _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 3);
#else
    (void)addr;
#endif
}

// returns the entry for the given hash.
// note there is deliberately no depth or age filtering here - the caller decides
// what an entry may be used for (a stale entry is still a fine move for ordering
// even when it may not cut), and this used to take `age` and `depth` arguments it
// silently ignored, which read as if that policy lived here.
struct hash_table_entry* get_hash_entry(struct hash_table *table, unsigned long long int hash){
    struct hash_table_entry* entry_index = table->table + ((hash & table->size_mask) * NUMBER_OF_BUCKETS);
    const unsigned int key = TT_KEY(hash);
    for (int i = 0; i < NUMBER_OF_BUCKETS; i++) {

        // the occupancy test is not redundant with the key test: a slot that has
        // never been written reads back as key 0, which is a perfectly legal key
        if (entry_index->key == key && (entry_index->flags & TT_OCCUPIED)){
#if TT_STATS
            // incriment the pv retrival count if relevent
            if (TT_NODE_TYPE(entry_index) == PV_NODE){
                table->pv_retrival_count++;
            }
            table->hit_count++;
#endif
            // return the entry
            return entry_index;
        }
        entry_index++;
    }
#if TT_STATS
    table->miss_count++;
#endif
    return NULL;
}

// returns the slot the new entry should be written to.
// prefers (in order): the entry for this exact hash, an empty slot, then the
// least valuable victim (older age first, then shallower depth, PV slightly protected)
struct hash_table_entry* get_storage_index(struct hash_table *table, unsigned long long int hash){
    struct hash_table_entry* base = table->table + ((hash & table->size_mask) * NUMBER_OF_BUCKETS);
    struct hash_table_entry* victim = NULL;
    int victim_score = 1 << 30;
    const unsigned int key = TT_KEY(hash);

    for (int i = 0; i < NUMBER_OF_BUCKETS; i++) {
        struct hash_table_entry* e = base + i;
        if (e->key == key && (e->flags & TT_OCCUPIED)){
            return e;
        }

        int score;
        if (!(e->flags & TT_OCCUPIED)){
            score = -(1 << 29);
        } else {
            int staleness = (unsigned char)(table->age - e->age);
            score = (int)e->depth - 8 * staleness + ((TT_NODE_TYPE(e) == PV_NODE) ? 4 : 0);
        }

        if (score < victim_score){
            victim_score = score;
            victim = e;
        }
    }

    return victim;
}

// compute the hash of a board (piece placement only; the caller xors in the
// side-to-move key and any forced-continuation key)
unsigned long long int get_hash(unsigned long long int p1, unsigned long long int p2, unsigned long long int p1k, unsigned long long int p2k, struct hash_table * table){
    unsigned long long int hash = 0;
    for (int i = 0; i < 64; i++){
        if (p1 & (1ull << i)){
            hash = hash ^ table->piece_hash_diff[i];
        }
        else if (p2 & (1ull << i)){
            hash = hash ^ table->piece_hash_diff[i + 64];
        }
        else if (p1k & (1ull << i)){
            hash = hash ^ table->piece_hash_diff[i + 128];
        }
        else if (p2k & (1ull << i)){
            hash = hash ^ table->piece_hash_diff[i + 192];
        }
    }
    return hash;
}

// compute the hash table piece diffs for quickly computing hashes of boards
// uses a fixed seed so hashes are stable across searches and processes
unsigned long long int* compute_piece_hash_diffs(){
    struct mt_state rng_state;
    mt_init(&rng_state, 0xC4CC5EEDC0FFEE01ull);
    unsigned long long int* piece_hash_diffs = (unsigned long long int*)malloc(sizeof(unsigned long long int) * ZOBRIST_KEYS);
    for (int i = 0; i < ZOBRIST_KEYS; i++){
        piece_hash_diffs[i] = mt_rand(&rng_state);
    }
    return piece_hash_diffs;
}

// frees the hash table
void free_hash_table(struct hash_table *table){
    free(table->table);
    free(table->piece_hash_diff);
    free(table);
}
