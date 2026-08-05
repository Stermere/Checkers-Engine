// Author: Collin kees
// Description: Hash table for storing and updating checkers hashes and relevant data

// includes
#include <stdlib.h>

#define PV_NODE 1
#define LOWER_BOUND 2
#define UPPER_BOUND 3
#define HORIZON_NODE 4
#define UNKNOWN_NODE 5
#define NULL_MOVE 4
#define NO_MOVE 0 // no move was found

// "no static evaluation is cached for this entry". Deliberately outside the
// heuristic band (board_search.c clamps static evals to +-EVAL_MAX = 1500), so
// it can never collide with a real score.
#define NO_EVAL 32000

#define MT_STATE_SIZE 624
#define NUMBER_OF_BUCKETS 4

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
struct hash_table_entry* get_hash_entry(struct hash_table *table, unsigned long long int hash, int age, int depth);

// stores the data related to the hashed value
// ie. the hash, its evaluation, what is the type of fail (fail high or fail low, or true value), the best move, etc.
struct hash_table_entry {
    unsigned long long int hash;
    int eval;
    signed char depth;
    unsigned char age;
    char player;
    // moves are stored in a format of: top byte is the start square, bottom byte is the end square
    short best_move;
    char node_type;
    // the STATIC evaluation of this position (what the net says without searching),
    // as opposed to `eval` which is the result of a search. Cached here because the
    // search now consults it at interior nodes for its pruning margins, and a
    // transposition hit should not have to pay for the network again. NO_EVAL when
    // this entry has never been evaluated statically.
    short static_eval;
};

// holds the tabel of hash_table_entry's and the size of the table
struct hash_table {
    struct hash_table_entry *table;
    long long int size;
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
    table->table = (struct hash_table_entry*)calloc(size, sizeof(struct hash_table_entry));
    // check for allocation failiure and exit if it does
    if (table->table == NULL){
        printf("Error: failed to allocate memory for hash table\n");
        exit(1);
    }

    table->size = size / NUMBER_OF_BUCKETS;
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
void add_hash_entry(struct hash_table *table, unsigned long long int hash, int eval, int depth, int age, int player,
                    short best_move, char node_type, short static_eval){

    struct hash_table_entry* entry = get_storage_index(table, hash);
    if (entry == NULL) {
        return;
    }

    if (entry->hash == hash) {
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
        if (entry->hash == 0llu) {
            table->num_entries++;
        }
    }

    entry->depth = (signed char)depth;
    entry->node_type = node_type;
    entry->best_move = best_move;
    entry->player = (char)player;
    entry->age = table->age;
    entry->hash = hash;
    entry->eval = eval;
    entry->static_eval = static_eval;
}

// returns the entry for the given hash
struct hash_table_entry* get_hash_entry(struct hash_table *table, unsigned long long int hash, int age, int depth){
    struct hash_table_entry* entry_index = table->table + ((hash % table->size) * NUMBER_OF_BUCKETS);
    for (int i = 0; i < NUMBER_OF_BUCKETS; i++) {

        if (entry_index->hash == hash){
            // incriment the pv retrival count if relevent
            if (entry_index->node_type == PV_NODE){
                table->pv_retrival_count++;
            }

            // return the entry
            table->hit_count++;
            return entry_index;
        }
        entry_index++;
    }
    table->miss_count++;
    return NULL;
}

// returns the slot the new entry should be written to.
// prefers (in order): the entry for this exact hash, an empty slot, then the
// least valuable victim (older age first, then shallower depth, PV slightly protected)
struct hash_table_entry* get_storage_index(struct hash_table *table, unsigned long long int hash){
    struct hash_table_entry* base = table->table + ((hash % table->size) * NUMBER_OF_BUCKETS);
    struct hash_table_entry* victim = NULL;
    int victim_score = 1 << 30;

    for (int i = 0; i < NUMBER_OF_BUCKETS; i++) {
        struct hash_table_entry* e = base + i;
        if (e->hash == hash){
            return e;
        }

        int score;
        if (e->hash == 0llu){
            score = -(1 << 29);
        } else {
            int staleness = (unsigned char)(table->age - e->age);
            score = (int)e->depth - 8 * staleness + ((e->node_type == PV_NODE) ? 4 : 0);
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
