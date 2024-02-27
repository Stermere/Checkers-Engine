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

#define MT_STATE_SIZE 624
#define NUMBER_OF_BUCKETS 2


// define functions
struct hash_table_entry;
struct hash_table;
unsigned long long int* compute_piece_hash_diffs();
long long int rand_num();
struct hash_table_entry* get_storage_index(struct hash_table *table, unsigned long long int hash, int age, int depth);
struct hash_table_entry* get_hash_entry(struct hash_table *table, unsigned long long int hash, int age, int depth);
struct hash_table_entry* check_for_empty_spot(struct hash_table *table, unsigned long long int hash);
int check_for_entry(struct hash_table_entry *table, unsigned long long int hash);

// stores the data related to the hashed value
// ie. the hash, its evaluation, what is the type of fail (fail high or fail low, or true value), the best move, etc.
struct hash_table_entry {
    unsigned long long int hash;
    int eval;
    char depth; 
    char age;
    char player;
    // moves are stored in a format of: top byte is the start square, bottom byte is the end square
    short best_move;
    char node_type;
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
    unsigned long long int* piece_hash_diff;
};

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

// initializes the hash table
struct hash_table* init_hash_table(long long int size){
    struct hash_table *table = (struct hash_table*)malloc(sizeof(struct hash_table));
    table->table = (struct hash_table_entry*)calloc(size, sizeof(struct hash_table_entry));
    // check for allocation failiure and exit if it does
    if (table->table == NULL){
        printf("Error: failed to allocate memory for hash table\n");
        exit(1);
    }

    table->size = (size - (size % 4)) / 4; // 4 buckets per hash value
    table->total_size = size;
    table->piece_hash_diff = compute_piece_hash_diffs();
    table->num_entries = 1;
    table->hit_count = 1;
    table->miss_count = 1;

    return table;
}

// adds a new entry to the hash table (depth is the depth remaining at the node)
void add_hash_entry(struct hash_table *table, unsigned long long int hash, int eval, int depth, int age, int player,
                    short best_move, char node_type){

    // get the entry and incriment the number of entries
    struct hash_table_entry* entry_index = get_storage_index(table, hash, age, depth);

    if (entry_index == NULL) {
        return;
    }

    // check if the entry is empty
    if (!check_for_entry(entry_index, hash)) {
        table->num_entries++;
    }

    if (entry_index->hash == hash && entry_index->depth > depth) {
        return;
    }
    
    entry_index->depth = depth;
    entry_index->node_type = node_type;
    entry_index->best_move = best_move;
    entry_index->player = player;
    entry_index->age = age; 
    entry_index->hash = hash;
    entry_index->eval = eval;

}

// check if there is a hash entry for the given hash
// return 0 if there is no entry, 1 if there is an entry
int check_for_entry(struct hash_table_entry* entry_index, unsigned long long int hash){
    return entry_index->hash != 0llu;
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

// returns the index of the hash table that will be replaced by the new entry
// if there is an empty spot, returns the index of the empty spot
struct hash_table_entry* get_storage_index(struct hash_table *table, unsigned long long int hash, int age, int depth){
    struct hash_table_entry* index = check_for_empty_spot(table, hash);
    if (index != NULL){
        return index;
    }

    struct hash_table_entry* entry_index = table->table + ((hash % table->size) * NUMBER_OF_BUCKETS);
    int min_depth = entry_index->depth;
    for (int i = 0; i < NUMBER_OF_BUCKETS; i++) {
        if ((entry_index->age) <= age) {
            return entry_index;
        }

        if ((entry_index->depth < min_depth) && (entry_index->node_type != PV_NODE)){
            index = entry_index;
            min_depth = entry_index->depth;
        }
        entry_index++;
    }


    return index;
}

// checks if any of the spots for this hash value are empty
// returns the index of the empty spot or a entry with the same hash if there is one, otherwise returns -1
struct hash_table_entry* check_for_empty_spot(struct hash_table *table, unsigned long long int hash){
    struct hash_table_entry* entry_index = table->table + ((hash % table->size) * NUMBER_OF_BUCKETS);
    struct hash_table_entry* empty_spot = NULL;
    for (int i = 0; i < NUMBER_OF_BUCKETS; i++) {
        if (entry_index->hash == hash){
            return entry_index;
        }
        else if (entry_index->hash == 0llu){
            empty_spot = entry_index;
        }
        entry_index++;
    }
    return empty_spot;
}

// compute the hash of a board
unsigned long long int get_hash(unsigned long long int p1, unsigned long long int p2, unsigned long long int p1k, unsigned long long int p2k, struct hash_table * table){
    long long int hash = 0;
    for (int i = 0; i < 64; i++){
        if (p1 & (1ll << i)){
            hash = hash ^ table->piece_hash_diff[i];
        }
        else if (p2 & (1ll << i)){
            hash = hash ^ table->piece_hash_diff[i + 64];
        }
        else if (p1k & (1ll << i)){
            hash = hash ^ table->piece_hash_diff[i + 128];
        }
        else if (p2k & (1ll << i)){
            hash = hash ^ table->piece_hash_diff[i + 192];
        }
    }
    return hash;
}

// compute the hash table piece diffs for quickly computing hashes of boards
unsigned long long int* compute_piece_hash_diffs(){
    srand(time(NULL));
    struct mt_state rng_state;
    mt_init(&rng_state, time(NULL));
    long long int* piece_hash_diffs = (long long int*)malloc(sizeof(long long int) * (64 * 4));
    for (int i = 0; i < (64 * 4); i++){
        piece_hash_diffs[i] = mt_rand(&rng_state);
    }
    return piece_hash_diffs;
}

// frees the hash table
void free_hash_table(struct hash_table *table){
    free(table->table);
    free(table);
}

