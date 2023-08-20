// Author: Collin kees
// Description: Hash table for storing and updating checkers hashes and relevant data

// includes
#include <stdlib.h>

#define PV_NODE 1
#define FAIL_HIGH 2
#define FAIL_LOW 3
#define HORIZON_NODE 4
#define UNKNOWN_NODE 5
#define NULL_MOVE 4
#define NO_MOVE 0 // no move was found

#define MT_STATE_SIZE 624


// define functions
struct hash_table_entry;
struct hash_table;
unsigned long long int* compute_piece_hash_diffs();
long long int rand_num();
struct hash_table_entry* get_storage_index(struct hash_table *table, unsigned long long int hash, int age, int depth);
struct hash_table_entry* get_hash_entry(struct hash_table *table, unsigned long long int hash, int age, int depth, int player);
struct hash_table_entry* check_for_empty_spot(struct hash_table *table, unsigned long long int hash);
int check_for_entry(struct hash_table_entry *table, unsigned long long int hash);
void mt_init(struct mt_state *state, unsigned long long seed);
unsigned long long mt_rand(struct mt_state *state);

// stores the data related to the hashed value
// ie. the hash, its evaluation, what is the type of fail (fail high or fail low, or true value), the best move, etc.
struct hash_table_entry {
    unsigned long long int hash;
    float eval;
    int depth; 
    int age;
    int player;
    // moves are stored in a format of: top byte is the start square, bottom byte is the end square
    short best_move;
    short refutation_move;
    char node_type;
};

// holds the tabel of hash_table_entry's and the size of the table
struct hash_table {
    struct hash_table_entry *table;
    long long int size;
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

// initializes the hash table
struct hash_table* init_hash_table(int size){
    struct hash_table *table = (struct hash_table*)malloc(sizeof(struct hash_table));
    table->table = (struct hash_table_entry*)calloc(size, sizeof(struct hash_table_entry));
    // check for allocation failiure and exit if it does
    if (table->table == NULL){
        printf("Error: failed to allocate memory for hash table\n");
        exit(1);
    }

    // print the size and step through each entry checking for allocation failiure
    printf("Hash table size: %d\n", size);
    for (int i = 0; i < size; i++){
        if (table->table[i].hash != 0llu){
            printf("Error: failed to allocate memory for hash table\n");
            exit(1);
        }
    }

    table->size = size;
    table->piece_hash_diff = compute_piece_hash_diffs();
    table->num_entries = 0;
    table->hit_count = 0;
    table->miss_count = 0;

    return table;
}

// adds a new entry to the hash table (depth should grow as it gets deeper, unlike the search depth which gets smaller)
void add_hash_entry(struct hash_table *table, unsigned long long int hash, float eval, int depth, int age, int player,
                    short best_move, short refutation_move, char node_type){

    // get the entry and incriment the number of entries
    struct hash_table_entry* entry_index = get_storage_index(table, hash, age, depth);


    if (entry_index == NULL) {
        return;
    }

    // check if the entry is empty
    if (!check_for_entry(entry_index, hash)) {
        table->num_entries++;
    }

    // if we have not returned yet add/replace the entry
    entry_index->hash = hash;
    entry_index->eval = eval;
    entry_index->depth = depth;
    entry_index->age = age; 
    entry_index->player = player;
    entry_index->best_move = best_move;
    entry_index->refutation_move = refutation_move;
    entry_index->node_type = node_type;
}

// check if there is a hash entry for the given hash
// return 0 if there is no entry, 1 if there is an entry
int check_for_entry(struct hash_table_entry* entry_index, unsigned long long int hash){
    return entry_index->hash != 0llu;
}


// returns the eval of a hash table entry if it exists and is the right hash value
// if the entry does not exits or is the wrong hash value, returns NAN
// also return NAN if the age of the entry is older than the current age
struct hash_table_entry* get_hash_entry(struct hash_table *table, unsigned long long int hash, int age, int depth, int player){
    for (int i = 0; i < 4; i++) {
        struct hash_table_entry* entry_index = table->table + ((hash + i) % table->size);

        if (entry_index->hash == hash){
            // incriment the pv retrival count if relevent
            if (entry_index->node_type == PV_NODE){
                table->pv_retrival_count++;
            }
            else {
                table->pv_retrival_count = 0;
            }

            // return the entry
            table->hit_count++;
            return entry_index;
        }
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

    for (int i = 0; i < 4; i++) {
        struct hash_table_entry* entry_index = table->table + ((hash + i) % table->size);
        if (entry_index->depth > depth){
            index = entry_index;
            depth = entry_index->depth;
        }
    }


    return index;
}

// checks if any of the spots for this hash value are empty
// returns the index of the empty spot or a entry with the same hash if there is one, otherwise returns -1
struct hash_table_entry* check_for_empty_spot(struct hash_table *table, unsigned long long int hash){
    for (int i = 0; i < 4; i++) {
        struct hash_table_entry* entry_index = table->table + ((hash + i) % table->size);
        if (entry_index->hash == 0llu || entry_index->hash == hash){
            return entry_index;
        }
    }
    return NULL;
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

// frees the hash table
void free_hash_table(struct hash_table *table){
    free(table->table);
    free(table);
}

