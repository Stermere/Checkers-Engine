// Author: Collin kees
// Description: Hash table for storing and updating checkers hashes and relevant data

// includes
#include <stdlib.h>

// define functions
struct hash_table_entry;
struct hash_table;
unsigned long long int* compute_piece_hash_diffs();
long long int rand_num();
int compare_hash_entries(struct hash_table_entry *entry1, int depth, int age);
float get_hash_entry(struct hash_table *table, unsigned long long int hash);
int check_for_entry(struct hash_table_entry *table, unsigned long long int hash);

// stores the data related to the hashed value
// ie. the hash, its evaluation, and the movs that come next
struct hash_table_entry {
    unsigned long long int hash;
    float eval;
    int depth; 
    int age;
};

// holds the tabel of hash_table_entry's and the size of the table
struct hash_table {
    struct hash_table_entry *table;
    long long int size;
    int num_entries;
    unsigned long long int* piece_hash_diff;
};

// initializes the hash table
struct hash_table* init_hash_table(int size){
    struct hash_table *table = (struct hash_table*)malloc(sizeof(struct hash_table));
    table->table = (struct hash_table_entry*)malloc(sizeof(struct hash_table_entry) * size);
    // check for allocation failiure and exit if it does
    if (table->table == NULL){
        printf("Error: failed to allocate memory for hash table\n");
        exit(1);
    }
    // loop through all the entries and set them to 0
    for (int i = 0; i < size; i++){
        table->table[i].hash = 0ull;
        table->table[i].eval = 0.0;
        table->table[i].depth = 0;
        table->table[i].age = 0;
    }
    table->size = size;
    table->piece_hash_diff = compute_piece_hash_diffs();
    table->num_entries = 0;

    return table;
}

// adds a new entry to the hash table (depth should grow as it gets deeper, unlike the search depth which gets smaller)
void add_hash_entry(struct hash_table *table, unsigned long long int hash, float eval, int depth, int age){
    struct hash_table_entry* entry_index = table->table + (hash % table->size);
    table->num_entries++;
    // check if the entry is empty
    if (check_for_entry(entry_index, hash) == 1){
        table->num_entries--;
        
        // if the entry is populated and the value stored is deamed more relevant return
        if (compare_hash_entries(entry_index, depth, age) == 1){
            return;
            }
        }
    // if the entry is empty or the value stored is deamed less relevant add the new entry at the old entrys location
    entry_index->hash = hash;
    entry_index->eval = eval;
    entry_index->depth = depth;
    entry_index->age = age; 
}

// check if there is a hash entry for the given hash
// return 0 if there is no entry, 1 if there is an entry
int check_for_entry(struct hash_table_entry* entry_index, unsigned long long int hash){
    if (entry_index->hash == 0){
        return 0;
    }
    else {
        return 1;
    }
}

// returns the eval of a hash table entry if it exists and is the right hash value
// if the entry does not exits or is the wrong hash value, returns NAN
float get_hash_entry(struct hash_table *table, unsigned long long int hash){
    struct hash_table_entry* entry_index = table->table + (hash % table->size);
    if (entry_index->hash == hash){
        return entry_index->eval;
    }
    else{
        return NAN;
    }
}

// compare two hash table entries and decide the one to keep
// returns 1 if entry1 is better, 0 if the new values are better, if they are equal return 0
int compare_hash_entries(struct hash_table_entry *entry1, int depth, int age){
    if(entry1->depth > depth){
        return 1;
    }
    else {
        return 0;
    }
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
        else if (p2k & (1l << i)){
            hash = hash ^ table->piece_hash_diff[i + 192];
        }
    }
    return hash;
}

// compute the hash table piece diffs for quickly computing hashes of boards
unsigned long long int* compute_piece_hash_diffs(){
    srand(time(NULL));
    long long int* piece_hash_diffs = (long long int*)malloc(sizeof(long long int) * (64 * 4));
    for (int i = 0; i < (64 * 4); i++){
        piece_hash_diffs[i] = rand_num();
    }
    return piece_hash_diffs;
}

// sudo random number generator
long long int rand_num(){
    return (((unsigned long long int)rand() << 48) | ((unsigned long long int)rand() << 32) | ((unsigned long long int)rand() << 16) | (unsigned long long int)rand());
}

// frees the hash table
void free_hash_table(struct hash_table *table){
    free(table->table);
    free(table);
}

