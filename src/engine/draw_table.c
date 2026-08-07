// a small hash table that stores the number of times a position has been reached in the current line of play

// A power of two, so the index is a mask rather than a 64-bit division. This
// table is touched THREE times per node (add on the way down, get on arrival,
// remove on the way back up), so it was running more divisions per node than the
// transposition table did.
//
// It is also direct mapped with no collision resolution: a position whose slot is
// already taken is simply not recorded, and its repetition then becomes invisible
// to the search. The live set is small (the current line, plus the seeded game
// history) so this was already rare, but the table costs 16 bytes an entry and
// making collisions negligible is 256 KB well spent.
#define DRAW_TABLE_SIZE 16384
#define DRAW_TABLE_MASK (DRAW_TABLE_SIZE - 1)

struct draw_table {
    struct draw_table_entry *table;
    int size;
    unsigned long long int collisions;
};

struct draw_table_entry {
    unsigned long long int hash;
    int count;
};

// create a draw table
struct draw_table* create_draw_table() {
    struct draw_table *table = malloc(sizeof(struct draw_table));
    table->size = DRAW_TABLE_SIZE;
    table->table = calloc(table->size, sizeof(struct draw_table_entry));
    return table;
}

void free_draw_table(struct draw_table *table) {
    free(table->table);
    free(table);
}

// add a new entry to the draw table
void add_draw_entry(struct draw_table *table, unsigned long long int hash) {
    int index = (int)(hash & DRAW_TABLE_MASK);
    
    if (table->table[index].hash == hash) {
        table->table[index].count++;
        return;
    }

    else if (table->table[index].hash == 0) {
        table->table[index].hash = hash;
        table->table[index].count++;
        return;
    }

    table->collisions++;
}

// remove an entry from the draw table
void remove_draw_entry(struct draw_table *table, unsigned long long int hash) {
    int index = (int)(hash & DRAW_TABLE_MASK);
    
    if (table->table[index].hash == hash) {
        table->table[index].count--;
        if (table->table[index].count == 0) {
            table->table[index].hash = 0;
        }
        return;
    }

    table->collisions++;
}

// get the number of times a position has been reached in the current line of play
int get_draw_entry(struct draw_table *table, unsigned long long int hash) {
    int index = (int)(hash & DRAW_TABLE_MASK);
    
    if (table->table[index].hash == hash) {
        return table->table[index].count;
    }

    else if (table->table[index].hash == 0) {
        return 0;
    }

    table->collisions++;
    return 0;
}

