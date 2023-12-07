// a small hash table that stores the number of times a position has been reached in the current line of play

#define DRAW_TABLE_SIZE 8000

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
    int index = hash % table->size;
    
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
    int index = hash % table->size;
    
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
    int index = hash % table->size;
    
    if (table->table[index].hash == hash) {
        return table->table[index].count;
    }

    else if (table->table[index].hash == 0) {
        return 0;
    }

    table->collisions++;
    return 0;
}

