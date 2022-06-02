// killer table for storing refutation tables for each depth
// Author: Collin Kees

#include <stdlib.h>

struct killer_entry {
    int move1_start;
    int move1_end;
    float score1;
    int move2_start;
    int move2_end;
    float score2;
};

// This is a table of killer moves for each depth. (initialized to allow depths of 100)
struct killer_table {
    int size;
    struct killer_entry *table;
};

// initialize the killer table
struct killer_table* init_killer_table(int depth){
    struct killer_table* table = (struct killer_table*)malloc(sizeof(struct killer_table));
    table->size = depth;
    table->table = (struct killer_entry*)malloc(sizeof(struct killer_entry) * depth);
    return table;
}

// clear the killer table for expanding the search
void clear_killer_table(struct killer_table *table){
    for(int i = 0; i < table->size; i++){
        table->table[i].move1_start = 0;
        table->table[i].move1_end = 0;
        table->table[i].score1 = 0;
        table->table[i].move2_start = 0;
        table->table[i].move2_end = 0;
        table->table[i].score2 = 0;
    }
}
