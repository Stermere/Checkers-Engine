// Killer table for storing refutation tables for each depth
// Author: Collin Kees

#include <stdlib.h>

struct killer_entry {
    short move1;
    short move2;
};

// This is a table of killer moves for each depth. like the transposition table 
// moves are stored top byte is the start square, bottom byte is the end square.
struct killer_table {
    int size;
    struct killer_entry *table;
};

// initialize the killer table
struct killer_table* init_killer_table(int max_depth){
    struct killer_table* table = (struct killer_table*)malloc(sizeof(struct killer_table));
    table->size = max_depth * 4 + 20; // add 20 to ensure extenstions cause no issues
    table->table = (struct killer_entry*)malloc(sizeof(struct killer_entry) * table->size);
    return table;
}

// clear the killer table for expanding the search
struct killer_entry get_depth_moves(struct killer_table *table, int depth){
    return table->table[depth];
}

// update a move in the killer table
// updating the move works by replacing the second move first and if it is played again, it is replaces the first move
void update_killer_table(struct killer_table *table, int depth, int move_start, int move_end){
    struct killer_entry *entry = &table->table[depth];
    // if the move is already in the table, replace the second move with the new move
    // unless, the second move is the new move then switch the first and second moves
    short move = move_start << 8 | move_end;
    if (entry->move2 == move){
        // swap the first and second moves
        short temp = entry->move1;
        entry->move1 = entry->move2;
        entry->move2 = temp;
    } else if (entry->move1 != move){
        // replace the second move with the new move
        entry->move2 = move;
    }
}
