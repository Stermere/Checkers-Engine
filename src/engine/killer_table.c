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
    table->size = max_depth * 4 + 30; // oversized so extensions cause no issues
    table->table = (struct killer_entry*)calloc(table->size, sizeof(struct killer_entry));
    return table;
}

// clear the killer table for expanding the search
struct killer_entry get_depth_moves(struct killer_table *table, int depth){
    return table->table[depth];
}

// the slot for a given ply, or a scratch slot if the ply is past the end.
//
// The table is sized from `search_depth`, but it is indexed by `depth_abs`, which
// is not bounded by search_depth: quiescence keeps descending while captures are
// available and free jump chains do not consume a ply. Today the slack covers the
// worst case (a capture chain is bounded by the pieces on the board), so this is
// not a live overrun - but it is bounded by an argument about the rules rather
// than by anything in the code, and the NNUE stack and the static-eval stack both
// already range check the exact same index. Killers are a heuristic, so falling
// back to a shared scratch entry costs move ordering at absurd depths, nothing else.
static struct killer_entry KILLER_SCRATCH = {0, 0};

struct killer_entry* killer_slot(struct killer_table *table, int depth){
    if (depth < 0 || depth >= table->size){
        return &KILLER_SCRATCH;
    }
    return &table->table[depth];
}

// update a move in the killer table
// updating the move works by replacing the second move first and if it is played again, it is replaces the first move
void update_killer_table(struct killer_table *table, int depth, int move_start, int move_end){
    struct killer_entry *entry = killer_slot(table, depth);
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
