// Author: Collin Kees
// board evaluation and hashing functions

#include <time.h>
#include <stdlib.h>
#include "set.c"
#include "hash_table.c"
#include "killer_table.c"
#include "draw_table.c"
#include "endgame_db.c"

// ---- evaluation source ----
// 0: the hand written calculate_eval below
// 1: the quantized neural network in nnue.c
//
// The current engine is the network; the handcrafted eval lives on in the
// OLD_ENGINE reference build (build_old.bat -> search_engine_old), which is the
// same mechanism the search feature flags already use. So an A/B strength
// measurement is "play search_engine against search_engine_old", with no
// third build and no git revert.
//
// The net is compiled in either way, because the Python hooks that
// src/python/nnue/verify_nnue_c.py uses to hold nnue.c to a bit-exact standard
// must exist regardless of which eval the search calls.
#ifdef OLD_ENGINE
#define USE_NNUE 0
#else
#define USE_NNUE 1
#endif

// included after USE_NNUE is known: nnue.c compiles its incremental accumulator
// stack either way (the Python test hooks need it), but the search only pays
// for it in the build that actually evaluates with the net
#include "nnue.c"


// define some functions
int* compute_piece_pos_p1();
int* compute_piece_pos_p2();
int* compute_king_pos();
int* init_distance_table();
struct board_evaler;
int calculate_eval(long long p1, long long p2, long long p1k, long long p2k, struct set* piece_loc, struct board_evaler* evaler);
int get_closest_enemy_dist(long long p1, long long p2, long long p1k, long long p2k, int pos, int type, int* piece_loc_array, int num_pieces, struct board_evaler* evaler);
int evaluate_pos(int type, int pos, struct board_evaler* evaler);
char* compute_offsets();
int king_dist(int pos, int player, int num_pieces);
int tail_pins(long long p1, long long p2, long long p1k, long long p2k);
int has_any_jump(long long p1, long long p2, long long p1k, long long p2k, int player);
int has_any_move(long long p1, long long p2, long long p1k, long long p2k, int player);
int can_jump_from(long long p1, long long p2, long long p1k, long long p2k, int pos, int piece_type);
int calculatePieceBonus(int num_pieces);
unsigned long long* compute_runaway_cones(int player);


// struct to hold the data for the hash table and other data related to getting a evaluation for a board
struct board_evaler{
    int *piece_pos_map_p1;
    int *piece_pos_map_p2;
    int *king_pos_map;
    int search_depth;
    int max_depth;
    int initial_piece_count_p1;
    int initial_piece_count_p2;
    struct hash_table* hash_table;
    struct draw_table* draw_table;
    struct killer_table* killer_table;
    struct search_results* search_results;
    int* dist_arr;
    char* piece_offsets;
    long long* history; // history heuristic scores: [player-1][from][to]
    unsigned long long* cone_p1; // promotion cones: squares a p1 man at [pos] may traverse to reach row 0
    unsigned long long* cone_p2; // same for p2 men heading to row 7
    struct nnue_stack* nnue; // per-ply incremental accumulators (NULL in the OLD_ENGINE build)
    clock_t start_time;
    double time_limit;

    // keep some interesting data
    long long int nodes;
    long long int avg_depth;
    // how many of those nodes actually reached the evaluation. The share matters
    // because the eval is the most expensive thing a node can do and most nodes
    // never do it - a transposition hit, a database hit or a repetition draw all
    // return first - so "what does a node cost" is unanswerable without it.
    long long int evals;
    int extended_depth;
    // The deepest ply the search actually reached, counting quiescence.
    // extended_depth deliberately does not: it only counts nodes that still had
    // depth budget left (depth >= 0), so it stops at the nominal horizon plus
    // extensions and says nothing about the capture chains past it - which in
    // checkers, where captures are mandatory and can run a long way, is most of
    // what "how deep did it look" means to someone reading it. Reported next to
    // extended_depth so the pair is not mistaken for one number.
    int max_ply;
#if SEARCH_DIAG
    long long int diag_capped_nodes;   // nodes sitting at the max_depth cap
    long long int diag_single_reply;   // nodes with exactly one legal move
    long long int diag_single_reply_payable;  // ...that still cost a ply
    long long int diag_single_reply_jump;     // ...of those, capture nodes
#endif
};

struct search_results {
    int num_moves;
    short moves[96];      // root moves recorded during the last completed root pass
    int evals[96];        // eval of each root move (parallel to moves)
    short best_move;      // best root move of the last completed iteration
    int best_eval;        // eval of that move

    // The best root move of the iteration currently IN PROGRESS, committed the
    // moment a root move raises alpha instead of at the end of the iteration.
    // Without this an iteration that runs out of time is discarded whole, which
    // throws away most of a move's thinking time - the branching factor means the
    // last, unfinished iteration is usually larger than every completed one put
    // together, and it is exactly the iteration that finds a refutation of the
    // move the previous depth liked.
    //
    // Adopting it is sound because the comparison is at a FIXED depth: the move
    // ordering searches the previous iteration's best move first, so it is
    // `iter_best_move` until some later move returns a strictly better score at
    // the same depth. If nothing beat it the value is a no-op; if something did,
    // it beat it fairly.
    short iter_best_move;
    int iter_best_eval;
    int iter_scored;      // how many root moves of this iteration have finished
};

struct board_evaler* board_evaler_constructor(long long p1_piece_loc, long long p2_piece_loc, int search_depth, double time_limit, clock_t start_time){
    struct board_evaler* evaler = (struct board_evaler*)malloc(sizeof(struct board_evaler));
    evaler->piece_pos_map_p1 = compute_piece_pos_p1();
    evaler->piece_pos_map_p2 = compute_piece_pos_p2(evaler->piece_pos_map_p1);
    evaler->king_pos_map = compute_king_pos();
    evaler->nodes = 0ll;
    evaler->avg_depth = 0ll;
    evaler->evals = 0ll;
    evaler->max_ply = 0;
#if SEARCH_DIAG
    evaler->diag_capped_nodes = 0ll;
    evaler->diag_single_reply = 0ll;
    evaler->diag_single_reply_payable = 0ll;
    evaler->diag_single_reply_jump = 0ll;
#endif
    evaler->initial_piece_count_p1 = get_bits_set(p1_piece_loc);
    evaler->initial_piece_count_p2 = get_bits_set(p2_piece_loc);
    // Transposition table size in entries, log2. 22 is 4Mi entries = 64MB at
    // 16 bytes an entry, which is the desktop engine. The browser build turns
    // it down: a tab also holds the endgame slices and the page itself, and at
    // the one second per move the GUI asks for, a table this large is never
    // close to full anyway.
#ifndef TT_ENTRIES_LOG2
#define TT_ENTRIES_LOG2 22
#endif
    long long int hash_table_size = 1ll << TT_ENTRIES_LOG2;
    evaler->hash_table = init_hash_table(hash_table_size);
    evaler->draw_table = create_draw_table();
    evaler->killer_table = init_killer_table(search_depth);
    evaler->dist_arr = init_distance_table();
    evaler->piece_offsets = compute_offsets();
    evaler->history = calloc(2 * 64 * 64, sizeof(long long));
    evaler->cone_p1 = compute_runaway_cones(1);
    evaler->cone_p2 = compute_runaway_cones(2);
#if USE_NNUE
    // one accumulator per ply. Sized like the killer table (which is indexed by
    // depth_abs the same way) with extra slack for quiescence and for jump
    // chains, which do not consume a ply. Running off the end is not a bug -
    // nnue_stack_eval falls back to a from-scratch evaluation - so the slack is
    // about speed, not correctness.
    evaler->nnue = nnue_stack_create(search_depth * 4 + 64);
#else
    evaler->nnue = NULL;
#endif
    evaler->search_results = malloc(sizeof(struct search_results));
    evaler->search_results->num_moves = 0;
    evaler->search_results->best_move = NO_MOVE;
    evaler->search_results->best_eval = 0;
    evaler->search_results->iter_best_move = NO_MOVE;
    evaler->search_results->iter_best_eval = 0;
    evaler->search_results->iter_scored = 0;
    evaler->start_time = start_time;
    evaler->time_limit = time_limit;
    evaler->extended_depth = 0;
    return evaler;
}

// get the evaluation for a board given the board state.
// the returned score is always relative to `player` (negamax convention).
// `depth_abs` is the ply this board sits at, which is how the network finds its
// incremental accumulator; it is ignored by the handcrafted eval.
int get_eval(long long p1, long long p2, long long p1k, long long p2k, int player, struct set* piece_loc, struct board_evaler* evaler, int depth_abs){
    evaler->evals++;
#if USE_NNUE
    // the net encodes the board from the side to move's point of view, so its
    // output is already player-relative and must NOT be negated again here.
    (void)piece_loc;
    return nnue_stack_eval(evaler->nnue, depth_abs, player, p1, p2, p1k, p2k);
#else
    (void)depth_abs;
    int eval = calculate_eval(p1, p2, p1k, p2k, piece_loc, evaler);

    // calculate_eval is player-1 relative; invert it for negamax
    return player == 1 ? eval : -eval;
#endif
}


// calculate the board evaluation
int calculate_eval(long long p1, long long p2, long long p1k, long long p2k, struct set* piece_loc, struct board_evaler* evaler){
    int eval = 0;
    int p1_piece_distance = 0;
    int p2_piece_distance = 0;
    int num_pieces = populate_set_array(piece_loc);
    int p1num = 0;
    int p1knum = 0;
    int p2num = 0;
    int p2knum = 0;
    long long all_pieces = p1 | p2 | p1k | p2k;
    for (int i = 0; i < num_pieces; i++){
        if (p1 >> piece_loc->array[i] & 1){
            int pos = piece_loc->array[i];
            eval += 50;
            eval += evaluate_pos(1, pos, evaler);
            eval += king_dist(pos, 1, num_pieces);
            // runaway checker: empty promotion cone, no enemy kings, and no enemy
            // men between this man and the back rank means promotion cannot be stopped
            if (p2k == 0 && (evaler->cone_p1[pos] & all_pieces) == 0
                && (p2 & ((1ull << ((pos / 8) * 8)) - 1)) == 0){
                eval += 20 + 3 * (7 - (pos / 8));
            }
            p1num++;

        }
        else if (p2 >> piece_loc->array[i] & 1){
            int pos = piece_loc->array[i];
            eval -= 50;
            eval -= evaluate_pos(2, pos, evaler);
            eval -= king_dist(pos, 2, num_pieces);
            if (p1k == 0 && (evaler->cone_p2[pos] & all_pieces) == 0
                && (p1 & ~((1ull << ((pos / 8 + 1) * 8)) - 1)) == 0){
                eval -= 20 + 3 * (pos / 8);
            }
            p2num++;

        }
        else if (p1k >> piece_loc->array[i] & 1){
            eval += 70;
            eval += evaluate_pos(3, piece_loc->array[i], evaler);
            //p1_piece_distance += get_closest_enemy_dist(p1, p2, p1k, p2k, piece_loc->array[i], 3, piece_loc->array, num_pieces, evaler);
            p1num++;
            p1knum++;

        }
        else if (p2k >> piece_loc->array[i] & 1){
            eval -= 70;
            eval -= evaluate_pos(4, piece_loc->array[i], evaler);
            //p2_piece_distance -= get_closest_enemy_dist(p1, p2, p1k, p2k, piece_loc->array[i], 4, piece_loc->array, num_pieces, evaler);
            p2num++;
            p2knum++;
        }
    }

    // evaluate the tail pins
    eval += tail_pins(p1, p2, p1k, p2k);
    
    // give the player with the most pieces a bonus
    if (p1num > p2num){
        eval += calculatePieceBonus(p1num);
        eval += p1_piece_distance;
    }
    else if (p2num > p1num){
        eval -= calculatePieceBonus(p2num);
        eval += p2_piece_distance;
    }


    // Give a bonus to players with structures on the board that are often good
    // Right Lock pattern
    if ((p1 & 0x8000000000 ^ 0x8000000000) == 0 && ((p2 & 0x40000000 ^ 0x40000000) == 0)){
        eval += 20;
    }
    if ((p2 & 0x1000000 ^ 0x1000000) == 0 && ((p1 & 0x200000000 ^ 0x200000000) == 0)){
        eval -= 20;
    }

    // Triangle pattern
    if ((p1 & 0x5020000000000000 ^ 0x5020000000000000) == 0) {
        eval += 10;
    }
    if ((p2 & 0x40a ^ 0x40a) == 0) {
        eval -= 10;
    }

    // Oreo Pattern
    if ((p1 & 0x1408000000000000 ^ 0x1408000000000000) == 0){
        eval += 10;
    }
    if ((p2 & 0x1028 ^ 0x1028) == 0){
        eval -= 10;
    }   

    // Bridge Pattern 
    if ((p1 & 0x4400000000000000 ^ 0x4400000000000000) == 0){
        eval += 15;
    }
    if ((p2 & 0x22 ^ 0x22) == 0){
        eval -= 15;
    }

    // Dog pattern
    if (((p1 & 0x4000000000000000 ^ 0x4000000000000000) == 0) && ((p2 & 0x80000000000000 ^ 0x80000000000000) == 0)){
        eval += 5;
    }
    if (((p2 & 0x2 ^ 0x2) == 0) && ((p1 & 0x100 ^ 0x100) == 0)){
        eval -= 5;
    }

    // King in the corner pattern
    if ((p1k & 0x80 ^ 0x80) == 0){
        eval -= 20;
    }
    if ((p2k & 0x100000000000000 ^ 0x100000000000000) == 0){
        eval += 20;
    }

    return eval;
}

// evaluate the position of a piece (type comes in two flavors 0 for normal piece and 1 for a king)
int evaluate_pos(int type, int pos, struct board_evaler* evaler){
    if (type == 1){
        return evaler->piece_pos_map_p1[pos];
    }
    else if (type == 2){
        return evaler->piece_pos_map_p2[pos];
    }
    else if (type == 3 || type == 4){
        return evaler->king_pos_map[pos];
    }

    return 0;
}

int calculatePieceBonus(int num_pieces) {
    if (num_pieces <= 2) return 200;
    if (num_pieces == 3) return 150;
    if (num_pieces == 4) return 100;
    if (num_pieces == 5) return 60;
    if (num_pieces == 6) return 30;
    return 10;  // For num_pieces > 6
}

// get the distance to the closest enemy piece
int get_closest_enemy_dist(long long p1, long long p2, long long p1k, long long p2k, int pos, int type, int* piece_loc_array, int num_pieces, struct board_evaler* evaler){
    // if the number of pieces is less than 8 begin to use the distance table
    if (num_pieces < 8) {
        return 0;
    }
    
    long long check;
    if (type == 3) {
        check = p2 | p2k;
    } else {
        check = p1 | p1k;
    }

    int dist = 14;
    for (int i = 0; i < num_pieces; i++){
        if (!(check >> piece_loc_array[i] & 1)){
            continue;
        }

        int new_dist = evaler->dist_arr[pos * 64 + piece_loc_array[i]];
        if (new_dist < dist){
            dist = new_dist;
        }
    }

    return 14 - dist;
}

int king_dist(int pos, int player, int num_pieces) {
    if (num_pieces > 12) {
        return 0;
    }

    if (player == 1) {
        return 8 - (pos / 8);
    } else {
        return pos / 8;
    }
}

// Give the player with more tail pins a bonus
// (a tail pin is a king piece holding back two enemy pieces from behind)
// column masks keep the diagonal shifts from wrapping around the board edge
int tail_pins(long long p1, long long p2, long long p1k, long long p2k) {
    const long long COLS_0_5 = 0x3F3F3F3F3F3F3F3Fll; // king col <= 5 for the +9/+7-toward-higher-cols chains
    const long long COLS_2_7 = 0xFCFCFCFCFCFCFCFCll; // king col >= 2 for the chains toward lower cols

    long long p1p = (p1k & COLS_0_5) & (p2 >> 9) & (p2 >> 18);
    p1p |= (p1k & COLS_2_7) & (p2 >> 7) & (p2 >> 14);

    long long p2p = (p2k & COLS_2_7) & (p1 << 9) & (p1 << 18);
    p2p |= (p2k & COLS_0_5) & (p1 << 7) & (p1 << 14);

    return (get_bits_set(p1p) - get_bits_set(p2p)) * 5;
}

// column masks that stop the diagonal shifts below from wrapping around the board edge
#define JUMP_COL_GE2 0xFCFCFCFCFCFCFCFCull // squares with column >= 2 (room to land two to the left)
#define JUMP_COL_LE5 0x3F3F3F3F3F3F3F3Full // squares with column <= 5 (room to land two to the right)

// fast branchless-ish test for "does this player have a capture available right now".
// a jump exists when a piece has an enemy one diagonal step away and an empty
// square one further step along the same diagonal.
// used by the search to tell tactical positions from quiet ones so that lines
// containing captures are never reduced.
int has_any_jump(long long p1, long long p2, long long p1k, long long p2k, int player){
    unsigned long long occupied = (unsigned long long)(p1 | p2 | p1k | p2k);
    unsigned long long empty = ~occupied;
    unsigned long long up, down, enemy;

    if (player == 1){
        // p1 men move toward row 0 (decreasing index), p1 kings move both ways
        up = (unsigned long long)(p1 | p1k);
        down = (unsigned long long)p1k;
        enemy = (unsigned long long)(p2 | p2k);
    } else {
        up = (unsigned long long)p2k;
        down = (unsigned long long)(p2 | p2k);
        enemy = (unsigned long long)(p1 | p1k);
    }

    // up-left (-9): enemy on pos-9, landing square on pos-18
    if (up & JUMP_COL_GE2 & (enemy << 9) & (empty << 18)) return 1;
    // up-right (-7): enemy on pos-7, landing square on pos-14
    if (up & JUMP_COL_LE5 & (enemy << 7) & (empty << 14)) return 1;
    // down-left (+7): enemy on pos+7, landing square on pos+14
    if (down & JUMP_COL_GE2 & (enemy >> 7) & (empty >> 14)) return 1;
    // down-right (+9): enemy on pos+9, landing square on pos+18
    if (down & JUMP_COL_LE5 & (enemy >> 9) & (empty >> 18)) return 1;

    return 0;
}

// column masks for a single diagonal step, the one-square counterparts of the
// jump masks above (a step left needs column >= 1, a step right column <= 6)
#define STEP_COL_GE1_E 0xFEFEFEFEFEFEFEFEull
#define STEP_COL_LE6_E 0x7F7F7F7F7F7F7F7Full

// "does this player have any legal move at all", answered set-wise.
//
// The quiescence search asks this at every leaf where no capture exists, to tell
// a quiet position (evaluate it) from a loss (the side to move is stuck). It used
// to answer it by running generate_all_moves a second time over every friendly
// piece and looking at the count - building a whole move list to test it against
// zero, at what is the most common node in the tree.
//
// Eight shift-and-mask tests over the whole bitboard replace that, and they are
// the same tests generate_moves makes per piece, so the answer is identical by
// construction: a jump exists exactly when has_any_jump says so, and otherwise a
// move exists exactly when some piece has an empty square one diagonal step away
// in a direction it may travel.
int has_any_move(long long p1, long long p2, long long p1k, long long p2k, int player){
    if (has_any_jump(p1, p2, p1k, p2k, player)) return 1;

    unsigned long long occupied = (unsigned long long)(p1 | p2 | p1k | p2k);
    unsigned long long empty = ~occupied;
    unsigned long long up, down;

    if (player == 1){
        up = (unsigned long long)(p1 | p1k);
        down = (unsigned long long)p1k;
    } else {
        up = (unsigned long long)p2k;
        down = (unsigned long long)(p2 | p2k);
    }

    if (up & STEP_COL_GE1_E & (empty << 9)) return 1;   // up-left (-9)
    if (up & STEP_COL_LE6_E & (empty << 7)) return 1;   // up-right (-7)
    if (down & STEP_COL_GE1_E & (empty >> 7)) return 1; // down-left (+7)
    if (down & STEP_COL_LE6_E & (empty >> 9)) return 1; // down-right (+9)

    return 0;
}

// "can the piece standing on `pos` capture from there", for a piece known to be
// of type `piece_type`. This is generate_moves' jump test for a single square,
// without the move list: get_next_board_state only ever needed the yes/no, and
// was calling the full generator into a throwaway buffer after every capture to
// get it.
int can_jump_from(long long p1, long long p2, long long p1k, long long p2k, int pos, int piece_type){
    unsigned long long occupied = (unsigned long long)(p1 | p2 | p1k | p2k);
    unsigned long long empty = ~occupied;
    unsigned long long b = 1ull << pos;
    unsigned long long enemy = (piece_type == 1 || piece_type == 3)
                                  ? (unsigned long long)(p2 | p2k)
                                  : (unsigned long long)(p1 | p1k);
    int can_up = (piece_type != 2);
    int can_down = (piece_type != 1);

    if (can_up){
        if (b & JUMP_COL_GE2 & (enemy << 9) & (empty << 18)) return 1;
        if (b & JUMP_COL_LE5 & (enemy << 7) & (empty << 14)) return 1;
    }
    if (can_down){
        if (b & JUMP_COL_GE2 & (enemy >> 7) & (empty >> 14)) return 1;
        if (b & JUMP_COL_LE5 & (enemy >> 9) & (empty >> 18)) return 1;
    }
    return 0;
}



// compute the array of piece positions containing how good it is to have a piece at each position
int* compute_piece_pos_p1() {
    int *eval_table = (int*)malloc(sizeof(int) * 64);
    int table[8][8] = { 
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 3, 0, 2, 0, 0},
        {0, 0, 1, 0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0, 1, 0, 0},
        {0, 0, 1, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1},
        {0, 0, 3, 0, 3, 0, 3, 0}
    };
    for (int i = 0; i < 64; i++){
        eval_table[i] = table[i / 8][i % 8];
    }
    
    return eval_table;
}

int* compute_piece_pos_p2(int* piece_pos_p1) {
    int *eval_table = (int*)malloc(sizeof(int) * 64);
    // flip the table
    for (int i = 0; i < 64; i++){
        eval_table[i] = piece_pos_p1[63 - i];
    }

    return eval_table;
}

int* compute_king_pos() {
    int *eval_table = (int*)malloc(sizeof(int) * 64);
    // give center squares a bonus
    int table[8][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 5, 0, 0, 0, 0, 0},
        {0, 5, 0, 5, 0, 4, 0, 0},
        {0, 0, 5, 0, 4, 0, 0, 0},
        {0, 0, 0, 4, 0, 5, 0, 0},
        {0, 0, 4, 0, 5, 0, 5, 0},
        {0, 0, 0, 0, 0, 5, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };
        
    for (int i = 0; i < 64; i++){
        eval_table[i] = table[i / 8][i % 8];
    }
    
    return eval_table;
}

// compute the valid move directions for every location on the board
// saves the results the the pointer passed to the function in the form of 4 chars either 0 or 1
// 1 means it is a valid direction to move in
// the first char is the direction of the move to the left and up
// the second char is the direction of the move to the right and up
// the third char is the direction of the move to the left and down
// the fourth char is the direction of the move to the right and down
char* compute_offsets(){
    char* offsets = malloc(sizeof(char) * 64 * 4);

    // init the array to true
    for (int i = 0; i < 64 * 4; i++){
        offsets[i] = 1;
    }
    // for moves that are not possible set them to false in the array of moves
    for (int i = 0; i < 64; i++){
        if (i % 8 == 0){
            offsets[i * 4 + 0] = 0;
            offsets[i * 4 + 2] = 0;
        }
        if ((i + 1) % 8 == 0){
            offsets[i * 4 + 1] = 0;
            offsets[i * 4 + 3] = 0;
        }
        if (i > 55){
            offsets[i * 4 + 2] = 0;
            offsets[i * 4 + 3] = 0;   
        }
        if (i < 8){
            offsets[i * 4 + 0] = 0;
            offsets[i * 4 + 1] = 0; 
        }
    }

    return offsets;
}

// precompute the promotion cone for a man on each square: every square it could
// possibly traverse on its way to the promotion row (diagonal spread, clipped at edges)
unsigned long long* compute_runaway_cones(int player){
    unsigned long long* cones = (unsigned long long*)malloc(sizeof(unsigned long long) * 64);
    for (int pos = 0; pos < 64; pos++){
        unsigned long long m = 0;
        int r = pos / 8;
        int c = pos % 8;
        if (player == 1){
            // p1 men move toward row 0
            for (int k = 1; k <= r; k++){
                for (int dc = -k; dc <= k; dc += 2){
                    int cc = c + dc;
                    if (cc >= 0 && cc <= 7){
                        m |= 1ull << ((r - k) * 8 + cc);
                    }
                }
            }
        } else {
            // p2 men move toward row 7
            for (int k = 1; k <= 7 - r; k++){
                for (int dc = -k; dc <= k; dc += 2){
                    int cc = c + dc;
                    if (cc >= 0 && cc <= 7){
                        m |= 1ull << ((r + k) * 8 + cc);
                    }
                }
            }
        }
        cones[pos] = m;
    }
    return cones;
}

// compute the distance between two bit positions
int distance(int pos1, int pos2){
    return abs(pos1 % 8 - pos2 % 8) + abs(pos1 / 8 - pos2 / 8);

}

// precompute the distance table. max distance is 1 
int* init_distance_table(){
    int* dist_arr = (int*)malloc(sizeof(int) * 64 * 64);
    for (int i = 0; i < 64; i++){
        for (int j = 0; j < 64; j++){
            dist_arr[i * 64 + j] = distance(i, j);
        }
    }

    return dist_arr;
}



