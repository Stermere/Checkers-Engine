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
    // static evaluation recorded at each ply, and the side it was relative to.
    // The search reads these back to tell whether the side to move is doing
    // better than it was on its previous turn. The player has to be stored
    // alongside because a multi jump does not flip the side to move, so plies
    // and turns are not the same thing here - depth_abs - 2 is NOT reliably the
    // same player's last turn.
    short* static_eval_stack;
    char* static_eval_player;
    int static_stack_size;
    clock_t start_time;
    double time_limit;

    // keep some interesting data
    long long int nodes;
    long long int avg_depth;
    int extended_depth;
};

struct search_results {
    int num_moves;
    short moves[96];      // root moves recorded during the last completed root pass
    int evals[96];        // eval of each root move (parallel to moves)
    short best_move;      // best root move of the last completed iteration
    int best_eval;        // eval of that move
};

struct board_evaler* board_evaler_constructor(long long p1_piece_loc, long long p2_piece_loc, int search_depth, double time_limit, clock_t start_time){
    struct board_evaler* evaler = (struct board_evaler*)malloc(sizeof(struct board_evaler));
    evaler->piece_pos_map_p1 = compute_piece_pos_p1();
    evaler->piece_pos_map_p2 = compute_piece_pos_p2(evaler->piece_pos_map_p1);
    evaler->king_pos_map = compute_king_pos();
    evaler->nodes = 0ll;
    evaler->avg_depth = 0ll;
    evaler->initial_piece_count_p1 = get_bits_set(p1_piece_loc);
    evaler->initial_piece_count_p2 = get_bits_set(p2_piece_loc);
    long long int hash_table_size = 1ll << 22;
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
    // sized and indexed exactly like the nnue stack above (depth_abs), with the
    // same slack for quiescence and jump chains; the search bounds-checks before
    // touching it so overrunning the slack costs a heuristic, not correctness
    evaler->static_stack_size = search_depth * 4 + 64;
    evaler->static_eval_stack = malloc(sizeof(short) * evaler->static_stack_size);
    evaler->static_eval_player = malloc(sizeof(char) * evaler->static_stack_size);
    for (int i = 0; i < evaler->static_stack_size; i++){
        evaler->static_eval_stack[i] = NO_EVAL;
        evaler->static_eval_player[i] = 0;
    }
    evaler->search_results = malloc(sizeof(struct search_results));
    evaler->search_results->num_moves = 0;
    evaler->search_results->best_move = NO_MOVE;
    evaler->search_results->best_eval = 0;
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



