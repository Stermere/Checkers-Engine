// Author: Collin Kees
// board evaluation and hashing functions

#include <stdlib.h>
#include "set.c"
#include "hash_table.c"
#include "killer_table.c"
#include "draw_table.c"
//#include "neural_net.c"


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
int calculatePieceBonus(int num_pieces);


// struct to hold the data for the hash table and other data related to getting a evaluation for a board
struct board_evaler{
    int *piece_pos_map_p1;
    int *piece_pos_map_p2;
    int *king_pos_map;
    int search_depth;
    int max_depth;
    int initial_piece_count_p1;
    int initial_piece_count_p2;
    struct neural_net *NN_evaler;
    struct hash_table* hash_table;
    struct draw_table* draw_table;
    struct killer_table* killer_table;
    int* dist_arr;
    char* piece_offsets;
    clock_t start_time;
    double time_limit;

    // keep some interesting data
    long long int nodes;
    long long int avg_depth;
    int extended_depth;

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
    // load the neural network
    //evaler->NN_evaler = load_network_from_file("neural_net/neural_net");
    long long int hash_table_size = 1 << 20;
    evaler->hash_table = init_hash_table(hash_table_size);
    evaler->draw_table = create_draw_table();
    evaler->killer_table = init_killer_table(search_depth);
    evaler->dist_arr = init_distance_table();
    evaler->piece_offsets = compute_offsets();
    evaler->start_time = start_time;
    evaler->time_limit = time_limit;
    evaler->extended_depth = 0;
    return evaler;
}

// get the evaluation for a board given the board state
int get_eval(long long p1, long long p2, long long p1k, long long p2k, int player, struct set* piece_loc, struct board_evaler* evaler){
    int eval = calculate_eval(p1, p2, p1k, p2k, piece_loc, evaler);

    // test neural net
    //float eval = (float)get_output(evaler->NN_evaler, p1, p2, p1k, p2k) - 100.0; // neural_net
    
    // invert the eval for negmax
    return player == 1 ? eval : -eval;
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
    for (int i = 0; i < num_pieces; i++){
        if (p1 >> piece_loc->array[i] & 1){
            eval += 50;
            eval += evaluate_pos(1, piece_loc->array[i], evaler);
            eval += king_dist(piece_loc->array[i], 1, num_pieces);
            //eval += pieces_ahead(p1, p2, p1k, p2k, 1, piece_loc->array[i], num_pieces);
            p1num++;

        }
        else if (p2 >> piece_loc->array[i] & 1){
            eval -= 50;
            eval -= evaluate_pos(2, piece_loc->array[i], evaler);
            eval -= king_dist(piece_loc->array[i], 2, num_pieces);
            //eval -= pieces_ahead(p1, p2, p1k, p2k, 2, piece_loc->array[i], num_pieces);
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
int tail_pins(long long p1, long long p2, long long p1k, long long p2k) {
    long long p1p = p1k & (p2 >> 9) & (p2 >> 18);
    p1p |= p1k & (p2 >> 7) & (p2 >> 14);

    long long p2p = p2k & (p1 << 9) & (p1 << 18);
    p2p |= p2k & (p1 << 7) & (p1 << 14);

    return (get_bits_set(p1p) - get_bits_set(p2p)) * 5;
}

// returns the number of pieces ahead of the piece at pos
int pieces_ahead(long long p1, long long p2, long long p1k, long long p2k, int type, int pos, int num_pieces) {
    // get a mask of just the enemy pieces
    long long check;
    if (type == 1) {
        check = p2 | p2k;
    } else {
        check = p1 | p1k;
    }

    int count = 0;

    // mask out pieces that are not in front of the piece
    if (type == 1) {
        check &= 0xFFFFFFFFFFFFFFFF >> (64 - (pos - (pos % 8)));
    }
    else {
        check &= 0xFFFFFFFFFFFFFFFF << (pos + (pos % 8));
    }

    return 8 - get_bits_set(check);
}
     

int is_trapped_king(long long p1, long long p2, long long p1k, long long p2k, int type, int pos) {

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



