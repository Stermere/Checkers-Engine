// Author: Collin Kees
// board evaluation and hashing functions

#include <stdlib.h>
#include "Cset.h"
#include "Chash_table.h"
#include "Cneural_net.h"
#include "Ckiller_table.h"


// define some functions
float* compute_piece_pos_p1();
float* compute_piece_pos_p2();
float* compute_king_pos();
float* init_distance_table();
struct board_evaler;
float calculate_eval(long long p1, long long p2, long long p1k, long long p2k, struct set* piece_loc, struct board_evaler* evaler);
float get_closest_enemy_dist(long long p1, long long p2, long long p1k, long long p2k, int pos, int type, int* piece_loc_array, int num_pieces, struct board_evaler* evaler);
float is_runaway_piece(long long p1, long long p2, long long p1k, long long p2k, int type, int pos);
float evaluate_pos(int type, int pos, struct board_evaler* evaler);


// struct to hold the data for the hash table and other data related to getting a evaluation for a board
struct board_evaler{
    float *piece_pos_map_p1;
    float *piece_pos_map_p2;
    float *king_pos_map;
    int search_depth;
    int max_depth;
    struct neural_net *NN_evaler;
    struct hash_table* hash_table;
    struct killer_table* killer_table;
    float* dist_arr;
    clock_t start_time;
    double time_limit;

    // keep some interesting data
    long long int nodes;
    long long int avg_depth;
    int extended_depth;

};

struct board_evaler* board_evaler_constructor(int search_depth, double time_limit, clock_t start_time){
    struct board_evaler* evaler = (struct board_evaler*)malloc(sizeof(struct board_evaler));
    evaler->piece_pos_map_p1 = compute_piece_pos_p1();
    evaler->piece_pos_map_p2 = compute_piece_pos_p2();
    evaler->king_pos_map = compute_king_pos();
    evaler->nodes = 0ll;
    evaler->avg_depth = 0ll;
    // load the neural network
    //evaler->NN_evaler = load_network_from_file("neural_net/neural_net");
    //long long int hash_table_size = 1 << 15;
    long long int hash_table_size = 1 << 20;
    evaler->hash_table = init_hash_table(hash_table_size);
    evaler->killer_table = init_killer_table(search_depth);
    evaler->dist_arr = init_distance_table();
    evaler->start_time = start_time;
    evaler->time_limit = time_limit;
    evaler->extended_depth = 0;
    return evaler;
}

// get the evaluation for a board given the board state
float get_eval(long long p1, long long p2, long long p1k, long long p2k, int player, struct set* piece_loc, struct board_evaler* evaler){
    // there was no entry found so lets calculate it
    float eval = calculate_eval(p1, p2, p1k, p2k, piece_loc, evaler);

    // test neural net
    //float eval = (float)get_output(evaler->NN_evaler, p1, p2, p1k, p2k) - 100.0; // neural_net
    
    // invert the eval for negmax
    return player == 1 ? eval : -eval;
}


// calculate the board evaluation
float calculate_eval(long long p1, long long p2, long long p1k, long long p2k, struct set* piece_loc, struct board_evaler* evaler){
    float eval = 0;
    int piece_loc_array[64];
    int num_pieces = populate_array(piece_loc, piece_loc_array);
    int p1num = 0;
    int p1knum = 0;
    int p2num = 0;
    int p2knum = 0;
    for (int i = 0; i < num_pieces; i++){
        if (p1 >> piece_loc_array[i] & 1){
            eval += 3.0f;
            eval += evaluate_pos(1, piece_loc_array[i], evaler);
            p1num++;

        }
        else if (p2 >> piece_loc_array[i] & 1){
            eval -= 3.0f;
            eval -= evaluate_pos(2, piece_loc_array[i], evaler);
            p2num++;
            
        }
        else if (p1k >> piece_loc_array[i] & 1){
            eval += 5.0f;
            eval += evaluate_pos(3, piece_loc_array[i], evaler);
            eval += get_closest_enemy_dist(p1, p2, p1k, p2k, piece_loc_array[i], 3, piece_loc_array, num_pieces, evaler);
            p1num++;
            p1knum++;

        }
        else if (p2k >> piece_loc_array[i] & 1){
            eval -= 5.0f;
            eval -= evaluate_pos(4, piece_loc_array[i], evaler);
            eval -= get_closest_enemy_dist(p1, p2, p1k, p2k, piece_loc_array[i], 4, piece_loc_array, num_pieces, evaler);
            p2num++;
            p2knum++;
        }
    }

    // if a player has no pieces its a win
    if  (p2num == 0) {
        return 1000.0f;
    }
    else if (p1num == 0) {
        return -1000.0f;
    }
    
    // give the player with the most pieces a bonus
    if (p1num > p2num){
        eval += (15.0f * (p1num - p2num)) / (num_pieces);
        if (p2num < 3)
            eval += 4.0f;
        if (p2num < 2)
            eval += 10.0f;
    }
    else if (p2num > p1num){
        eval -= (15.0f * (p2num - p1num)) / (num_pieces);
        if (p1num < 3)
            eval -= 4.0f;
        else if (p1num < 2)
            eval -= 10.0f;
    }
    else if (p1num == p2num){
        if (p1knum > p2knum){
            eval += 10.0f / (num_pieces);
        }
        else if (p2knum > p1knum){
            eval -= 10.0f / (num_pieces);
        }
    }

    // give a bonus to players with structures on the board that are often good
    if (p1 & 0x4400000000000000 ^ 0x4400000000000000 == 0){
        eval += 0.5f;
    }
    if (p2 & 0x22 ^ 0x22 == 0){
        eval -= 0.5f;
    }

    return eval;
}

// evaluate the position of a piece (type comes in two flavors 0 for normal piece and 1 for a king)
float evaluate_pos(int type, int pos, struct board_evaler* evaler){
    if (type == 1){
        return evaler->piece_pos_map_p1[pos];
    }
    else if (type == 2){
        return evaler->piece_pos_map_p2[pos];
    }
    else if (type == 3){
        return evaler->king_pos_map[pos];
    }
    else if (type == 4){
        return evaler->king_pos_map[pos];
    }
}

// get the distance to the closest enemy piece
float get_closest_enemy_dist(long long p1, long long p2, long long p1k, long long p2k, int pos, int type, int* piece_loc_array, int num_pieces, struct board_evaler* evaler){
    // if the number of pieces is less than 10 begin to use the distance table
    if (num_pieces > 8) {
        return 0.0f;
    }
    
    long long check;
    if (type == 3) {
        check = p2 | p2k;
    } else {
        check = p1 | p1k;
    }

    float dist = 1.0f;
    for (int i = 0; i < num_pieces; i++){
        if (!(check >> piece_loc_array[i] & 1)){
            continue;
        }

        float new_dist = evaler->dist_arr[pos * 64 + piece_loc_array[i]];
        if (new_dist < dist){
            dist = new_dist;
        }
    }

    return 1.0f - dist;
}
     

float is_trapped_king(long long p1, long long p2, long long p1k, long long p2k, int type, int pos) {
    
}

// check if a piece is a runnaway piece (a piece that has a clear path to the other side of the board)
float is_runaway_piece(long long p1, long long p2, long long p1k, long long p2k, int type, int pos) {
    long long check;
    int dir;
    if (type == 3 || type == 4) {
        return 0.0f;
    }

    if (type == 1) {
        check = p2 | p2k;
        dir = -1;
    } else {
        check = p1 | p1k;
        dir = 1;
    }

    // make a bit make of every position that this piece could reach
    long long mask = 0;
    int i = pos;
    int placed = 0;


    i += dir * 7;
    while (i >= 0 && i < 64 && (i % 8 != 0 && i % 8 != 7)) {
        mask |= 1ll << i;
        i += dir * 7;
        placed = 1;
    }
    while (i >= 0 && i < 64 && placed == 1) {
        mask |= 1ll << i;
        i += dir * 8;
    }

    i = pos;
    i += dir * 9;
    placed = 0;
    while (i >= 0 && i < 64 && (i % 8 != 7 && i % 8 != 0)) {
        mask |= 1ll << i;
        i += dir * 9;
        placed = 1;
    }
    while (i >= 0 && i < 64 && placed == 1) {
        mask |= 1ll << i;
        i += dir * 8;
    }

    // fill in the middle
    i = pos + dir;
    long long masker = 0;
    while (i >= 0 && i < 64) {
        if ((mask & (1ll << i))) {
            masker = !masker;
            i += dir;
            continue;
        }

        mask |= masker << i;
        if (i % 8 == 0 || i % 8 == 7) {
            masker = 0;
        }
        i += dir;

    }
     
    // check if the mask has any enemy pieces in it
    if ((mask & check) == 0) {
        return 0.5f;
    }

    return 0.0f;
    
}

// compute the array of piece positions containing how good it is to have a piece at each position
float* compute_piece_pos_p1(){
    float *eval_table = (float*)malloc(sizeof(float) * 64);
    float table[8][8] = { 
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 3, 0, 3, 0, 0},
        {0, 0, 1, 1, 1, 1, 0, 0},
        {0, 1, 1, 1, 1, 1, 0, 0},
        {1, 0, 1, 1, 1, 1, 1, 0},
        {0, 2, 0, 2, 0, 2, 0, 2},
        {0, 0, 3, 0, 3, 0, 3, 0}
    };
    for (int i = 0; i < 64; i++){
        eval_table[i] = table[i / 8][i % 8]/ 10.0;
    }
    
    return eval_table;
}

float* compute_piece_pos_p2(){
    float *eval_table = (float*)malloc(sizeof(float) * 64);
    // mirror the table
    float table[8][8] = { 
        {0, 3, 0, 3, 0, 3, 0, 0},
        {2, 0, 2, 0, 2, 0, 2, 0},
        {0, 1, 1, 1, 1, 1, 0, 1},
        {0, 0, 1, 1, 1, 1, 1, 0},
        {0, 0, 1, 1, 1, 1, 0, 0},
        {0, 0, 3, 0, 3, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };
    for (int i = 0; i < 64; i++){
        eval_table[i] = table[i / 8][i % 8]/ 10.0;
    }

    return eval_table;
}



// compute the array of king positions containing how good it is to have a king at each position
// TODO make the eval relative to the proximity of the king to the other players king (should help solve endgames)
float* compute_king_pos(){
    float *eval_table = (float*)malloc(sizeof(float) * 64);
        float init_table[8][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
        };
    for (int i = 0; i < 64; i++){
        eval_table[i] = init_table[i / 8][i % 8] / 10.0;
    }



    return eval_table;
}

// compute the distance between two bit positions
int distance(int pos1, int pos2){
    return abs(pos1 % 8 - pos2 % 8) + abs(pos1 / 8 - pos2 / 8);

}

// precompute the distance table. max distance is 1 
float* init_distance_table(){
    float* dist_arr = (float*)malloc(sizeof(float) * 64 * 64);
    for (int i = 0; i < 64; i++){
        for (int j = 0; j < 64; j++){
            dist_arr[i * 64 + j] = (float)distance(i, j) / 14.0f;
        }
    }

    return dist_arr;
}



