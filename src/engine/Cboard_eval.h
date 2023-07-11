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
struct board_evaler;
float calculate_eval(long long p1, long long p2, long long p1k, long long p2k, struct set* piece_loc, struct board_evaler* evaler);
float evaluate_pos(int type, int pos, struct board_evaler* evaler);


// struct to hold the data for the hash table and other data related to getting a evaluation for a board
struct board_evaler{
    float *piece_pos_map_p1;
    float *piece_pos_map_p2;
    float *king_pos_map;
    int search_depth;
    int max_depth;
    float alpha;
    float beta;
    struct neural_net *NN_evaler;
    struct hash_table* hash_table;
    struct killer_table* killer_table;
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
    // prepare a table of size 2097152
    long long int hash_table_size = 1 << 21;
    evaler->hash_table = init_hash_table(hash_table_size);
    evaler->killer_table = init_killer_table(search_depth);
    evaler->start_time = start_time;
    evaler->time_limit = time_limit;
    evaler->extended_depth = 0;
    return evaler;
}

// get the evaluation for a board given the board state
float get_eval(long long p1, long long p2, long long p1k, long long p2k, struct set* piece_loc, struct board_evaler* evaler){
    // there was no entry found so lets calculate it
    float eval = calculate_eval(p1, p2, p1k, p2k, piece_loc, evaler);

    // test neural net
    //float eval = (float)get_output(evaler->NN_evaler, p1, p2, p1k, p2k) - 100.0; // neural_net
    return eval;
}


// calculate the board evaluation
float calculate_eval(long long p1, long long p2, long long p1k, long long p2k, struct set* piece_loc, struct board_evaler* evaler){
    float eval = 0;
    int piece_loc_array[64];
    int num_pieces = populate_array(piece_loc, piece_loc_array);
    int p1num = 0;
    int p2num = 0;
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
            p1num++;

        }
        else if (p2k >> piece_loc_array[i] & 1){
            eval -= 5.0f;
            eval -= evaluate_pos(4, piece_loc_array[i], evaler);
            p2num++;
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
        eval += (15.0f * (p1num - p2num)) / (p1num + p2num);
        if (p2num < 3)
            eval += 2.0f;
        if (p2num < 2)
            eval += 5.0f;
    }
    else if (p2num > p1num){
        eval -= (15.0f * (p2num - p1num)) / (p1num + p2num);
        if (p1num < 3)
            eval -= 2.0f;
        else if (p1num < 2)
            eval -= 5.0f;
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

// compute the array of piece positions containing how good it is to have a piece at each position
float* compute_piece_pos_p1(){
    float *eval_table = (float*)malloc(sizeof(float) * 64);
    float table[8][8] = { 
        {0, 1, 5, 5, 5, 5, 1, 0},
        {0, 1, 3, 3, 3, 3, 1, 0},
        {0, 0, 3, 3, 2, 2, 0, 0},
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
        {0, 0, 2, 2, 3, 3, 0, 0},
        {0, 1, 3, 3, 3, 3, 1, 0},
        {0, 1, 5, 5, 5, 5, 1, 0}
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
        {0, 1, 0, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 2, 2, 2, 2, 0, 0}, 
        {0, 0, 2, 2, 2, 2, 0, 0}, 
        {0, 0, 2, 2, 2, 2, 0, 0}, 
        {0, 0, 2, 2, 2, 2, 0, 0}, 
        {0, 0, 2, 2, 2, 2, 0, 1}, 
        {0, 0, 0, 0, 0, 0, 1, 0}
        };
    for (int i = 0; i < 64; i++){
        eval_table[i] = init_table[i / 8][i % 8] / 10.0;
    }



    return eval_table;
}

