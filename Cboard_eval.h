// Author: Collin Kees
// board evaluation and hashing functions

#include <stdlib.h>
#include "Cset.h"
#include "Chash_table.h"
#include "Cneural_net.h"
#include "Ckiller_table.h"


// define some functions
float* compute_piece_pos();
float* compute_king_pos();
struct board_evaler;
float calculate_eval(long long p1, long long p2, long long p1k, long long p2k, struct set* piece_loc, struct board_evaler* evaler);
float evaluate_pos(int type, int pos, struct board_evaler* evaler);


// struct to hold the data for the hash table and other data related to getting a evaluation for a board
struct board_evaler{
    float *piece_pos_map;
    float *king_pos_map;
    long long int nodes;
    int search_depth;
    struct neural_net *NN_evaler;
    struct hash_table* hash_table;
    struct killer_table* killer_table;
};

struct board_evaler* board_evaler_constructor(int search_depth){
    struct board_evaler* evaler = (struct board_evaler*)malloc(sizeof(struct board_evaler));
    evaler->piece_pos_map = compute_piece_pos();
    evaler->king_pos_map = compute_king_pos();
    evaler->nodes = 0ll;
    evaler->NN_evaler = load_network_from_file("neural_net/neural_net");
    // prepare a table of size 8,388,608 
    long long int hash_table_size = 1 << 23;
    evaler->hash_table = init_hash_table(hash_table_size);
    evaler->killer_table = init_killer_table(search_depth);

    return evaler;
}

// get the evaluation for a board given the board state
float get_eval(long long p1, long long p2, long long p1k, long long p2k, struct set* piece_loc, struct board_evaler* evaler){
    // there was no entry found so lets calculate it
    float eval = calculate_eval(p1, p2, p1k, p2k, piece_loc, evaler);

    // test neural net (subtract 5 from the eval to 0 being a draw)
    //eval = (float)get_output(evaler->NN_evaler, p1, p2, p1k, p2k) - 5.0; // neural_net

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
            eval += 3.0;
            eval += evaluate_pos(0, piece_loc_array[i], evaler);
            p1num++;

        }
        else if (p2 >> piece_loc_array[i] & 1){
            eval -= 3.0;
            eval -= evaluate_pos(0, piece_loc_array[i], evaler);
            p2num++;
            
        }
        else if (p1k >> piece_loc_array[i] & 1){
            eval += 5.0;
            eval += evaluate_pos(1, piece_loc_array[i], evaler);
            p1num++;

        }
        else if (p2k >> piece_loc_array[i] & 1){
            eval -= 5.0;
            eval -= evaluate_pos(1, piece_loc_array[i], evaler);
            p2num++;
        }
    }
    // give the player with the most pieces a bonus
    if (p1num > p2num){
        eval += 4.0 / (p1num + p2num);
    }
    else if (p2num > p1num){
        eval -= 4.0 / (p1num + p2num);
    }

    return eval;
}

// evaluate the position of a piece (type comes in two flavors 0 for normal piece and 1 for a king)
float evaluate_pos(int type, int pos, struct board_evaler* evaler){
    if (type == 0){
        return evaler->piece_pos_map[pos];
    }
    else{
        return evaler->king_pos_map[pos];
    }
}

// compute the array of piece positions containing how good it is to have a piece at each position
float* compute_piece_pos(){
    float *eval_table = (float*)malloc(sizeof(float) * 64);
    float table[8][8] = { 
        {2, 2, 2, 2, 2, 2, 2, 2},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0 ,0, 0},
        {0, 0, 0, 0, 0, 0 ,0, 0},
        {0, 0, 0, 0, 0, 0 ,0, 0},
        {0, 0, 0, 0, 0, 0 ,0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {2, 2, 2, 2, 2, 2, 2, 2}
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
         {0, 0, 3, 3, 3, 3, 0, 0},
         {0, 0, 3, 3, 3, 3, 0, 0},
         {0, 0, 3, 3, 3, 3, 0, 0},
         {0, 0, 3, 3, 3, 3, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0}
        };
    for (int i = 0; i < 64; i++){
        eval_table[i] = init_table[i / 8][i % 8] / 10.0;
    }



    return eval_table;
}


