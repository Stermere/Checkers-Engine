#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "Cset.h"

// make a python extension function that takes a touple as argument
// and returns a list of integers that represent the best moves for the board
static PyObject* board_search(PyObject *self, PyObject *args){
    int x, y, z, w, v;

    if (!PyArg_ParseTuple(args, "iiiii", &x, &y, &z, &w, &v))
        return NULL;

    // sum the arguments
    int sum = x + y + z + w + v;
    // return the sum
    return Py_BuildValue("i", sum);
}

// tell the pyhton interpreter about the functions we want to use
static PyMethodDef c_board_search_methods[] = {
    {"board_search", board_search, METH_VARARGS, "python interface for a checkers engine board search"},
    {NULL, NULL, 0, NULL}
};

// define the module
static struct PyModuleDef c_board_search = {
    PyModuleDef_HEAD_INIT,
    "board_search",
    "C board search logic",
    -1,
    c_board_search_methods
};

PyMODINIT_FUNC
PyInit_board_search(void){
    return PyModule_Create(&c_board_search);
}

// End of setup code

// Start of main code

// structure that holds the state of the current board and the move that is made to get from the last board
// to the current board. The other piece of data is an array of pointers to the next board_data structures
// that represent the possible next moves.
// the point of this struct is to make a tree of moves that have been traversed and pass it to the next deaper search
// so that more prunning occurs and the search is faster
struct board_data {    
    float score;
    int player;
    int move_start;
    int move_end;
    int num_moves;
    struct board_data *next_boards;
};
// constructor for the board_data struct. sets the start/end move, player and num_moves variables as these are not going to change.
// the rest of the data is to be added after the search returns to the node this was created from
// at that point the next_boards array will be given the pointer to the malloc'd array of board_data structs
// corisponding to the next possible moves the array should then be sorted according to the score of the move

// TODO - check this function copilot made it and in the moment it looked like it was working fine
struct board_data *board_data_constructor(int player, int move_start, int move_end, int num_moves){
    struct board_data *board = malloc(sizeof(struct board_data));
    board->score = 0;
    board->player = player;
    board->move_start = move_start;
    board->move_end = move_end;
    board->num_moves = num_moves;
    board->next_boards = malloc(sizeof(struct board_data) * num_moves);
    return board;
}

// sorts the pointers in a board_data struct array according to the score of the move
// and the player that is making the move
void sort_moves(struct board_data* ptr, int player){
    
}

// generate all possible moves for a given board - 
// takes the board and a memory location to save to as arguments
// returns the number of moves generated
// returns -1 if there are no moves
int generate_all_moves(int p1, int p2, int p1k, int p2k, int* moves, int* piece_loc){
    return 0;
}

// generate the move's for a single piece
// takes the board, the piece, and a memory location to save to as arguments
// move_count is the current moves for the board
// returns the aditional moves generated
int generate_moves(int p1, int p2, int p1k, int p2k, int piece_loc, int* move_loc, int move_count){

}


// get the location of all the piece's on the board to avoid looping over unused spots
// takes the board as arguments
// returns a set of all the pieces on the board
// note: assumes a valid board
struct set* get_piece_locations(int p1, int p2, int p1k, int p2k){
    struct set* piece_loc = set_constructor();
    for (int i = 0; i < 64; i++){
        if (get_piece_at_location(p1, p2, p1k, p2k, i) != 0){
            set_add(piece_loc, i);
        }
    }
    return piece_loc;
}

// update the array of piece locations
void update_piece_locations(int piece_loc_initial, int piece_loc_after, int* piece_loc){

}

// undo a update the the array of piece locations
void undo_piece_locations_update(int piece_loc_initial, int piece_loc_after, int* piece_loc){

}

// find the state of the next board after a move
// returns 1 if the moving player is player 2 returns 0 if the player is player 1
// 
int get_next_board_state(int p1, int p2, int p1k, int p2k, int piece_loc_initial, int piece_loc_after){

}

// get the type of piece at the position specified
int get_piece_at_location(int p1, int p2, int p1k, int p2k, int piece_loc){
    
}

// compute the valid move directions for every location on the board
// saves the results the the pointer passed to the function in the form of 4 integers either 0 or 1
// 1 means it is a valid direction to move in
// the first integer is the direction of the move to the left and up
// the second integer is the direction of the move to the right and up
// the third integer is the direction of the move to the left and down
// the fourth integer is the direction of the move to the right and down
void compute_offsets(int* offsets){
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
}

// update the board with a move and return the type of piece that was captured
// returns -1 if no piece was captured
int update_board(int* p1, int* p2, int* p1k, int* p2k, int piece_loc_initial, int piece_loc_after){

}

// reverse the a board update
void undo_board_update(int* p1, int* p2, int* p1k, int* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type){

}

// search the board for the best move recursivly and return a pointer to the memory
// location that holds the moves for the board ordered in the order best to worst
struct board_data* search_board(int p1, int p2, int p1k, int p2k, int player,
                                int* piece_loc, int* offsets, int depth,
                                float alpha, float beta, struct board_data* best_moves){
    // if the depth is 0 then we are at the end of the standard search so begin the captures search

}


// search to the depth specified and count to total amount of boards for a certain depth
// this is used to test if the move generation is working as expected and to see how much time is spent
long long n_ply_search(int p1, int p2, int p1k, int p2k, int player, int* piece_loc, int* offsets, int depth){

}


// prepare needed memory for a search and call the search function to find the best move and return a pointer to the memory 
// location that holds the moves for the board ordered in the order best to worst
struct board_data* search_board_for_best_move(int p1, int p2, int p1k, int p2k, int player){
    // malloc the data needed for the array (data needed is num spaces * nummovedir * sizeof(int))
    int* piece_offsets = malloc(sizeof(int) * 64 * 4);
    compute_offsets(piece_offsets);
    

}








