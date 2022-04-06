#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "Cset.h"

// define some constants
#define intLong long long
#define True 1
#define False 0
#define Null 0

/*
#define PY_SSIZE_T_CLEAN
#include <Python.h>
*/

// define some functions
struct board_data;
struct board_data *board_data_constructor(int player, int move_start, int move_end);
void compute_offsets(int* offsets);
struct set* get_piece_locations(intLong p1, intLong p2, intLong p1k, intLong p2k);
void update_piece_locations(int piece_loc_initial, int piece_loc_after, struct set* piece_loc);
void undo_piece_locations_update(int piece_loc_initial, int piece_loc_after, struct set* piece_loc);
void sort_moves(struct board_data* ptr, int player);
int get_next_board_state(intLong p1, intLong p2, intLong p1k, intLong p2k, int pos_init, int pos_after, int player, int* offsets);
int get_piece_at_location(intLong p1, intLong p2, intLong p1k, intLong p2k, int pos);
int update_board(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int piece_loc_initial, int piece_loc_after);
void undo_board_update(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type);
int generate_all_moves(intLong p1, intLong p2, intLong p1k, intLong p2k, int player, int* moves, struct set* piece_loc, int* offsets);
int generate_moves(intLong p1, intLong p2, intLong p1k, intLong p2k, int pos, int* save_loc, int* offsets, int only_jump);
struct board_data* search_board(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int player,
                                int* piece_loc, int* offsets, int depth,
                                float alpha, float beta, struct board_data* best_moves);
struct board_data* start_board_search(intLong p1, intLong p2, intLong p1k, intLong p2k, int player);
void human_readble_board(intLong p1, intLong p2, intLong p1k, intLong p2k);
long long n_ply_search(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int player, struct set* piece_loc, int* offsets, int depth);


/*
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

*/
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
// constructor for the board_data struct. sets the start/end move, and player variables as these are not going to change.
// the rest of the data is to be added after the search returns to the node this was created from
// at that point the next_boards array will be given the pointer to the malloc'd array of board_data structs
// corisponding to the next possible moves the array should then be sorted according to the score of the move
struct board_data *board_data_constructor(int player, int move_start, int move_end){
    struct board_data *board = malloc(sizeof(struct board_data));
    board->score = 0;
    board->player = player;
    board->move_start = move_start;
    board->move_end = move_end;
    return board;
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

// get the location of all the piece's on the board to avoid looping over unused spots
// takes the board as arguments
// returns a set of all the pieces on the board
// note: assumes a valid board
struct set* get_piece_locations(intLong p1, intLong p2, intLong p1k, intLong p2k){
    struct set* piece_loc = create_set();
    for (int i = 0; i < 64; i++){
        if (get_piece_at_location(p1, p2, p1k, p2k, i) != 0){
            set_add(piece_loc, i);
        }
    }
    return piece_loc;
}

// update the array of piece locations
void update_piece_locations(int piece_loc_initial, int piece_loc_after, struct set* piece_loc){
    // set the index of a potential jumped piece to -1
    int middle_piece_loc = -1;
    // check for a jump
    if (abs(piece_loc_initial - piece_loc_after) > 10){
        // get the location of the piece that was jumped
        middle_piece_loc = (piece_loc_initial + piece_loc_after) / 2;
        // remove the jumped piece from the set
        set_remove(piece_loc, middle_piece_loc);
    }
    // remove the piece that was moved from the set and add its new location to the set
    set_remove(piece_loc, piece_loc_initial);
    set_add(piece_loc, piece_loc_after);
}

// undo a update the the array of piece locations
void undo_piece_locations_update(int piece_loc_initial, int piece_loc_after, struct set* piece_loc){
    // set the index of a potential jumped piece to -1
    int middle_piece_loc = -1;
    // check for a jump
    if (abs(piece_loc_initial - piece_loc_after) > 10){
        // get the location of the piece that was jumped
        middle_piece_loc = (piece_loc_initial + piece_loc_after) / 2;
        // add the jumped piece back in to the set
        set_add(piece_loc, middle_piece_loc);
    }

    // add the piece that moved back to the set and remove its new location from the set (ie. undo the move)
    set_remove(piece_loc, piece_loc_after);
    set_add(piece_loc, piece_loc_initial);
}

// sorts the pointers in a board_data struct array according to the score of the move's
// each pointer points to and the player that is making the move
void sort_moves(struct board_data* ptr, int player){

}

// find the state of the next board after a move
// returns 1 if the moving player is player 2 returns 0 if the player is player 1
// pos is the position the piece will be after the first jump and leading in to the second one
// convenion is that player 1 has internal state of 1 and player 2 has internal state of 2 (ie. 1 is red and 2 is black in a normal match)
int get_next_board_state(intLong p1, intLong p2, intLong p1k, intLong p2k, int pos_init, int pos_after, int player, int* offsets){
    // check if the last move was a jump if not invert the state
    if (!(abs(pos_init - pos_after) > 10)){
        return player ^ 0x3;
    }
    // if the last move was a jump see if it can jump again if so do not change the state
    int move_bucket[8];
    int num_moves = generate_moves(p1, p2, p1k, p2k, pos_after, &move_bucket[0], offsets, True);
    if (num_moves != 0){
        return player;
    } else {
        return player ^ 0x3;
    }
}   

// get the type of piece at the position specified
int get_piece_at_location(intLong p1, intLong p2, intLong p1k, intLong p2k, int pos){
    if (p1 >> pos & 1){
        return 1;
    } else if (p2 >> pos & 1){
        return 2;
    } else if (p1k >> pos & 1){
        return 3;
    } else if (p2k >> pos & 1){
        return 4;
    } else {
        return 0;
    }
}

// update the board with a move and return the type of piece that was captured
// returns -1 if no piece was captured
int update_board(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int piece_loc_initial, int piece_loc_after){
    int piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, piece_loc_initial);
    int return_value = -1;
    if (piece_type == 1){
        *p1 = *p1 ^ (1ll << piece_loc_initial);
        *p1 = *p1 ^ (1ll << piece_loc_after);
    } else if (piece_type == 2){
        *p2 = *p2 ^ (1ll << piece_loc_initial);
        *p2 = *p2 ^ (1ll << piece_loc_after);
    } else if (piece_type == 3){
        *p1k = *p1k ^ (1ll << piece_loc_initial);
        *p1k = *p1k ^ (1ll << piece_loc_after);
    } else if (piece_type == 4){
        *p2k = *p2k ^ (1ll << piece_loc_initial);
        *p2k = *p2k ^ (1ll << piece_loc_after);
    }
    if (abs(piece_loc_initial - piece_loc_after) > 10){
        return_value = get_piece_at_location(*p1, *p2, *p1k, *p2k, (piece_loc_initial + piece_loc_after) / 2);
        if (return_value == 1){
            *p1 = *p1 ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (return_value == 2){
            *p2 = *p2 ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (return_value == 3){
            *p1k = *p1k ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (return_value == 4){
            *p2k = *p2k ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        }
    }

    return return_value;
}

// reverse the a board update
void undo_board_update(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type){
    int piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, piece_loc_after);
    if (piece_type == 1){
        *p1 = *p1 ^ (1ll << piece_loc_initial);
        *p1 = *p1 ^ (1ll << piece_loc_after);
    } else if (piece_type == 2){
        *p2 = *p2 ^ (1ll << piece_loc_initial);
        *p2 = *p2 ^ (1ll << piece_loc_after);
    } else if (piece_type == 3){
        *p1k = *p1k ^ (1ll << piece_loc_initial);
        *p1k = *p1k ^ (1ll << piece_loc_after);
    } else if (piece_type == 4){
        *p2k = *p2k ^ (1ll << piece_loc_initial);
        *p2k = *p2k ^ (1ll << piece_loc_after);
    }
    if (jumped_piece_type != -1){
        if (jumped_piece_type == 1){
            *p1 = *p1 ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (jumped_piece_type == 2){
            *p2 = *p2 ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (jumped_piece_type == 3){
            *p1k = *p1k ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        } else if (jumped_piece_type == 4){
            *p2k = *p2k ^ (1ll << ((piece_loc_initial + piece_loc_after) / 2));
        }
    }

}

// generate all possible moves for a given board - 
// takes the board and a memory location to save to as arguments
// returns the number of moves generated
// note: moves should have room for 96 elements as this is the maximum number of moves possible on a legal board
int generate_all_moves(intLong p1, intLong p2, intLong p1k, intLong p2k, int player, int* moves, struct set* piece_loc, int* offsets){
    // setup variables
    int num_moves = 0;
    int move_from_pos = 0;
    int type;
    int jump = 0;
    int friendly_pieces;
    if (player == 1){
        friendly_pieces = 3;
    } else {
        friendly_pieces = 4;
    }
    // loop over all the locations with a piece and generate the moves for each one
    while (piece_loc != Null){
        // check if it is a valid piece
        type = get_piece_at_location(p1, p2, p1k, p2k, piece_loc->value);
        if (type == player || type == friendly_pieces){
            // generate the moves for the piece
            move_from_pos = generate_moves(p1, p2, p1k, p2k, piece_loc->value, moves + (num_moves * 2), offsets, jump);
            if (move_from_pos == -1){
                num_moves = 0;
                jump = 1;
                move_from_pos = generate_moves(p1, p2, p1k, p2k, piece_loc->value, moves + (num_moves * 2), offsets, jump);
                num_moves += move_from_pos;
            }
            // if no special cases occured add the moves to the counter
            else {
                num_moves += move_from_pos;
            }
        }
        piece_loc = piece_loc->next;
        
    }
    return num_moves;
}

// generate the move's for a single piece
// takes the board, the piece position, memory location to save, and a jump flag as arguments
// returns the number of moves generated or -1 if a jump was found and the jump flag was not set
// note: save_loc should have room for 8 int
int generate_moves(intLong p1, intLong p2, intLong p1k, intLong p2k, int pos, int* save_loc, int* offsets, int only_jump){
    // get the offset index for the position, then initialized some variables
    int offset_index = pos * 4;
    int num_moves = 0;
    int friendly_type = 0;
    int piece_type = get_piece_at_location(p1, p2, p1k, p2k, pos);
    // load offsets into a local variable
    int number_offsets[] = {-9, -7, 7, 9};
    
    // set the friendly piece type
    if (piece_type == 1){
        friendly_type = 3;
    }
    else if (piece_type == 2){
        friendly_type = 4;
    }
    else if (piece_type == 3){
        friendly_type = 1;
    }
    else if (piece_type == 4){
        friendly_type = 2;
    }
    // loop over all the offsets
    int new_pos = pos;
    int temp_pos_type;
    for (int i = 0; i < 4; i++){
        // if it is not a valid offset for that piece continue
        if (offsets[offset_index + i] == 0 || (piece_type == 2 & (i==0 || i == 1)) || (piece_type == 1 & (i==2 || i==3))){
            continue;
        }
        // get the data of the potential jump location
        new_pos = pos + number_offsets[i];
        temp_pos_type = get_piece_at_location(p1, p2, p1k, p2k, new_pos);
        // if the location is empty, add it to the list of moves - also check for only jump flag
        if (temp_pos_type == 0 && !only_jump){
            save_loc[num_moves * 2] = pos;
            save_loc[(num_moves * 2) + 1] = new_pos;
            num_moves++;
        // if the location is a friendly piece, continue
        } else if (temp_pos_type == friendly_type || temp_pos_type == piece_type){
            continue;
        // if it is an enemy piece see if the location one further is empty and in that case it is a capture move!
        } else if (temp_pos_type != 0){
            // make sure the offset is valid at the new location
            if (offsets[(new_pos * 4) + i] == 0){
                continue;
            }
            new_pos = new_pos + number_offsets[i];
            temp_pos_type = get_piece_at_location(p1, p2, p1k, p2k, new_pos);
            if (temp_pos_type == 0){
                // if a jump was found and the jump flag was not set, return -1
                if (!only_jump){
                    return -1;
                }
                save_loc[num_moves * 2] = pos;
                save_loc[(num_moves * 2) + 1] = new_pos;
                num_moves++;
            }
        }
    }
    return num_moves;
}

// search the board for the best move recursivly and return a pointer to the memory
// location that holds the moves for the board ordered in the order best to worst
struct board_data* search_board(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int player,
                                int* piece_loc, int* offsets, int depth,
                                float alpha, float beta, struct board_data* best_moves){
    // if the depth is 0 then we are at the end of the standard search so begin the captures search

}


// prepare needed memory for a search and call the search function to find the best move and return a pointer to the memory 
// location that holds the moves for the board ordered in the order best to worst
struct board_data* start_board_search(intLong p1, intLong p2, intLong p1k, intLong p2k, int player){
    // malloc the data needed for the array (data needed is num spaces * nummovedir * sizeof(int))
    int* piece_offsets = malloc(sizeof(int) * 64 * 4);
    compute_offsets(piece_offsets);
    struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);


}

// search to the depth specified and count to total amount of boards for a certain depth
// this is used to test if the move generation is working as expected and to see how much time is spent
long long n_ply_search(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int player, struct set* piece_loc, int* offsets, int depth){
    // make a int array of size 96 elements to hold the moves
    int moves[96];
    long long total_boards = 0;
    int player_next;
    int num_moves = generate_all_moves(*p1, *p2, *p1k, *p2k, player, &moves[0], piece_loc, offsets);
    // if there are no moves or the depth is 0 return 1
    if (depth == 0){
        total_boards += 1;
        return total_boards;
    }

    // otherwise loop over all the moves and call the function recursivly
    for (int i = 0; i < num_moves; i++){
        // update the board and set values
        int jumped_piece_type = update_board(p1, p2, p1k, p2k, moves[i * 2], moves[(i * 2) + 1]);
        update_piece_locations(moves[i * 2], moves[(i * 2) + 1], piece_loc);

        // get the player that is going to move after this move
        player_next = get_next_board_state(*p1, *p2, *p1k, *p2k, moves[i * 2], moves[(i * 2) + 1], player, offsets);

        // call the function recursivly
        total_boards += n_ply_search(p1, p2, p1k, p2k, player_next, piece_loc, offsets, depth - 1);

        // undo the update to the board and piece locations
        undo_piece_locations_update(moves[i * 2], moves[(i * 2) + 1], piece_loc);
        undo_board_update(p1, p2, p1k, p2k, moves[i * 2], moves[(i * 2) + 1], jumped_piece_type);
    }
    
    return total_boards;

}

// print the board to standard out in a human readable format (for debugging)
void human_readble_board(intLong p1, intLong p2, intLong p1k, intLong p2k){
    // loop over all the pieces and print them out
    for (int i = 0; i < 64; i++){
        int piece_type = get_piece_at_location(p1, p2, p1k, p2k, i);
        if (piece_type == 1){
            printf("1");
        } else if (piece_type == 2){
            printf("2");
        } else if (piece_type == 3){
            printf("3");
        } else if (piece_type == 4){
            printf("4");
        } else {
            printf("-");
        }
        if ((i + 1) % 8 == 0){
            printf("\n");
        }
    }
    printf("\n");
}


// main function
// runs a n_ply search to verify the move generation is working as expected
// also benchmarks the time it takes to generate the moves
int main(){
    // setup initial board and search structures
    intLong p1 = 6172839697753047040;
    intLong p2 = 11163050;
    intLong p1k = 0;
    intLong p2k = 0;
    int player = 1;
    struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);
    int* offsets = malloc(sizeof(int) * 64 * 4);
    compute_offsets(offsets);

    int depth = 10;

    for (int i = 1; i < depth; i++){
        long long n_ply_search_result = n_ply_search(&p1, &p2, &p1k, &p2k, player, piece_loc, offsets, i);
        printf("%d ply search result: %lld\n", i, n_ply_search_result);

    }

    return 0;
}