// Author: Collin Kees
// language: c
// Description: this program serves as the C library for move generation and evaluation of checkers boards

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
// including the Cboard_eval library also includes the set and transposition tables for the engine
#include "Cboard_eval.h"

// define some constants
#define intLong long long int
#define True 1
#define False 0
#define Null 0
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))


#define PY_SSIZE_T_CLEAN
#include <Python.h>


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
void undo_board_update(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type, int initial_piece_type);
int generate_all_moves(intLong p1, intLong p2, intLong p1k, intLong p2k, int player, int* moves, struct set* piece_loc, int* offsets, int jump);
int generate_moves(intLong p1, intLong p2, intLong p1k, intLong p2k, int pos, int* save_loc, int* offsets, int only_jump);
float search_board(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int player,
    struct set* piece_loc, int* offsets, int depth, float alpha, float beta, int captures_only, struct board_data* best_moves,
    struct board_evaler* evaler, unsigned long long int hash);
struct board_info* start_board_search(intLong p1, intLong p2, intLong p1k, intLong p2k, int player, float search_time, int search_depth);
void human_readble_board(intLong p1, intLong p2, intLong p1k, intLong p2k);
long long n_ply_search(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int player, struct set* piece_loc, int* offsets, int depth);
void merge(struct board_data* ptr, int half, int num_elements);
void merge_sort(struct board_data* ptr, int num_elements);
unsigned long long int update_hash(intLong p1, intLong p2, intLong p1k, intLong p2k, int pos_init, int pos_after, unsigned long long int hash, struct board_evaler* evaler);
struct board_data* get_best_move(struct board_data *head, int player);

// for some reason this has to be above the python stuff even if it is defined, otherwise it doesn't work

// structure that holds the state of the current board and the move that is made to get from the last board
// to the current board. The other piece of data is an array of pointers to the next board_data structures
// that represent the possible next moves.
// the point of this struct is to make a tree of moves that have been traversed and pass it to the next deaper search
// so that more prunning occurs and the search is faster
struct board_data {    
    float eval;
    unsigned long long int hash;
    int player;
    int move_start;
    int move_end;
    int num_moves;
    int prunned_loc;
    struct board_data *next_boards;
};

// holds the head to the board_tree and the evaler struct
struct search_info {
    struct board_data* head;
    struct board_evaler* evaler;
};

// constructor for the board_data struct. sets the start/end move, and player variables as these are not going to change.
// the rest of the data is to be added after the search returns to the node this was created from
// at that point the next_boards array will be given the pointer to the malloc'd array of board_data structs
// corisponding to the next possible moves the array should then be sorted according to the score of the move
struct board_data *board_data_constructor(int player, int move_start, int move_end){
    struct board_data *board = malloc(sizeof(struct board_data));
    board->eval = 0;
    board->hash = 0;
    board->player = player;
    board->move_start = move_start;
    board->move_end = move_end;
    board->num_moves = -1;
    return board;
}

// python comunication code 

// make a python extension function that takes a touple as argument
// and returns a list of integers that represent the best moves for the board
static PyObject* search_position(PyObject *self, PyObject *args){
    unsigned long long int p1, p2, p1k, p2k, player, search_depth;
    float search_time;

    // get the arguments from python
    if (!PyArg_ParseTuple(args, "KKKKKfK", &p1, &p2, &p1k, &p2k, &player, &search_time, &search_depth))
        return NULL;

    // call the search function
    struct search_info* board_info = start_board_search(p1, p2, p1k, p2k, player, search_time, search_depth);

    // package the relevant data into a python tuple
    PyObject* py_list = PyList_New(0);
    PyObject* py_tuple;
    for (int i = 0; i < board_info->head->num_moves; i++){
        py_tuple = Py_BuildValue("fii", board_info->head->next_boards[i].eval, board_info->head->next_boards[i].move_start, board_info->head->next_boards[i].move_end);
        PyList_Append(py_list, py_tuple);
    }
    // add some info about the search to the end of the list
    // get the recomenend best move (for usecases without post processing)
    struct board_data* best_move = get_best_move(board_info->head, player);
    py_tuple = Py_BuildValue("ii", best_move->move_start, best_move->move_end);
    PyList_Append(py_list, py_tuple);

    // get some stats about the search
    py_tuple = Py_BuildValue("KKKf", board_info->evaler->search_depth, board_info->evaler->boards_evaluated, board_info->evaler->hash_table->num_entries, best_move->eval);
    PyList_Append(py_list, py_tuple);


    // return the python tuple
    return py_list;
}

// tell the pyhton interpreter about the functions we want to use
static PyMethodDef c_board_search_methods[] = {
    {"search_position", search_position, METH_VARARGS, 
    "takes the 6 64bit integers for the board\
     the first 4 come from convert to bit board, the remaining is the player\
     and the time and depth to search for"},
    {NULL, NULL, 0, NULL}
};

// define the module
static struct PyModuleDef search_engine = {
    PyModuleDef_HEAD_INIT,
    "search_engine",
    "C board search logic",
    -1,
    c_board_search_methods
};

PyMODINIT_FUNC
PyInit_search_engine(void){
    return PyModule_Create(&search_engine);
}


// End of python comunication code

// Start of main code search code

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

// gets the best move from a searched tree (returns the node that has the best move stored in it)
struct board_data* get_best_move(struct board_data *head, int player){
    // get the best move from the tree
    struct board_data *best_move = head->next_boards;
    int order = 1;
    if (player == 2){
        order = -1;
    }
    for (int i = 0; i < head->num_moves; i++){

        if ((head->next_boards[i].eval * order) >= (best_move->eval * order)){
            best_move = head->next_boards + i;
        }
    }
    return best_move;
}


// get the location of all the piece's on the board to avoid looping over unused spots
// takes the board as arguments
// returns a set of all the pieces on the board
// note: assumes a valid board
struct set* get_piece_locations(intLong p1, intLong p2, intLong p1k, intLong p2k){
    struct set* piece_loc = create_set();
    piece_loc->value = -1;
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

// takes a pointer to the board_data struct that we want to sort the moves for 
// sorts the array according to the score of the move
// the pointer to the array that is passed is still the pointer to the array of board_data structs
// uses the quicksort algorithm
void sort_moves(struct board_data* ptr, int player){
    struct board_data temp;
    int made_swap = 1;
    struct board_data* board_data_list = ptr->next_boards;
    // if the player is player 2 we want the moves from least to greatest
    // if the player is player 1 we want the moves from greatest to least
    // to do this we can multiply the evals by -1 sort them and then multiply the evals back by -1
    if (player == 2){
        for (int i = 0; i < ptr->num_moves; i++){
            board_data_list[i].eval = board_data_list[i].eval * -1;
        }
    }
    // use the quicksort algorithm to sort the moves
    merge_sort(board_data_list, ptr->num_moves);

    // undo the multiplication by the player variable
    if (player == 2){
        for (int i = 0; i < ptr->num_moves; i++){
            board_data_list[i].eval = board_data_list[i].eval * -1;
        }
    }
}

// the quicksort recursive funtion
// takes the pointer to the board_data array and the number of elements in the array
void merge_sort(struct board_data* ptr, int num_elements){
    // base case
    if (num_elements <= 1){
        return;
    }
    // recursive case
    // split the array into two halves
    int half = num_elements / 2;
    // sort the two halves
    merge_sort(ptr, half);
    merge_sort(ptr + half, num_elements - half);
    // merge the two sorted halves
    merge(ptr, half, num_elements - half);

}

// the merge part of the quicksort algorithm
void merge(struct board_data* ptr, int half, int num_elements){
    // create two arrays to hold the two halves of the array
    struct board_data* left = malloc(sizeof(struct board_data) * half);
    struct board_data* right = malloc(sizeof(struct board_data) * num_elements - half);
    // copy the left half of the array into the left array
    for (int i = 0; i < half; i++){
        left[i] = ptr[i];
    }
    // copy the right half of the array into the right array
    for (int i = 0; i < num_elements - half; i++){
        right[i] = ptr[half + i];
    }
    // merge the two sorted halves
    int left_index = 0;
    int right_index = 0;
    int index = 0;
    while (left_index < half && right_index < num_elements - half){
        if (left[left_index].eval > right[right_index].eval){
            ptr[index] = left[left_index];
            left_index++;
        }
        else{
            ptr[index] = right[right_index];
            right_index++;
        }
        index++;
    }
    // copy the rest of the left array into the array
    while (left_index < half){
        ptr[index] = left[left_index];
        left_index++;
        index++;
    }
    // copy the rest of the right array into the array
    while (right_index < num_elements - half){
        ptr[index] = right[right_index];
        right_index++;
        index++;
    }
    // free the two arrays
    free(left);
    free(right);
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
        // check if the piece should become a king
        if (piece_loc_after < 8){
            *p1k = *p1k ^ (1ll << piece_loc_after);
            *p1 = *p1 ^ (1ll << piece_loc_after);
        }
    } else if (piece_type == 2){
        *p2 = *p2 ^ (1ll << piece_loc_initial);
        *p2 = *p2 ^ (1ll << piece_loc_after);
        // check if the piece should become a king
        if (piece_loc_after > 55){
            *p2k = *p2k ^ (1ll << piece_loc_after);
            *p2 = *p2 ^ (1ll << piece_loc_after);
        }
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
void undo_board_update(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type, int initial_piece_type){
    int piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, piece_loc_after);
    if (piece_type == 1){
        *p1 = *p1 ^ (1ll << piece_loc_initial);
        *p1 = *p1 ^ (1ll << piece_loc_after);
        // check if the piece should be unkinged
        if (initial_piece_type == 1){
            *p1k = *p1k ^ (1ll << piece_loc_initial);
            *p1 = *p1 ^ (1ll << piece_loc_initial);
        }
    } else if (piece_type == 2){
        *p2 = *p2 ^ (1ll << piece_loc_initial);
        *p2 = *p2 ^ (1ll << piece_loc_after);
        // check if the piece should be unkinged
        if (initial_piece_type == 2){
            *p2k = *p2k ^ (1ll << piece_loc_initial);
            *p2 = *p2 ^ (1ll << piece_loc_initial);
        }
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

// updates the hash with the moves that will be made to get to the next move
unsigned long long int update_hash(intLong p1, intLong p2, intLong p1k, intLong p2k, int pos_init, int pos_after, unsigned long long int hash, struct board_evaler* evaler){
    int piece_type = get_piece_at_location(p1, p2, p1k, p2k, pos_init);
    int jumped_piece_type = -1;
    if (abs(pos_init - pos_after) > 10){
        jumped_piece_type = get_piece_at_location(p1, p2, p1k, p2k, (pos_init + pos_after) / 2);
        if(jumped_piece_type == 1){
            hash ^= evaler->hash_table->piece_hash_diff[((pos_init + pos_after)/2)];
        } else if (jumped_piece_type == 2){
            hash ^= evaler->hash_table->piece_hash_diff[((pos_init + pos_after)/2) + 64];
        } else if (jumped_piece_type == 3){
            hash ^= evaler->hash_table->piece_hash_diff[((pos_init + pos_after)/2) + 128];
        } else if (jumped_piece_type == 4){
            hash ^= evaler->hash_table->piece_hash_diff[((pos_init + pos_after)/2) + 192];
        }
    }
    if (piece_type == 1){
        hash ^= evaler->hash_table->piece_hash_diff[pos_init];
        hash ^= evaler->hash_table->piece_hash_diff[pos_after];
    } else if (piece_type == 2){
        hash ^= evaler->hash_table->piece_hash_diff[pos_init + 64];
        hash ^= evaler->hash_table->piece_hash_diff[pos_after + 64];
    } else if (piece_type == 3){
        hash ^= evaler->hash_table->piece_hash_diff[pos_init + 128];
        hash ^= evaler->hash_table->piece_hash_diff[pos_after + 128];
    } else if (piece_type == 4){
        hash ^= evaler->hash_table->piece_hash_diff[pos_init + 192];
        hash ^= evaler->hash_table->piece_hash_diff[pos_after + 192];
    }
    return hash;
}

// generate all possible moves for a given board - 
// takes the board and a memory location to save to as arguments
// returns the number of moves generated
// note: moves should have room for 96 elements as this is the maximum number of moves possible on a legal board
int generate_all_moves(intLong p1, intLong p2, intLong p1k, intLong p2k, int player, int* moves, struct set* piece_loc, int* offsets, int jump){
    // setup variables
    int num_moves = 0;
    int move_from_pos = 0;
    int type;
    int friendly_pieces;
    if (player == 1){
        friendly_pieces = 3;
    } else {
        friendly_pieces = 4;
    }
    int piece_loc_array[64];
    int num_pieces = populate_array(piece_loc, piece_loc_array);

    // loop over all the locations with a piece and generate the moves for each one
    for (int i = 0; i < num_pieces; i++){
        // check if it is a valid piece
        type = get_piece_at_location(p1, p2, p1k, p2k, piece_loc_array[i]);
        if (type == player || type == friendly_pieces){
            // generate the moves for the piece
            move_from_pos = generate_moves(p1, p2, p1k, p2k, piece_loc_array[i], moves + (num_moves * 2), offsets, jump);
            if (move_from_pos == -1){
                num_moves = 0;
                jump = 1;
                move_from_pos = generate_moves(p1, p2, p1k, p2k, piece_loc_array[i], moves + (num_moves * 2), offsets, jump);
                num_moves += move_from_pos;
            }
            // if no special cases occured add the moves to the counter
            else {
                num_moves += move_from_pos;
            }
        }    
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
        if (offsets[offset_index + i] == 0 || ((piece_type == 2) & (i==0 || i == 1)) || ((piece_type == 1) & (i==2 || i==3))){
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

// search the board for the best move recursivly and return the best eval that can be achived from that board position
// the best_moves struct will be populated to the search depth at the end of this search
float search_board(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int player,
    struct set* piece_loc, int* offsets, int depth, float alpha, float beta, int captures_only, struct board_data* best_moves,
    struct board_evaler* evaler, unsigned long long int hash){
    // setup variables
    int player_next;
    float board_eval = 0.0;
    int num_moves;
    int initial_piece_type;
    float min_eval = 1000.0;
    float max_eval = -1000.0;
    unsigned long long int next_hash;
    struct board_data* temp_board;
    // if the depth is 0 then we are at the end of the standard search so begin the captures search
    if (depth <= 0){
        captures_only = True;
    }
    // get the moves for this board and player combo or use the moves generated from the last depth search
    if (best_moves->num_moves == -1){
        int moves[96];
        num_moves = generate_all_moves(*p1, *p2, *p1k, *p2k, player, &moves[0], piece_loc, offsets, captures_only);
        // put all the moves into the best moves struct and fill this layer of the tree to the extent that we can
        if (!captures_only){
            best_moves->num_moves = num_moves;
        }
        // set the boards data
        best_moves->player = player;

        if (num_moves > 0){
            best_moves->next_boards = malloc(sizeof(struct board_data) * num_moves);  
        }
        // fill the next boards with the moves that got them there
        for (int i = 0; i < num_moves; i++){
            temp_board = best_moves->next_boards + i;
            temp_board->move_start = moves[i * 2];
            temp_board->move_end = moves[(i * 2) + 1];
            temp_board->num_moves = -1;
            temp_board->player = -1;
            temp_board->hash = 0ull;
        }
    }
    // if moves where already in the move tree then use them
    else{
        num_moves = best_moves->num_moves;
        // sort moves here (optimizatizes alpha beta prunning if better move's are looked at first)
        // only sort the moves that where evaluated as moves after a prune should be considered low eval for the player
        sort_moves(best_moves, best_moves->prunned_loc);
    }
    // if there are no move then a player must have won, or there are no captures so end this branch (only count as a win if captures_only is false)
    // this is not a perfect check but it should be good enough (ie a player has a piece with no legal moves and we are looking at the captures search)
    // this one case should be fairly rare so it is not worth the extra time that would be taken to check for it (ie the search will go search deaper and thus find the move anyways)
    if (num_moves == 0){
        // if there are no moves and captures only is false then a win has occured eval who won and return
        if (!captures_only){
            if (player == 1){
                board_eval = -100.0 - depth;
            }
            else{
                board_eval = 100.0 + depth;
            }
            evaler->boards_evaluated += 1;
            best_moves->eval = board_eval;
            
            return board_eval;
        }
        // if there are no moves and captures only is true then we found the end of a catures only search evaluate the position and return
        else{
            board_eval = get_eval(*p1, *p2, *p1k, *p2k, piece_loc, evaler, depth, hash);
            best_moves->eval = board_eval;
            evaler->boards_evaluated += 1;
            return board_eval;
        }
    }
    temp_board = best_moves->next_boards;
    // if there are moves then we need to search them all
    for (int i = 0; i < num_moves; i++){
        // get the board data for the next move
        temp_board = best_moves->next_boards + i;

        // get the hash for this next board (always do this before the move is made on the board)
        next_hash = update_hash(*p1, *p2, *p1k, *p2k, temp_board->move_start, temp_board->move_end, hash, evaler);
        temp_board->hash = next_hash;

        // update the board and set values
        initial_piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, temp_board->move_start);
        int jumped_piece_type = update_board(p1, p2, p1k, p2k, temp_board->move_start, temp_board->move_end);
        update_piece_locations(temp_board->move_start, temp_board->move_end, piece_loc);

        // get the player that is going to move after this move
        if (temp_board->player != -1){
            player_next = temp_board->player;
        }
        else{
            player_next = get_next_board_state(*p1, *p2, *p1k, *p2k, temp_board->move_start, temp_board->move_end, player, offsets);
            temp_board->player = player_next;
        }

        // call the function recursivly 
        temp_board->eval = search_board(p1, p2, p1k, p2k, player_next, piece_loc, offsets, depth - 1, alpha, beta, captures_only, temp_board, evaler, next_hash);
        
        // undo the update to the board and piece locations
        undo_piece_locations_update(temp_board->move_start, temp_board->move_end, piece_loc);
        undo_board_update(p1, p2, p1k, p2k, temp_board->move_start, temp_board->move_end, jumped_piece_type, initial_piece_type);

        // alpha beta prunning
        if (player == 1){
            max_eval = max(max_eval, temp_board->eval);
            alpha = max(alpha, max_eval);
            if (alpha > beta){
                best_moves->prunned_loc = i + 1;
                break;
            }
        }
        else{
            min_eval = min(min_eval, temp_board->eval);
            beta = min(beta, min_eval);
            if (alpha > beta){
                best_moves->prunned_loc = i + 1;
                break;
            }
        }
    }
    // once all the moves have been searched set the eval of this board to the best move found and return the best eval
    if (player == 1){
        board_eval = max_eval;
    }
    else {
        board_eval = min_eval;
    }
    // update the eval of this board
    best_moves->eval = board_eval;
    
    return board_eval;
}


// prepare needed memory for a search and call the search function to find the best move and return a pointer to the memory 
// location that holds the moves for the board ordered in the order best to worst
struct board_info* start_board_search(intLong p1, intLong p2, intLong p1k, intLong p2k, int player, float search_time, int search_depth){
    struct search_info* return_struct = malloc(sizeof(struct search_info));

    // malloc the data needed for the array (data needed is num spaces * nummovedir * sizeof(int))
    int* piece_offsets = malloc(sizeof(int) * 64 * 4);
    compute_offsets(piece_offsets);
    struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);
    // the tree will be stored in a linked list of best_move structs (populated during search)
    struct board_data* best_moves = board_data_constructor(player, -1, -1);
    struct board_evaler* evaler = board_evaler_constructor(); 
    // get the starting hash
    intLong hash = get_hash(p1, p2, p1k, p2k, evaler->hash_table);
    printf("hash initial: %lld\n", hash);
    float eval_;

    clock_t start, end;
    double cpu_time_used;
    int terminate = 0;
    start = clock();
    // call the search function
    for (int i = 2; i < search_depth; i++){
        if (terminate == 1){
            break;
        }
        // update the evalers search depth
        evaler->search_depth = i;
        eval_ = search_board(&p1, &p2, &p1k, &p2k, player, piece_loc, piece_offsets, i, -1000, 1000, 0, best_moves, evaler, hash);
        
        // get the end time
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        if (cpu_time_used > search_time){
            terminate = 1;
        }
        else if (eval_ > 100.0 || eval_ < -100.0){
            terminate = 1;
        }
        else if (best_moves->num_moves == 1){
            terminate = 1;
        }   
        // print the results
        printf("%d ply search time: %lf\n", i, cpu_time_used);
        printf("best_eval: %f\n\n", best_moves->eval);
    }

    // print some final info about the search
    printf("boards searched: %lld\n", evaler->boards_evaluated);
    printf("hashes stored: %lld\n\n", evaler->hash_table->num_entries);

    // free the memory that is no longer needed
    free(piece_offsets);
    free(piece_loc);
    free(evaler->hash_table->table);
    free(evaler->hash_table);

    return_struct->head = best_moves;
    return_struct->evaler = evaler;
    
    // return the best moves and evaler
    return return_struct;
}

// free the board tree from memory
int free_board_data(struct board_data* data){
    return 0;
}


// clean up after the search has been concluded and the thread is about to exit
void end_board_search(struct board_info* best_moves, struct board_evaler* evaler){
    // free the memory used by the evaler
    free(evaler->hash_table);
    free(evaler);
    // free the memory used by the best moves
    free_board_data(best_moves);
}

// search to the depth specified and count to total amount of boards for a certain depth
// this is used to test if the move generation is working as expected and to see how much time is spent
long long n_ply_search(intLong* p1, intLong* p2, intLong* p1k, intLong* p2k, int player, struct set* piece_loc, int* offsets, int depth){
    // make a int array of size 96 elements to hold the moves
    int moves[96];
    long long total_boards = 0;
    int player_next;
    int pos_init;
    int pos_final;

     // if there are no moves or the depth is 0 return 1
    if (depth == 0){
        total_boards += 1;
        return total_boards;
    }

    // generate the moves for the current board
    int num_moves = generate_all_moves(*p1, *p2, *p1k, *p2k, player, &moves[0], piece_loc, offsets, False);

    // otherwise loop over all the moves and call the function recursivly
    for (int i = 0; i < num_moves; i++){
        // get the moves being made this iteration of the search
        pos_init = moves[i * 2];
        pos_final = moves[(i * 2) + 1];

        // update the board and set values
        int intitial_piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, pos_init);
        int jumped_piece_type = update_board(p1, p2, p1k, p2k, pos_init, pos_final);
        update_piece_locations(pos_init, pos_final, piece_loc);

        // get the player that is going to move after this move
        player_next = get_next_board_state(*p1, *p2, *p1k, *p2k, pos_init, pos_final, player, offsets);

        // call the function recursivly
        total_boards += n_ply_search(p1, p2, p1k, p2k, player_next, piece_loc, offsets, depth - 1);

        // undo the update to the board and piece locations
        undo_piece_locations_update(pos_init, pos_final, piece_loc);
        undo_board_update(p1, p2, p1k, p2k, pos_init, pos_final, jumped_piece_type, intitial_piece_type);
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

    // remove return to test the move generation
    return 0;

    // setup initial board and search structures
    // starting bit values
    intLong p1 = 6172839697753047040;
    intLong p2 = 11163050;
    intLong p1k = 0;
    intLong p2k = 0;

    // end game player 2 winning endgame
    //intLong p1 = 5838922414443986944;
    //intLong p2 = 8409224;
    //intLong p1k = 0;
    //intLong p2k = 288230376154333184;


    int player = 1;
    struct board_data* best_moves;

    struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);

    int* offsets = malloc(sizeof(int) * 64 * 4);
    compute_offsets(offsets);

    int depth = 9;

    // run the search
    best_moves = start_board_search(p1, p2, p1k, p2k, player, 100, 12);

    // exit for now as this is just a test
    exit(1);

    clock_t start, end;
    double cpu_time_used;

    for (int i = 1; i < depth; i++){
        // get the start time
        start = clock();

        long long n_ply_search_result = n_ply_search(&p1, &p2, &p1k, &p2k, player, piece_loc, offsets, i);

        // get the end time
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        // print the results
        printf("%d ply search result: %lld\n", i, n_ply_search_result);
        printf("%d ply search time: %f\n\n", i, cpu_time_used);

    }

    return 0;
}