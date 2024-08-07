// Author: Collin Kees
// language: C
// Description: this program serves as the C library for move generation and evaluation of checkers boards

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "board_eval.c" // includes set, transposition table, and board evaler

// Define some constants
#define True 1
#define False 0
#define Null 0
#define INFINITY 1000000000
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define PRINT_OUTPUT 1


#define PY_SSIZE_T_CLEAN
#include <Python.h>


// Define some functions TODO use a header file for this
struct board_data;
struct board_data *board_data_constructor(int player, int move_start, int move_end);
struct set* get_piece_locations(long long p1, long long p2, long long p1k, long long p2k);
void update_piece_locations(int piece_loc_initial, int piece_loc_after, struct set* piece_loc);
void undo_piece_locations_update(int piece_loc_initial, int piece_loc_after, struct set* piece_loc);
void sort_moves(struct board_data* ptr, int player);
int get_next_board_state(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, int player, int piece_type, char* offsets);
int get_piece_at_location(long long p1, long long p2, long long p1k, long long p2k, int pos);
int update_board(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after);
void undo_board_update(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type, int initial_piece_type);
int generate_all_moves(long long p1, long long p2, long long p1k, long long p2k, int player, int* moves, struct set* piece_loc, char* offsets, int* jump);
int generate_moves(long long p1, long long p2, long long p1k, long long p2k, int pos, int* save_loc, char* offsets, int only_jump);
int negmax(long long* p1, long long* p2, long long* p1k, long long* p2k, int player,
    struct set* piece_loc, int depth, int alpha, int beta,
    struct board_evaler* evaler, unsigned long long int hash, int depth_abs, int node_num);
struct search_info* start_board_search(long long p1, long long p2, long long p1k, long long p2k, int player, float search_time, int search_depth);
void human_readble_board(long long p1, long long p2, long long p1k, long long p2k);
long long n_ply_search(long long* p1, long long* p2, long long* p1k, long long* p2k, int player, struct set* piece_loc, int* offsets, int depth);
void quick_sort(struct board_data* ptr, int low, int high);
int partition(struct board_data* ptr, int low, int high);
unsigned long long int update_hash(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, unsigned long long int hash, struct board_evaler* evaler);
struct board_data* get_best_move(struct board_data *head, int player);
void end_board_search(struct board_evaler* evaler);
int free_board_data(struct board_data* data);
void print_line(long long p1, long long p2, long long p1k, long long p2k, unsigned long long hash, struct board_evaler* evaler);


// Holds the head to the board_tree and the evaler struct
struct search_info {
    struct board_evaler* evaler;
    struct hash_table_entry* entry;
};

// Python comunication code 

// Python function to search the board
static PyObject* search_position(PyObject *self, PyObject *args){
    unsigned long long int p1, p2, p1k, p2k, player, search_depth;
    float search_time;

    // Get the arguments from python (p1, p2, p1k, p2k, player, search_time, search_depth)
    if (!PyArg_ParseTuple(args, "KKKKKfK", &p1, &p2, &p1k, &p2k, &player, &search_time, &search_depth))
        return NULL;

    // Get the Result
    struct search_info* search_info = start_board_search(p1, p2, p1k, p2k, player, search_time, search_depth);

    // Package the relevant data into a python tuple
    PyObject* py_list = PyList_New(0);
    PyObject* py_tuple;
    py_tuple = Py_BuildValue("ii", (search_info->entry->best_move >> 8) & 0xFF, search_info->entry->best_move & 0xFF);
    PyList_Append(py_list, py_tuple);

    
    // Get some stats about the search
    py_tuple = Py_BuildValue("KKKKi", search_info->evaler->search_depth, search_info->evaler->extended_depth,
                        search_info->evaler->nodes, search_info->evaler->hash_table->num_entries,
                        search_info->entry->eval);
    PyList_Append(py_list, py_tuple);

    // Free the search tree the evaler and the transposition table
    end_board_search(search_info->evaler);

    // Return the python tuple
    return py_list;
}

// Tell the pyhton interpreter about the functions we want to use
static PyMethodDef c_board_search_methods[] = {
    {"search_position", search_position, METH_VARARGS, 
    "takes the 6 64bit integers for the board\
     the first 4 come from convert to bit board, the remaining is the player\
     and the time and depth to search for"},
    {NULL, NULL, 0, NULL}
};

// Define the module
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

// a function to round the float to 2 decimal places
float round_float(float num){
    return (float)((int)(num * 100 + 0.5)) / 100;
}

// get the location of all the piece's on the board to avoid looping over unused spots
// takes the board as arguments
// returns a set of all the pieces on the board
// note: assumes a valid board
struct set* get_piece_locations(long long p1, long long p2, long long p1k, long long p2k){
    struct set* piece_loc = create_set();
    for (int i = 0; i < 64; i++){
        if (get_piece_at_location(p1, p2, p1k, p2k, i) != 0){
            set_add(piece_loc, i);
        }
    }
    populate_set_array(piece_loc);

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

// after sorting or at anytime before continuing to search the tree, move the moves in the transposition table
// to the start of the move list
void order_moves(int* moves, int num_moves, struct hash_table_entry* entry, struct killer_entry* killer_entry){
    short best_move = (entry != NULL) ? entry->best_move : NO_MOVE;
    int sorted_index = 0;
    // move the transposition table moves to the start of the move list
    if (best_move != 0){
        char start = best_move >> 8;
        char end = best_move & 0xFF;
        // move the best move to the start of the move list
        for (int i = sorted_index; i < num_moves; i++){
            if (moves[i * 2] == start && moves[i * 2 + 1] == end){
                int temp = moves[i * 2];
                moves[i * 2] = moves[sorted_index * 2];
                moves[sorted_index * 2] = temp;
                temp = moves[i * 2 + 1];
                moves[i * 2 + 1] = moves[sorted_index * 2 + 1];
                moves[sorted_index * 2 + 1] = temp;
                sorted_index++;
                break;
            }
        }
    }

    // move the killer moves
    if (killer_entry->move1 != best_move && killer_entry->move1 != 0){
        char start = killer_entry->move1 >> 8;
        char end = killer_entry->move1 & 0xFF;
        // move the killer move to the start of the move list
        for (int i = sorted_index; i < num_moves; i++){
            if (moves[i * 2] == start && moves[i * 2 + 1] == end){
                int temp = moves[i * 2];
                moves[i * 2] = moves[sorted_index * 2];
                moves[sorted_index * 2] = temp;
                temp = moves[i * 2 + 1];
                moves[i * 2 + 1] = moves[sorted_index * 2 + 1];
                moves[sorted_index * 2 + 1] = temp;
                sorted_index++;
                break;   
            }
        }
    }
    if (killer_entry->move2 != best_move && killer_entry->move2 != 0){
        char start = killer_entry->move2 >> 8;
        char end = killer_entry->move2 & 0xFF;
        // move the killer move to the start of the move list
        for (int i = sorted_index; i < num_moves; i++){
            if (moves[i * 2] == start && moves[i * 2 + 1] == end){
                int temp = moves[i * 2];
                moves[i * 2] = moves[sorted_index * 2];
                moves[sorted_index * 2] = temp;
                temp = moves[i * 2 + 1];
                moves[i * 2 + 1] = moves[sorted_index * 2 + 1];
                moves[sorted_index * 2 + 1] = temp;
                sorted_index++;
                break;    
            }
        }
    }
}

// find the state of the next board after a move
// returns 1 if the moving player is player 2 returns 0 if the player is player 1
// pos is the position the piece will be after the first jump and leading in to the second one
// convenion is that player 1 has internal state of 1 and player 2 has internal state of 2 (ie. 1 is red and 2 is black in a normal match)
int get_next_board_state(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, int player, int piece_type, char* offsets){
    // check if the last move was a non-promoting jump, if it wasn't invert the state
    if ((abs(pos_init - pos_after) < 10) || ((pos_after < 8 || pos_after > 56) && piece_type <= 2)){
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
int get_piece_at_location(long long p1, long long p2, long long p1k, long long p2k, int pos){
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
// returns 0 if no piece was captured
int update_board(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after){
    int piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, piece_loc_initial);
    int return_value = 0;
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
void undo_board_update(long long* p1, long long* p2, long long* p1k, long long* p2k, int piece_loc_initial, int piece_loc_after, int jumped_piece_type, int initial_piece_type){
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
        // check if the piece should be unkinged
        if (initial_piece_type == 1){
            *p1k = *p1k ^ (1ll << piece_loc_initial);
            *p1 = *p1 ^ (1ll << piece_loc_initial);
        }
    } else if (piece_type == 4){
        *p2k = *p2k ^ (1ll << piece_loc_initial);
        *p2k = *p2k ^ (1ll << piece_loc_after);
        // check if the piece should be unkinged
        if (initial_piece_type == 2){
            *p2k = *p2k ^ (1ll << piece_loc_initial);
            *p2 = *p2 ^ (1ll << piece_loc_initial);
        }
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
unsigned long long int update_hash(long long p1, long long p2, long long p1k, long long p2k, int pos_init, int pos_after, unsigned long long int hash, struct board_evaler* evaler){
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
int generate_all_moves(long long p1, long long p2, long long p1k, long long p2k, int player, int* moves, struct set* piece_loc, char* offsets, int* jump){
    // setup variables
    int num_moves = 0;
    int move_from_pos = 0;
    int type;
    int friendly_pieces = 2 + player;
    int num_pieces = populate_set_array(piece_loc);

    // loop over all the locations with a piece and generate the moves for each one
    for (int i = 0; i < num_pieces; i++){
        // check if it is a valid piece
        type = get_piece_at_location(p1, p2, p1k, p2k, piece_loc->array[i]);
        if (type == player || type == friendly_pieces){
            // generate the moves for the piece
            move_from_pos = generate_moves(p1, p2, p1k, p2k, piece_loc->array[i], moves + (num_moves * 2), offsets, *jump);
            if (move_from_pos == -1){
                num_moves = 0;
                *jump = 1;
                move_from_pos = generate_moves(p1, p2, p1k, p2k, piece_loc->array[i], moves + (num_moves * 2), offsets, *jump);
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
int generate_moves(long long p1, long long p2, long long p1k, long long p2k, int pos, int* save_loc, char* offsets, int only_jump){
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

int adjust_mate_score(int eval) {
    return (eval > 900) ?
         eval - 1 : ((eval < -900) ?
         eval + 1 : eval);
}

// a function that decides if a search should be extended or reduced at a certain node
int should_extend_or_reduce(int depth, int depth_abs, int node_num, int eval, int jump, int player, long long p1All, long long p2All,
                            struct hash_table_entry* table_entry,
                            struct board_evaler* evaler) {
    if (depth_abs >= evaler->max_depth){
        return -100;
    }

    if (jump){
        return depth;
    }

    // Extract the node type from the table entry
    int node_type = UNKNOWN_NODE;
    if (table_entry != NULL){
        node_type = table_entry->node_type;
    }

    // PV-node extension
    if (node_type == PV_NODE && depth_abs > 8){
           depth++;
    }

    int p1_initial_piece_count = evaler->initial_piece_count_p1;
    int p2_initial_piece_count = evaler->initial_piece_count_p2;
    int p1_piece_count = get_bits_set(p1All);
    int p2_piece_count = get_bits_set(p2All);

    // Reduce lines that are not likely to be good ie where one player has lost 2 or more pieces and the other has not
    if (player == 1 && (p2_initial_piece_count - p2_piece_count) - (p1_initial_piece_count - p1_piece_count) >= 2){
        depth -= 2;
    } else if (player == 2 && (p1_initial_piece_count - p1_piece_count) - (p2_initial_piece_count - p2_piece_count) >= 2){
        depth -= 2;
    }

    // Reduce late moves
    if (node_num > 5 && depth_abs > 8){
        depth--;
    }

    return depth;
}

// search the board for the best move recursivly and return the best eval that can be achived from that board position
// the best_moves struct will be populated to the search depth at the end of this search
int negmax(long long* p1, long long* p2, long long* p1k, long long* p2k, int player,
    struct set* piece_loc, int depth, int alpha, int beta,
    struct board_evaler* evaler, unsigned long long int hash, int depth_abs, int node_num){
    // setup variables
    int player_next;
    int board_eval = -2000;
    int num_moves;
    int alpha_orig = alpha;
    int initial_piece_type;
    unsigned long long int next_hash;

    // update search stats
    evaler->nodes++;
    evaler->avg_depth += depth_abs;

    if (depth_abs > evaler->extended_depth && depth >= 0){
        evaler->extended_depth = depth_abs;
    }

    // if nodes is divisible by 10000, check the time
    if (evaler->nodes % 10000 == 0 && evaler->nodes != 0){
        clock_t current_time = clock();
        double cpu_time_used = ((double)(current_time - evaler->start_time)) / CLOCKS_PER_SEC;
        if (cpu_time_used > evaler->time_limit){
            return INFINITY;

        }
    }
    
    // check if this board has been searched to depth before and if so return the eval from the hash table
    // if the value is not a PV-node then use the info that can be used to prune the search
    struct hash_table_entry* table_entry = get_hash_entry(evaler->hash_table, hash, evaler->search_depth, depth);
    if (table_entry != NULL && table_entry->depth >= depth && table_entry->player == player) {
        if (table_entry->node_type == PV_NODE) {
            return adjust_mate_score(table_entry->eval);
        } 
        else if (table_entry->node_type == LOWER_BOUND) {
            alpha = max(alpha, table_entry->eval);
        } 
        else if (table_entry->node_type == UPPER_BOUND) {
            beta = min(beta, table_entry->eval);
        }
        if (alpha >= beta) {
            return adjust_mate_score(table_entry->eval);
        }
    }

    // check for a draw by repetition
    if (get_draw_entry(evaler->draw_table, hash) == 2){
        return 0;
    }

    // get the moves for this board and player combo
    int moves[96];
    int force_captures = (depth <= 0) ? 1 : 0;
    int is_jump = force_captures;

    num_moves = generate_all_moves(*p1, *p2, *p1k, *p2k, player, &moves[0], piece_loc, evaler->piece_offsets, &is_jump);

    // before continuing from this block of code, check if the hash table has a value for this board
    // if it does then use that to order the moves
    if (num_moves > 1){
        order_moves(&moves[0], num_moves, table_entry, evaler->killer_table->table + depth_abs);
    }

    // if there are no move then a player must have won, or there are no captures so end this branch (only count as a win if captures_only is false)
    // this is not a perfect check but it should be good enough
    if (num_moves == 0){
        // if there are no moves and captures only is false then a win has occured eval who won and return
        if (!force_captures){
            return -1000;
        }
        // if not already generating all moves then generate all moves regardless of depth to check for a win
        int captures = 0;
        num_moves = generate_all_moves(*p1, *p2, *p1k, *p2k, player, &moves[0], piece_loc, evaler->piece_offsets, &captures);
        if (num_moves == 0){
            return -1000;
        }

        // if there are no moves and captures only is true then we found the end of a catures only search evaluate the position and return
        return get_eval(*p1, *p2, *p1k, *p2k, player, piece_loc, evaler);
    }

    depth = should_extend_or_reduce(depth, depth_abs, node_num, board_eval, force_captures, player, *p1 | *p1k, *p2 | *p2k, table_entry, evaler);

    // the the next boards are ready to be searched so begin the search!
    short best_move = NO_MOVE;
    int eval, flip, next_alpha, next_beta;
    for (int i = 0; i < num_moves; i++){
        // prepare moves
        int move_start = moves[i * 2];
        int move_end = moves[(i * 2) + 1];

        // get the hash for this next board (always do this before the move is made on the board)
        next_hash = update_hash(*p1, *p2, *p1k, *p2k, move_start, move_end, hash, evaler);

        // update the board and set values
        initial_piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, move_start);
        int jumped_piece_type = update_board(p1, p2, p1k, p2k, move_start, move_end);
        update_piece_locations(move_start, move_end, piece_loc);
        player_next = get_next_board_state(*p1, *p2, *p1k, *p2k, move_start, move_end, player, initial_piece_type, evaler->piece_offsets);
        add_draw_entry(evaler->draw_table, next_hash);

        // only flip the eval if the player changed
        flip = (player != player_next) ? -1 : 1;
        next_alpha = (flip == 1) ? alpha : -beta;
        next_beta = (flip == 1) ? beta : -alpha;
        eval = negmax(p1, p2, p1k, p2k, player_next, piece_loc, depth - 1,
                    next_alpha, next_beta, evaler, next_hash,
                    depth_abs + 1, i) * flip;
        
        // undo the update to the board and piece locations
        undo_piece_locations_update(move_start, move_end, piece_loc);
        undo_board_update(p1, p2, p1k, p2k, move_start, move_end, jumped_piece_type, initial_piece_type);
        remove_draw_entry(evaler->draw_table, next_hash);

        // if the eval is infinity the search is trying to end so return
        if (eval == INFINITY || eval == -INFINITY){
            return INFINITY;
        }

        // alpha beta prunning
        if (board_eval < eval){
            board_eval = eval;
            best_move = (move_start << 8) | move_end; 
        }
        if (board_eval > alpha){
            alpha = board_eval;
        }
        // prune if a cut off occurs
        if (alpha >= beta){
            update_killer_table(evaler->killer_table, depth_abs, move_start, move_end);
            break;
        }
    }

    // if alpha and beta have improved then this is a PV node
    char node_type;
    if (board_eval <= alpha_orig){
        node_type = UPPER_BOUND;
    }
    else if (board_eval >= beta){
        node_type = LOWER_BOUND;
    }
    else {
        node_type = PV_NODE;
    }

    // store the eval in the hash table
    add_hash_entry(evaler->hash_table, hash, board_eval, depth, evaler->search_depth, player, best_move, node_type);
    
    return adjust_mate_score(board_eval);
}

// marks pv nodes in the hash table
void PV_labler(long long* p1, long long* p2, long long* p1k, long long* p2k, int player,
    struct set* piece_loc, int depth, unsigned long long int hash, struct board_evaler* evaler) {

    struct hash_table_entry* table_entry = get_hash_entry(evaler->hash_table, hash, evaler->search_depth, depth);
    if (table_entry == NULL || table_entry->best_move == NO_MOVE || depth <= 0) {
        return;
    }

    table_entry->node_type = PV_NODE;

    short move_start = (table_entry->best_move >> 8) & 0xFF;
    short move_end = table_entry->best_move & 0xFF;

    hash = update_hash(*p1, *p2, *p1k, *p2k, move_start, move_end, hash, evaler);
    int initial_piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, move_start);
    int jumped_piece_type = update_board(p1, p2, p1k, p2k, move_start, move_end);
    update_piece_locations(move_start, move_end, piece_loc);
    int player_next = get_next_board_state(*p1, *p2, *p1k, *p2k, move_start, move_end, player, initial_piece_type, evaler->piece_offsets);

    PV_labler(p1, p2, p1k, p2k, player_next, piece_loc, depth - 1, hash, evaler);

    undo_piece_locations_update(move_start, move_end, piece_loc);
    undo_board_update(p1, p2, p1k, p2k, move_start, move_end, jumped_piece_type, initial_piece_type);   
}

int MTDF(long long* p1, long long* p2, long long* p1k, long long* p2k, int player,
    struct set* piece_loc, int depth, int f, struct board_evaler* evaler,
    unsigned long long int hash) {
    
    int g = f;
    int upper_bound = INFINITY;
    int lower_bound = -INFINITY;



    while (lower_bound < upper_bound) {
        int beta = (g == lower_bound) ? g + 1 : g;

        g = negmax(p1, p2, p1k, p2k, player, piece_loc, depth, beta - 1, beta,
            evaler, hash, 0, 0);

        if (g < beta) {
            upper_bound = g;
        } else {
            lower_bound = g;
        }
    }

    // follow the pv line labeling each node as a PV node
    PV_labler(p1, p2, p1k, p2k, player, piece_loc, depth, hash, evaler);

    return g;
}



// prepare needed memory for a search and call the search function to find the best move and return a pointer to the memory 
// location that holds the moves for the board ordered in the order best to worst
struct search_info* start_board_search(long long p1, long long p2, long long p1k, long long p2k, int player, float search_time, int search_depth){
    struct search_info* return_struct = malloc(sizeof(struct search_info));

    // set for the piece locations
    struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);
    int moves[96];

    clock_t start, end;
    start = clock();
    struct board_evaler* evaler = board_evaler_constructor(p1 | p1k, p2 | p2k, search_depth, search_time, start);

    long long hash = get_hash(p1, p2, p1k, p2k, evaler->hash_table);
    double cpu_time_used;
    int depth;
    int extended_depth;
    int terminate = 0;
    int eval_ = 0;
    int jump = 0;
    
    // call the search function
    for (int i = 1; i <= search_depth; i++){
        if (terminate == 1){
            break;
        }
        // update the evalers search depth
        evaler->search_depth = i;
        evaler->max_depth = min(max(i + 10, 5), search_depth);

        // search with MTDF 
        //eval_ = MTDF(&p1, &p2, &p1k, &p2k, player, piece_loc, i, eval_, evaler, hash);

        eval_ = negmax(&p1, &p2, &p1k, &p2k, player, piece_loc, i, -INFINITY, INFINITY, evaler, hash, 0, 0);

        // get the end time
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        // if the eval is infinity the search is trying to end
        if (eval_ == INFINITY || eval_ == -INFINITY){
            terminate = 1;
            evaler->search_depth = depth;
            evaler->extended_depth = extended_depth;
            break;
        }

        depth = i;
        extended_depth = evaler->extended_depth;
        if (depth > extended_depth){
            extended_depth = depth;
        }

        if (PRINT_OUTPUT){
            printf("\rPLY: %d\t PLYEX: %d\t Eval: %d   ", depth, evaler->extended_depth, eval_);
        }

        if (cpu_time_used > search_time){
            terminate = 1;
        }

        // only terminate once the eval has been a mate score for 3 plys
        else if (eval_ > 900 || eval_ < -900){
            terminate = 1;
        }

        // if there is only one move left then terminate
        else if (generate_all_moves(p1, p2, p1k, p2k, player, &moves[0], piece_loc, evaler->piece_offsets, &jump) == 1){
            terminate = 1;
        }
    }

    // get the table entry for the best move and eval
    struct hash_table_entry* table_entry = get_hash_entry(evaler->hash_table, hash, evaler->search_depth, depth);

    // print the output of the engine
    if (PRINT_OUTPUT) {
        printf("\r                                                                   \r");
        printf("Search Results:\n");
        printf("HashTable Hit ratio: %lld\n", (evaler->hash_table->hit_count * 100) / (evaler->hash_table->hit_count + evaler->hash_table->miss_count));
        printf("HashTable Usage: %lld\n", (evaler->hash_table->num_entries * 100llu) / evaler->hash_table->total_size);
        printf("Nodes/s: %fM\n", round_float(((double)evaler->nodes / (cpu_time_used + 0.01)) / 1000000.0));
        printf("Time: %fs\n", cpu_time_used);
        printf("Depth: %d\n", evaler->extended_depth);
        printf("Avg depth: %lld\n", evaler->avg_depth / evaler->nodes);
        printf("Eval: %d\n\n", table_entry->eval);
    }

    // print the line of best moves to the terminal (deguggigng)
    print_line(p1, p2, p1k, p2k, hash, evaler);

    free(piece_loc);

    return_struct->evaler = evaler;
    return_struct->entry = table_entry;
    
    
    // return the best moves and evaler
    return return_struct;
}

// clean up after the search has been concluded and the thread is about to exit
void end_board_search(struct board_evaler* evaler){
    // free the memory used by the evaler
    free(evaler->hash_table->table);
    free(evaler->hash_table);
    free(evaler->killer_table->table);
    free(evaler->killer_table);
    free(evaler->piece_offsets);
    free_draw_table(evaler->draw_table);
    free(evaler);
}

// search to the depth specified and count to total amount of boards for a certain depth
// this is used to test if the move generation is working as expected and to see how much time is spent
long long n_ply_search(long long* p1, long long* p2, long long* p1k, long long* p2k, int player, struct set* piece_loc, int* offsets, int depth){
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
        player_next = get_next_board_state(*p1, *p2, *p1k, *p2k, pos_init, pos_final, player, intitial_piece_type, offsets);

        // call the function recursivly
        total_boards += n_ply_search(p1, p2, p1k, p2k, player_next, piece_loc, offsets, depth - 1);

        // undo the update to the board and piece locations
        undo_piece_locations_update(pos_init, pos_final, piece_loc);
        undo_board_update(p1, p2, p1k, p2k, pos_init, pos_final, jumped_piece_type, intitial_piece_type);
    }
    
    return total_boards;

}

// print the board to standard out in a human readable format (for debugging)
void human_readble_board(long long p1, long long p2, long long p1k, long long p2k){
    printf("  0 1 2 3 4 5 6 7 \n");
    for (int row = 0; row < 8; row++){
        printf("%d ", row);
        for (int col = 0; col < 8; col++){
            int piece = get_piece_at_location(p1, p2, p1k, p2k, (row * 8) + col);
            if (piece == 0){
                printf("  ");
            }
            else if (piece == 1){
                printf("o ");
            }
            else if (piece == 2){
                printf("x ");
            }
            else if (piece == 3){
                printf("O ");
            }
            else if (piece == 4){
                printf("X ");
            }
        }
        printf("\n");
    }
}

// print the expected line to standard out in a human readable format (for debugging)
void print_line(long long p1, long long p2, long long p1k, long long p2k, unsigned long long hash, struct board_evaler* evaler){
    // use the hash table to make the line of moves
    printf("PV ");
    int depth = 0;
    while (depth < evaler->search_depth) {
        struct hash_table_entry* table_entry = get_hash_entry(evaler->hash_table, hash, evaler->search_depth, depth);
        if (table_entry == NULL || table_entry->best_move == NO_MOVE || table_entry->node_type != PV_NODE){
            break;
        }

        // convert from 64 bit board to 32 square representation
        int start = (65 - (table_entry->best_move >> 8 & 0xFF)) / 2;
        int end = (65 - (table_entry->best_move & 0xFF)) / 2;


        printf("- %d %d ", start, end);

        hash = update_hash(p1, p2, p1k, p2k, (table_entry->best_move >> 8) & 0xFF, table_entry->best_move & 0xFF, hash, evaler);
        update_board(&p1, &p2, &p1k, &p2k, (table_entry->best_move >> 8) & 0xFF, table_entry->best_move & 0xFF);
        depth++;
    }
    printf("\n");
    //human_readble_board(p1, p2, p1k, p2k);
}


// main function
// runs a n_ply search to verify the move generation is working as expected
// also benchmarks the time it takes to generate the moves
int main(){

    printf("beginning tests\n");

    // setup initial board and search structures
    // starting bit values
    long long p1 = 6172839697753047040;
    long long p2 = 11163050;
    long long p1k = 0;
    long long p2k = 0;

    // end game player 2 winning endgame
    //long long p1 = 5838922414443986944;
    //long long p2 = 8409224;
    //long long p1k = 0;
    //long long p2k = 288230376154333184;


    int player = 1;

    struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);

    int* offsets = malloc(sizeof(int) * 64 * 4);
    compute_offsets(offsets);

    int depth = 9;

    clock_t start, end;
    double cpu_time_used;

    for (int i = 2; i < depth; i++){
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

    // now do a search to depth 20 and print the results
    start_board_search(p1, p2, p1k, p2k, player, 10, 40);

    return 0;
}