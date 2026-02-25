#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "board_search.c"


#define SEARCH_DEPTH 17
#define TIME 1000.0

#define BOOK_DEPTH 15
#define ACCEPTABLE_EVAL_LOSS 10
#define EVAL_CUTOFF -10
#define REQUIRED_OCCURENCES 3

// -----------------------------------------------------------------------------
// Structures for recording move outcomes and positions in the opening book
// -----------------------------------------------------------------------------

typedef struct {
    int wins;
    int losses;
    int ties;
} MoveOutcome;

typedef struct {
    short move;  // internal short move representation
    MoveOutcome outcome;
} MoveEntry;

typedef struct {
    char* key;         // Key: concatenation of p1, p2, p1k, p2k in hex.
    int occurrence;    // How many times this position was encountered
    MoveEntry* moves;  // Dynamic array of moves from this position
    int moveCount;
    int moveCapacity;
} PositionEntry;

// Global array for positions.
PositionEntry* positions = NULL;
int positionsCount = 0;
int positionsCapacity = 0;


// Map of Pdn positions to engine positions
static int pdn_to_engine[33] = {
    0,   // index 0 unused
    62, 60, 58, 56,   // PDN 1-4
    55, 53, 51, 49,   // PDN 5-8
    46, 44, 42, 40,   // PDN 9-12 
    39, 37, 35, 33,   // PDN 13-16
    30, 28, 26, 24,   // PDN 17-20
    23, 21, 19, 17,   // PDN 21-24
    14, 12, 10, 8,    // PDN 25-28
    7, 5, 3, 1        // PDN 29-32
};

// -----------------------------------------------------------------------------
// Function: compute_position_key
// Converts the four bit masks into a concatenated hexadecimal string key.
// The returned key is dynamically allocated and must be freed when no longer needed.
// -----------------------------------------------------------------------------
char* compute_position_key(long long p1, long long p2, long long p1k, long long p2k) {
    // Each 64-bit mask is represented as 16 hex digits.
    // We'll separate them by underscores for readability.
    // Total length: 16*4 + 3 (underscores) + 1 (null terminator) = 67.
    char* key = malloc(67);
    if (!key) {
        fprintf(stderr, "Memory allocation error in compute_position_key.\n");
        exit(1);
    }
    sprintf(key, "%016llX_%016llX_%016llX_%016llX", p1, p2, p1k, p2k);
    return key;
}

// -----------------------------------------------------------------------------
// Function: update_position_book
// Searches the positions array for an entry with the given key.
// If found, updates its occurrence and the outcome counts for the move.
// If not found, creates a new entry. The key parameter is assumed to be
// dynamically allocated (from compute_position_key); if a match is found,
// we free the newly computed key.
// outcome: 1 (win), -1 (loss), 0 (tie)
// -----------------------------------------------------------------------------
void update_position_book(char* key, short move, int outcome) {
    // Search for an existing entry with the same key.
    for (int i = 0; i < positionsCount; i++) {
        if (strcmp(positions[i].key, key) == 0) {
            // Match found; free the new key since we already have one.
            free(key);
            positions[i].occurrence++;
            // Look for the move in the existing entry.
            for (int j = 0; j < positions[i].moveCount; j++) {
                if (positions[i].moves[j].move == move) {
                    if (outcome == 1)
                        positions[i].moves[j].outcome.wins++;
                    else if (outcome == -1)
                        positions[i].moves[j].outcome.losses++;
                    else
                        positions[i].moves[j].outcome.ties++;
                    return;
                }
            }
            // Move not found; add the new move.
            if (positions[i].moveCount == positions[i].moveCapacity) {
                positions[i].moveCapacity *= 2;
                positions[i].moves = realloc(positions[i].moves, sizeof(MoveEntry) * positions[i].moveCapacity);
            }
            positions[i].moves[positions[i].moveCount].move = move;
            positions[i].moves[positions[i].moveCount].outcome.wins = (outcome == 1) ? 1 : 0;
            positions[i].moves[positions[i].moveCount].outcome.losses = (outcome == -1) ? 1 : 0;
            positions[i].moves[positions[i].moveCount].outcome.ties = (outcome == 0) ? 1 : 0;
            positions[i].moveCount++;
            return;
        }
    }
    
    // Not found: add a new entry.
    if (positionsCount == positionsCapacity) {
        positionsCapacity = (positionsCapacity == 0) ? 16 : positionsCapacity * 2;
        positions = realloc(positions, sizeof(PositionEntry) * positionsCapacity);
    }
    
    // Create a new entry.
    PositionEntry newEntry;
    newEntry.key = key; // key is already allocated.
    newEntry.occurrence = 1;
    newEntry.moveCapacity = 4;
    newEntry.moveCount = 0;
    newEntry.moves = malloc(sizeof(MoveEntry) * newEntry.moveCapacity);
    if (!newEntry.moves) {
        fprintf(stderr, "Memory allocation error in update_position_book.\n");
        exit(1);
    }
    // Add the move.
    newEntry.moves[newEntry.moveCount].move = move;
    newEntry.moves[newEntry.moveCount].outcome.wins = (outcome == 1) ? 1 : 0;
    newEntry.moves[newEntry.moveCount].outcome.losses = (outcome == -1) ? 1 : 0;
    newEntry.moves[newEntry.moveCount].outcome.ties = (outcome == 0) ? 1 : 0;
    newEntry.moveCount++;
    
    // Append the new entry to the array.
    positions[positionsCount] = newEntry;
    positionsCount++;
}

// -----------------------------------------------------------------------------
// Function: convert_moves
// Converts a PDN move string (which may include multiple captures, e.g. "14x15x16")
// into an array of engine moves. Each engine move is a short where the high byte
// is the internal start (computed as 65 - 2*pdnSquare) and the low byte is the internal end.
// The function returns the number of moves, and allocates an array of shorts which
// is returned via the moves_out pointer. Caller must free the returned array.
// -----------------------------------------------------------------------------
int convert_moves(const char* pdnMove, short** moves_out) {
    // Create a local copy so that strtok can modify it.
    char moveCopy[128];
    strncpy(moveCopy, pdnMove, sizeof(moveCopy) - 1);
    moveCopy[sizeof(moveCopy) - 1] = '\0';

    // Count how many numbers (squares) are present.
    // Delimiters: '-' and 'x'
    int count = 0;
    char* token = strtok(moveCopy, "-x");
    while (token != NULL) {
        count++;
        token = strtok(NULL, "-x");
    }
    // Need at least two numbers for one move.
    if (count < 2) {
        *moves_out = NULL;
        return 0;
    }

    // Allocate an array for (count - 1) moves.
    short* moves = malloc(sizeof(short) * (count - 1));
    if (!moves) {
        *moves_out = NULL;
        return 0;
    }

    // Parse the numbers again into an array.
    int* squares = malloc(sizeof(int) * count);
    if (!squares) {
        free(moves);
        *moves_out = NULL;
        return 0;
    }
    // Make a fresh copy.
    strncpy(moveCopy, pdnMove, sizeof(moveCopy) - 1);
    moveCopy[sizeof(moveCopy) - 1] = '\0';
    int idx = 0;
    token = strtok(moveCopy, "-x");
    while (token != NULL) {
        squares[idx++] = atoi(token);
        token = strtok(NULL, "-x");
    }

    // For each consecutive pair, compute the engine move.
    for (int i = 0; i < count - 1; i++) {
        int startSquare = squares[i];
        int endSquare = squares[i + 1];
        int internal_start = pdn_to_engine[startSquare];
        int internal_end = pdn_to_engine[endSquare];
        moves[i] = (short)((internal_start << 8) | (internal_end & 0xFF));
    }

    free(squares);
    *moves_out = moves;
    return count - 1;
}


void process_header_line(const char* line, int* outcome) {
    // Check if the header is for result.
    if (strncmp(line, "[Result", 7) == 0) {
        // Find the first double quote.
        const char* start = strchr(line, '"');
        if (start) {
            start++; // Move past the opening quote.
            // Find the closing quote.
            const char* end = strchr(start, '"');
            if (end) {
                int len = end - start;
                char result[16]; // Should be large enough for result strings.
                if (len < sizeof(result)) {
                    strncpy(result, start, len);
                    result[len] = '\0';
                    // Convert result string to outcome value.
                    if (strcmp(result, "1-0") == 0) {
                        *outcome = 1;
                    } else if (strcmp(result, "0-1") == 0) {
                        *outcome = -1;
                    } else if (strcmp(result, "1/2-1/2") == 0) {
                        *outcome = 0;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// process_game: Processes a single PDN game text up to a specified ply depth.
// Now supports multi-step moves (e.g. multiple captures) by using convert_moves().
// -----------------------------------------------------------------------------

void process_game(const char* gameText, int plyDepth) {
    int outcome = 1;

    // Initialize board state using the provided starting bitboard values.
    long long p1 = 6172839697753047040LL;
    long long p2 = 11163050LL;
    long long p1k = 0;
    long long p2k = 0;
    
    struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);
    int* offsets = malloc(sizeof(int) * 64 * 4);
    compute_offsets(offsets);

    // Make a mutable copy of gameText.
    char* gameCopy = strdup(gameText);
    if (!gameCopy) {
        fprintf(stderr, "Memory allocation error\n");
        return;
    }

    char *saveptr_line;
    char *line = strtok_r(gameCopy, "\n", &saveptr_line);
    int ply = 0;
    while (line != NULL && ply < plyDepth) {
        // Trim leading spaces.
        while (*line == ' ') {
            line++;
        }
        
        // Process header lines.
        if (line[0] == '[') {
            process_header_line(line, &outcome);
            line = strtok_r(NULL, "\n", &saveptr_line);
            continue;
        }
        
        // Process moves on this line.
        char *saveptr_token;
        char *token = strtok_r(line, " ", &saveptr_token);
        while (token != NULL && ply < plyDepth) {
            // Skip move numbers (tokens that contain a period).
            if (strchr(token, '.')) {
                token = strtok_r(NULL, " ", &saveptr_token);
                continue;
            }
            // If token starts with '{', skip all tokens until one ends with '}'.
            if (token[0] == '{') {
                // If the token itself contains the closing brace, skip it.
                if (strchr(token, '}') == NULL) {
                    // Otherwise, skip tokens until we find one with '}'
                    while (token != NULL && strchr(token, '}') == NULL) {
                        token = strtok_r(NULL, " ", &saveptr_token);
                    }
                }
                token = strtok_r(NULL, " ", &saveptr_token);
                continue;
            }
            // Skip tokens that don't have any digits.
            if (strpbrk(token, "0123456789") == NULL) {
                token = strtok_r(NULL, " ", &saveptr_token);
                continue;
            }
            // Convert the token into one or more engine moves.
            short* moveList = NULL;
            int moveCount = convert_moves(token, &moveList);
            if (moveCount == 0) {
                fprintf(stderr, "Error parsing move: %s\n", token);
                token = strtok_r(NULL, " ", &saveptr_token);
                continue;
            }
            // Process each engine move.
            for (int i = 0; i < moveCount; i++) {
                short engine_move = moveList[i];
                unsigned long long key = compute_position_key(p1, p2, p1k, p2k);
                update_position_book(key, engine_move, outcome);

                int internal_start = engine_move >> 8;
                int internal_end = engine_move & 0xFF;

                int initial_piece_type = get_piece_at_location(p1, p2, p1k, p2k, internal_start);
                int jumped_piece_type = update_board(&p1, &p2, &p1k, &p2k, internal_start, internal_end);

                //human_readble_board(p1, p2, p1k, p2k);

                update_piece_locations(internal_start, internal_end, piece_loc);
                //print_set(piece_loc);
                ply++;
            }
            free(moveList);
            token = strtok_r(NULL, " ", &saveptr_token);
        }
        line = strtok_r(NULL, "\n", &saveptr_line);
    }
    free(gameCopy);
    free(offsets);
    free(piece_loc);
}

// -----------------------------------------------------------------------------
// validate_book_position: Given a board state (p1, p2, p1k, p2k), use negmax 
// to get the best evaluation and best move for this position. Then, for each move
// already in the book for this position, re-evaluate it by applying the move and
// calling negmax. If its evaluation is worse than (best_eval - 20), remove that move.
// -----------------------------------------------------------------------------
void validate_book_position(long long p1, long long p2, long long p1k, long long p2k,
                              int player, float search_time, int search_depth) {
    // Setup board state for the search.
    struct set *piece_loc = get_piece_locations(p1, p2, p1k, p2k);
    int *offsets = malloc(sizeof(int) * 64 * 4);
    compute_offsets(offsets);
    clock_t start_time = clock();
    struct board_evaler *evaler = board_evaler_constructor(p1 | p1k, p2 | p2k, search_depth, search_time, start_time);
    evaler->search_depth = search_depth;
    evaler->max_depth = search_depth * 2;
    long long hash = get_hash(p1, p2, p1k, p2k, evaler->hash_table);
    struct hash_table_entry* table_entry;
    int best_eval;
    
    // Get the best evaluation (and best move) from negmax.
    while (table_entry == Null) {
        best_eval = negmax(&p1, &p2, &p1k, &p2k, player, piece_loc, search_depth,
                                -INFINITY, INFINITY, evaler, hash, 0, 0);
        table_entry = get_hash_entry(evaler->hash_table, hash, evaler->search_depth, search_depth);
        if (table_entry == Null) {
            printf("NULL table entry");
        }
    }

    int engine_best_move = table_entry->best_move;
    printf("Search Depth: %d\t Extended Depth: %d\t Best Eval: %d\n", search_depth, evaler->extended_depth, best_eval);
    
    // Compute the key for the current board state.
    char *key = compute_position_key(p1, p2, p1k, p2k);

    // Look for the matching entry in our global book.
    for (int i = 0; i < positionsCount; i++) {
        if (strcmp(positions[i].key, key) == 0) {
            int new_move_count = 0;
            // For each move in the book for this position...
            for (int j = 0; j < positions[i].moveCount; j++) {
                short cand_move = positions[i].moves[j].move;
                // Decode the engine move into internal coordinates.
                int internal_start = cand_move >> 8;
                int internal_end   = cand_move & 0xFF;
                
                // Save current state info if needed.
                int next_hash = update_hash(p1, p2, p1k, p2k, internal_start, internal_end, hash, evaler);
                int initial_piece_type = get_piece_at_location(p1, p2, p1k, p2k, internal_start);
                int jumped_piece_type  = update_board(&p1, &p2, &p1k, &p2k, internal_start, internal_end);
                update_piece_locations(internal_start, internal_end, piece_loc);
                
                // Evaluate the candidate move.
                int cand_eval = -negmax(&p1, &p2, &p1k, &p2k, player ^ 0x3, piece_loc, search_depth - 1,
                                       -INFINITY, INFINITY, evaler, next_hash, 0, 0);
                
                // Undo the move.
                undo_piece_locations_update(internal_start, internal_end, piece_loc);
                undo_board_update(&p1, &p2, &p1k, &p2k, internal_start, internal_end, jumped_piece_type, initial_piece_type);
                
                // If the candidate evaluation is within 20 points of best, keep it.
                if (cand_eval >= best_eval - ACCEPTABLE_EVAL_LOSS && cand_eval >= EVAL_CUTOFF) {
                    // Move the candidate move to the new index.
                    positions[i].moves[new_move_count] = positions[i].moves[j];
                    new_move_count++;
                } else {
                    // Optionally print that we are removing this move.
                    printf("Removing move %d %d with eval %d, best eval %d\n", 
                        internal_start, internal_end, cand_eval, best_eval);
                }
            }
            positions[i].moveCount = new_move_count;
            break;
        }
    }

    // Ensure the engine's best move is in the book.
    for (int i = 0; i < positionsCount; i++) {
        if (strcmp(positions[i].key, key) == 0) {
            bool best_found = false;
            for (int j = 0; j < positions[i].moveCount; j++) {
                if (positions[i].moves[j].move == (short)engine_best_move) {
                    best_found = true;
                    break;
                }
            }
            if (!best_found) {
                if (positions[i].moveCount == positions[i].moveCapacity) {
                    positions[i].moveCapacity *= 2;
                    positions[i].moves = realloc(positions[i].moves, sizeof(MoveEntry) * positions[i].moveCapacity);
                }
                positions[i].moves[positions[i].moveCount].move = (short)engine_best_move;
                positions[i].moves[positions[i].moveCount].outcome.wins = 0;
                positions[i].moves[positions[i].moveCount].outcome.losses = 0;
                positions[i].moves[positions[i].moveCount].outcome.ties = 0;
                positions[i].moveCount++;
            }
            break;
        }
    }
    
    free(key);
    free(offsets);
    free(piece_loc);
    end_board_search(evaler);
}

// -----------------------------------------------------------------------------
// validate_book: Loop over all positions in the global book, parse their keys
// to recover the board state, then call validate_book_position() for each.
// -----------------------------------------------------------------------------
void validate_book(float search_time, int search_depth) {
    for (int i = 0; i < positionsCount; i++) {
        long long p1, p2, p1k, p2k;
        if (sscanf(positions[i].key, "%016llX_%016llX_%016llX_%016llX",
                   &p1, &p2, &p1k, &p2k) != 4) {
            fprintf(stderr, "Error parsing key: %s\n", positions[i].key);
            continue;
        }

        int piece = get_piece_at_location(p1, p2, p1k, p2k, positions[i].moves[0].move >> 8);
        int player = (piece % 2 == 1) ? 1 : 2;

        validate_book_position(p1, p2, p1k, p2k, player, search_time, search_depth);

        // Print the precentage of processed positions
        double percentage = ((double)i / (double)positionsCount) * 100.0;
        printf("Processed: %.2f%% of positions\n", percentage);
    }
}

void write_book_to_file(const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s for writing\n", filename);
        return;
    }
    // For each position in the global book
    for (int i = 0; i < positionsCount; i++) {
        // Write the key first.
        fprintf(fp, "%s", positions[i].key);
        // Write each move separated by a space.
        for (int j = 0; j < positions[i].moveCount; j++) {
            // Print the engine move in hexadecimal (adjust formatting if needed)
            fprintf(fp, " %04X", positions[i].moves[j].move);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// -----------------------------------------------------------------------------
// Main entry point
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // Open the PDN file.
    FILE *fp = fopen("CheckersGames.pdn", "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open CheckersGames.pdn\n");
        return 1;
    }

    // Read the entire file into memory.
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *fileBuffer = malloc(fileSize + 1);
    if (!fileBuffer) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        return 1;
    }
    fread(fileBuffer, 1, fileSize, fp);
    fileBuffer[fileSize] = '\0';
    fclose(fp);

    // Process each game block.
    char *start = fileBuffer;
    char *sep = NULL;
    while ((sep = strstr(start, "\n\n")) != NULL) {
        *sep = '\0';  // Temporarily terminate the current game block
        process_game(start, BOOK_DEPTH);
        start = sep + 2; // Move past the "\n\n" separator
    }

    // Filter all positions that only occured once
    int filtered_count = 0;
    for (int i = 0; i < positionsCount; i++) {
        if (positions[i].occurrence > REQUIRED_OCCURENCES) {
            // Keep the position.
            positions[filtered_count++] = positions[i];
        } else {
            // Free the memory for positions that occurred only once.
            free(positions[i].key);
            free(positions[i].moves);
        }
    }
    positionsCount = filtered_count;

    // Optionally, shrink the positions array if desired:
    positions = realloc(positions, sizeof(PositionEntry) * positionsCount);

    // Output the aggregated opening book data from the positions array.
    for (int i = 0; i < positionsCount; i++) {
        printf("Position Key: %s, Occurrences: %d\n", positions[i].key, positions[i].occurrence);
        for (int j = 0; j < positions[i].moveCount; j++) {
            printf("  Move: %d %d, Wins: %d, Losses: %d, Ties: %d\n",
                   positions[i].moves[j].move >> 8,
                   positions[i].moves[j].move & 0xff,
                   positions[i].moves[j].outcome.wins,
                   positions[i].moves[j].outcome.losses,
                   positions[i].moves[j].outcome.ties);
        }
    }

    // Print some final details
    printf("positionsCount: %d", positionsCount);

    validate_book(TIME, SEARCH_DEPTH);

    write_book_to_file("book_moves.txt");

    // Print some final details
    printf("positionsCount: %d", positionsCount);

    // Clean up the allocated positions array.
    for (int i = 0; i < positionsCount; i++) {
        free(positions[i].key);
        free(positions[i].moves);
    }
    free(positions);
    free(fileBuffer);

    return 0;
}