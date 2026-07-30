// Opening book builder: a PDN game collection in, `book_moves.txt` out.
// Author: Collin Kees
//
// Build:  gcc -O2 -o book.exe src/engine/opening_book.c
//         cl /O2 /Fe:book.exe src/engine/opening_book.c
// Run:    book.exe --help
//
// The book is built in two stages, and the split between them is deliberate:
// the game collection chooses the positions, the engine chooses the moves.
//
//   1. Aggregation. Replay every game in the PDN file, record the position
//      before each of the first BOOK_DEPTH plies, and keep the positions that
//      enough different games reached. This is what 22,000 master games know and
//      a search cannot tell you - which positions actually arise in play, and so
//      which ones are worth spending an hour of search on.
//   2. Validation. Search each of those positions and keep every legal move
//      scoring within ACCEPTABLE_EVAL_LOSS of the best, and no worse than
//      EVAL_CUTOFF. Note *legal*, not *played*: the engine at depth 17 is a
//      stronger player than the humans in the file, so a move nobody happened to
//      try is kept or thrown out on exactly the same terms as one they did, and
//      the engine's own choice is always in the book. A book move is played with
//      no search at all, so an unsound entry is a game thrown away before the
//      engine starts thinking.
//
//      Whether that matters depends on how well travelled the position is. Over
//      the 100 shallowest positions, where master theory is dense, considering
//      every legal move rather than only the played ones added 5 moves to the
//      book. Over the 100 deepest, where a position may have been reached by
//      only four games, it added 26 - roughly one extra move for every three
//      positions. It is close to free either way, because the root search has
//      just looked at all of those moves and left them in the transposition
//      table. `--human-moves-only` restores the narrower behaviour.
//
// The output format is one line per position:
//
//     <p1>_<p2>_<p1k>_<p2k> <move> <move> ...
//
// with the four bitboards as 16 upper case hex digits and each move as 4 hex
// digits, high byte the start square and low byte the end square. That is what
// `Marcher_Engine_GUI/flask-server/GameHandler.py` reads, and it picks uniformly
// at random among the moves on the line - which is the whole point of keeping
// several, and also why every one of them has to be sound.
//
// What dominates the runtime, and what is done about it
// ----------------------------------------------------
// Validation is ~99% of the work: one search per position, and there are tens of
// thousands of positions. Three things make it affordable.
//
//   * Iterative deepening, like the real search. The old code called negmax once
//     at the full depth with no move ordering to work from, which is far slower
//     than reaching that depth one iteration at a time.
//   * A null window for the candidate moves. The question is never "what is this
//     move worth", it is only "is it within ACCEPTABLE_EVAL_LOSS of the best" -
//     a yes/no against a fixed threshold. Asking it with a null window instead of
//     a full one is several times cheaper, and the root search has just filled
//     the table with exactly the entries needed to answer it. Checked against a
//     full window search on 145 candidate moves: the two never disagreed.
//   * One evaler for the entire run. The transposition table is a process-wide
//     singleton, but `board_evaler_constructor` bumps its age and negmax refuses
//     cutoffs from entries of an older age, so building an evaler per position
//     threw away every cutoff the previous position had paid for. Book positions
//     are siblings and children of each other and share enormous subtrees. It is
//     sound here because the reason for the age check is stale repetition
//     context, and every position is searched as a fresh root with an empty draw
//     table - the context is identical for all of them. Measured over 100
//     positions: same depth reached (17.00 vs 16.97 ply), half the nodes, half
//     the time. `--fresh-table` turns it off.
//
// Together those take a position from the 1000 second budget this file used to
// ask for down to about one second, at the same depth 17.
//
// A caveat on reproducibility. Two runs do not have to produce the same book: a
// time limited search reaches its own depth, and ACCEPTABLE_EVAL_LOSS of 10 is
// small next to the spread of the engine's own opening evaluations, so moves
// sitting on the threshold move in and out between runs. Across two runs of the
// same 100 positions the engine's chosen move agreed 83% of the time but the
// full move list only 58%. If that matters, widen --eval-loss (which decides how
// much of the human repertoire survives) rather than expecting the boundary to
// be stable.
//
// Positions are validated shallowest first, so an interrupted run has finished
// the positions nearest the start of the game, which are the ones the book gets
// asked about most. `--out` is rewritten every CHECKPOINT_EVERY positions and
// `--resume` picks up from it. `--shard i/n` splits the work across n processes;
// the shard files concatenate into a complete book because the format is one
// independent line per position.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "board_search.c"

// ---- defaults (all overridable on the command line) ----
#define BOOK_DEPTH 15            // plies of each game to record
#define SEARCH_DEPTH 17          // depth cap for the validation search
#define SEARCH_SECONDS 3.0       // time budget per position, per stage
#define ACCEPTABLE_EVAL_LOSS 10  // how far below the best move a move may score
#define EVAL_CUTOFF -10          // ... and the worst score a book move may have
#define REQUIRED_OCCURENCES 3    // a position must be reached MORE times than this
#define CHECKPOINT_EVERY 200     // rewrite the output file this often

#define DEFAULT_PDN "src/python/nnue/data/CheckersGames.pdn"
#define DEFAULT_OUT "book_moves.txt"

#define MAX_HOPS 12              // longest capture chain we will reconstruct
#define MAX_LEGAL_MOVES 48       // generate_all_moves fills 96 ints, two per move
// buffer for a key: 4 x 16 hex digits + 3 underscores is 67 characters, plus the
// terminator. The 67 this file used to allocate was one byte short of that.
#define KEY_LENGTH 68

// game outcomes, player absolute. The values match the winning player number so
// `result == player` reads as "the side that moved went on to win". This
// orientation ("1-0" is PDN Black, which is our player 1) is not a guess: it is
// measured in src/python/nnue/pdn.py against the final material of 8,429
// decisive games.
#define RES_DRAW 0
#define RES_P1 1
#define RES_P2 2
#define RES_UNKNOWN -1

// -----------------------------------------------------------------------------
// configuration
// -----------------------------------------------------------------------------

static const char* cfg_pdn = DEFAULT_PDN;
static const char* cfg_out = DEFAULT_OUT;
static int    cfg_book_depth = BOOK_DEPTH;
static int    cfg_search_depth = SEARCH_DEPTH;
static double cfg_seconds = SEARCH_SECONDS;
static int    cfg_eval_loss = ACCEPTABLE_EVAL_LOSS;
static int    cfg_eval_cutoff = EVAL_CUTOFF;
static int    cfg_min_occurrence = REQUIRED_OCCURENCES;
static int    cfg_max_games = 0;      // 0 = every game in the file
static int    cfg_max_positions = 0;  // 0 = every position in the book
static int    cfg_shard = 0;
static int    cfg_shards = 1;
static int    cfg_resume = 0;
static int    cfg_verbose = 0;
static int    cfg_no_validate = 0;
static int    cfg_fresh_table = 0;   // one position's search may not use the last one's table
static int    cfg_human_moves_only = 0;  // consider only moves a master played
static int    cfg_jobs = 1;          // worker processes to split the run across
static int    cfg_quiet = 0;         // a worker: say nothing, the parent reports

// -----------------------------------------------------------------------------
// the book
// -----------------------------------------------------------------------------

typedef struct {
    int wins;    // games won by the player who made this move
    int losses;
    int ties;
} MoveOutcome;

typedef struct {
    short move;           // start square in the high byte, end square in the low byte
    MoveOutcome outcome;
    int eval;             // what the validation search thought of it
    int games;            // how many games played it from here
} MoveEntry;

typedef struct {
    long long p1, p2, p1k, p2k;
    int player;           // side to move, tracked through the replay rather than guessed
    int ply;              // shallowest ply this position was seen at
    int seq;              // insertion order, so the sort below is deterministic
    int occurrence;       // how many games reached it
    MoveEntry* moves;
    int moveCount;
    int moveCapacity;
    int validated;        // the search has passed over it (or it was resumed from disk)
    int ambiguous;        // also reached with the other side to move - see below
} PositionEntry;

static PositionEntry* positions = NULL;
static int positionsCount = 0;
static int positionsCapacity = 0;

// open addressed index over `positions`, keyed on the four bitboards. Slots hold
// (position index + 1) so that zero can mean empty. Without this the aggregation
// pass is a linear strcmp scan of the whole book per ply, which is quadratic in
// the number of positions and was taking longer than it took to read the file.
static int* index_table = NULL;
static int index_size = 0;

// Two games can reach the same piece configuration with opposite sides to move -
// rare (about one ply in two thousand) but real, because a quiet move and a
// capture both cost one ply while moving a piece a different distance, so the
// number of plies behind a position is not a function of the position.
//
// The file format has no room for it: the key is the placement alone, and the GUI
// looks a position up without saying whose move it is. A line holding the other
// side's moves would hand the engine a move for the enemy's pieces. So a position
// seen with both sides to move is dropped rather than guessed at, and only its
// first side's moves are ever recorded.
static int side_to_move_clashes = 0;

// Map of PDN squares to engine squares. PDN numbers the 32 dark squares 1..32
// with Black on 1..12; this engine numbers a full 8x8 matrix with player 1 at the
// bottom, so the mapping is a 180 degree rotation. Identical to PDN_TO_BIT in
// src/python/nnue/pdn.py, which checks it by replaying the whole file.
static const int pdn_to_engine[33] = {
    0,   // index 0 unused: PDN has no square 0, which is what makes the range
         // check in next_move_token able to reject result tokens like "0-1"
    62, 60, 58, 56,   // PDN 1-4
    55, 53, 51, 49,   // PDN 5-8
    46, 44, 42, 40,   // PDN 9-12
    39, 37, 35, 33,   // PDN 13-16
    30, 28, 26, 24,   // PDN 17-20
    23, 21, 19, 17,   // PDN 21-24
    14, 12, 10, 8,    // PDN 25-28
    7, 5, 3, 1        // PDN 29-32
};

// the starting position, in this engine's bitboards
#define START_P1 6172839697753047040LL
#define START_P2 11163050LL

// -----------------------------------------------------------------------------
// position index
// -----------------------------------------------------------------------------

static unsigned long long mix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static unsigned long long position_hash(long long p1, long long p2, long long p1k, long long p2k) {
    unsigned long long h = mix64((unsigned long long)p1);
    h = mix64(h ^ (unsigned long long)p2);
    h = mix64(h ^ (unsigned long long)p1k);
    return mix64(h ^ (unsigned long long)p2k);
}

static void index_insert(int idx) {
    unsigned long long h = position_hash(positions[idx].p1, positions[idx].p2,
                                         positions[idx].p1k, positions[idx].p2k);
    int slot = (int)(h & (unsigned long long)(index_size - 1));
    while (index_table[slot] != 0) {
        slot = (slot + 1) & (index_size - 1);
    }
    index_table[slot] = idx + 1;
}

// (re)build the index over every position currently in the book. Also called
// after the book is sorted, since the slots hold indices into the array.
static void index_rebuild(void) {
    int wanted = 1024;
    while (wanted < (positionsCount + 1) * 4) {
        wanted *= 2;
    }
    free(index_table);
    index_size = wanted;
    index_table = calloc(index_size, sizeof(int));
    if (index_table == NULL) {
        fprintf(stderr, "out of memory building the position index\n");
        exit(1);
    }
    for (int i = 0; i < positionsCount; i++) {
        index_insert(i);
    }
}

static int index_find(long long p1, long long p2, long long p1k, long long p2k) {
    unsigned long long h = position_hash(p1, p2, p1k, p2k);
    int slot = (int)(h & (unsigned long long)(index_size - 1));
    while (index_table[slot] != 0) {
        PositionEntry* e = &positions[index_table[slot] - 1];
        if (e->p1 == p1 && e->p2 == p2 && e->p1k == p1k && e->p2k == p2k) {
            return index_table[slot] - 1;
        }
        slot = (slot + 1) & (index_size - 1);
    }
    return -1;
}

// -----------------------------------------------------------------------------
// building the book
// -----------------------------------------------------------------------------

// The key as it appears in the output file. Written into `key`, which must have
// room for KEY_LENGTH bytes.
static void compute_position_key(char* key, long long p1, long long p2, long long p1k, long long p2k) {
    sprintf(key, "%016llX_%016llX_%016llX_%016llX",
            (unsigned long long)p1, (unsigned long long)p2,
            (unsigned long long)p1k, (unsigned long long)p2k);
}

// find the entry for a position, adding it if it is new. Returns its index, or
// -1 if the position is already in the book with the other side to move.
static int book_position(long long p1, long long p2, long long p1k, long long p2k, int player, int ply) {
    int idx = index_find(p1, p2, p1k, p2k);
    if (idx >= 0) {
        if (positions[idx].player != player) {
            positions[idx].ambiguous = 1;
            side_to_move_clashes++;
            return -1;
        }
        positions[idx].occurrence++;
        if (ply < positions[idx].ply) {
            positions[idx].ply = ply;
        }
        return idx;
    }

    if (positionsCount == positionsCapacity) {
        positionsCapacity = (positionsCapacity == 0) ? 4096 : positionsCapacity * 2;
        positions = realloc(positions, sizeof(PositionEntry) * positionsCapacity);
        if (positions == NULL) {
            fprintf(stderr, "out of memory growing the book\n");
            exit(1);
        }
    }

    idx = positionsCount++;
    PositionEntry* entry = &positions[idx];
    entry->p1 = p1;
    entry->p2 = p2;
    entry->p1k = p1k;
    entry->p2k = p2k;
    entry->player = player;
    entry->ply = ply;
    entry->seq = idx;
    entry->occurrence = 1;
    entry->moveCapacity = 4;
    entry->moveCount = 0;
    entry->validated = 0;
    entry->ambiguous = 0;
    entry->moves = malloc(sizeof(MoveEntry) * entry->moveCapacity);
    if (entry->moves == NULL) {
        fprintf(stderr, "out of memory growing the book\n");
        exit(1);
    }

    if (positionsCount * 4 >= index_size) {
        index_rebuild();
    } else {
        index_insert(idx);
    }
    return idx;
}

static MoveEntry* book_move(int idx, short move) {
    PositionEntry* entry = &positions[idx];
    for (int i = 0; i < entry->moveCount; i++) {
        if (entry->moves[i].move == move) {
            return &entry->moves[i];
        }
    }
    if (entry->moveCount == entry->moveCapacity) {
        entry->moveCapacity *= 2;
        entry->moves = realloc(entry->moves, sizeof(MoveEntry) * entry->moveCapacity);
        if (entry->moves == NULL) {
            fprintf(stderr, "out of memory growing a move list\n");
            exit(1);
        }
    }
    MoveEntry* m = &entry->moves[entry->moveCount++];
    m->move = move;
    m->outcome.wins = 0;
    m->outcome.losses = 0;
    m->outcome.ties = 0;
    m->eval = 0;
    m->games = 0;
    return m;
}

// record that `move` was played from this position by `player`, in a game that
// finished `result`. The outcome is stored from the point of view of the player
// who made the move, so a move played by the loser of a game is a loss whichever
// colour they were.
static void book_record(int idx, short move, int player, int result) {
    MoveEntry* m = book_move(idx, move);
    m->games++;
    if (result == player) {
        m->outcome.wins++;
    } else if (result == RES_DRAW) {
        m->outcome.ties++;
    } else if (result != RES_UNKNOWN) {
        m->outcome.losses++;
    }
}

// -----------------------------------------------------------------------------
// reading PDN
// -----------------------------------------------------------------------------

static char* read_whole_file(const char* path, long* size_out) {
    FILE* fp = fopen(path, "rb");
    if (fp == NULL) {
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* buffer = malloc((size_t)size + 1);
    if (buffer == NULL) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buffer, 1, (size_t)size, fp);
    buffer[got] = '\0';
    fclose(fp);
    if (size_out != NULL) {
        *size_out = (long)got;
    }
    return buffer;
}

static void blank_span(char* from, char* to) {
    while (from < to && *from != '\0') {
        *from++ = ' ';
    }
}

// Blank out everything in the movetext that is not a move: header lines,
// `{comments}`, `(variations)` and the drawn result token.
//
// `1/2-1/2` has to go before tokenising, not after: it contains `2-1`, which is a
// perfectly well formed move token and would otherwise be replayed as a move.
// `1-0` and `0-1` must NOT be stripped textually, because as substrings they
// occur inside real moves (`20-16` contains `0-1`); they are discarded by the
// 1..32 range check instead, since PDN has no square 0.
static void strip_pdn_noise(char* text, int* result) {
    char* p = text;

    // header lines, which is also where the result comes from
    while (*p != '\0') {
        char* line_end = strchr(p, '\n');
        if (line_end == NULL) {
            line_end = p + strlen(p);
        }
        char* scan = p;
        while (*scan == ' ' || *scan == '\t' || *scan == '\r') {
            scan++;
        }
        if (*scan == '[') {
            if (strncmp(scan, "[Result", 7) == 0) {
                char* quote = memchr(scan, '"', (size_t)(line_end - scan));
                if (quote != NULL) {
                    quote++;
                    char* end = memchr(quote, '"', (size_t)(line_end - quote));
                    if (end != NULL) {
                        size_t len = (size_t)(end - quote);
                        if (len == 3 && memcmp(quote, "1-0", 3) == 0) {
                            *result = RES_P1;
                        } else if (len == 3 && memcmp(quote, "0-1", 3) == 0) {
                            *result = RES_P2;
                        } else if (len == 7 && memcmp(quote, "1/2-1/2", 7) == 0) {
                            *result = RES_DRAW;
                        }
                    }
                }
            }
            blank_span(p, line_end);
        }
        if (*line_end == '\0') {
            break;
        }
        p = line_end + 1;
    }

    // comments and variations, which may span lines. An unclosed one is left
    // alone rather than swallowing the rest of the game.
    for (p = text; *p != '\0'; p++) {
        char close = (*p == '{') ? '}' : ((*p == '(') ? ')' : '\0');
        if (close != '\0') {
            char* end = strchr(p, close);
            if (end != NULL) {
                blank_span(p, end + 1);
            }
        }
    }

    // the drawn result token
    for (p = strstr(text, "1/2-1/2"); p != NULL; p = strstr(p, "1/2-1/2")) {
        blank_span(p, p + 7);
    }
}

// Pull the next move token out of `text`, in the shape `11-15`, `15x22` or
// `26x17x10x1`. Returns a pointer just past it, or NULL when the text is spent.
// A token with a square outside 1..32 is not a move (a result fragment, or a
// mangled comment) and is skipped.
static const char* next_move_token(const char* text, int* squares, int* count, int* is_capture) {
    while (*text != '\0') {
        if (*text < '0' || *text > '9') {
            text++;
            continue;
        }

        const char* scan = text;
        int n = 0;
        int capture = 0;
        int bad = 0;

        for (;;) {
            int value = 0;
            int digits = 0;
            while (*scan >= '0' && *scan <= '9') {
                value = value * 10 + (*scan - '0');
                scan++;
                digits++;
                if (digits > 3) {
                    bad = 1;
                    break;
                }
            }
            if (bad) {
                break;
            }
            if (value < 1 || value > 32) {
                bad = 1;
                break;
            }
            if (n < MAX_HOPS + 1) {
                squares[n++] = value;
            } else {
                bad = 1;
                break;
            }
            if (*scan == '-' || *scan == 'x' || *scan == 'X') {
                capture = capture || (*scan != '-');
                scan++;
                if (*scan < '0' || *scan > '9') {
                    bad = 1;
                    break;
                }
                continue;
            }
            break;
        }

        // skip the rest of whatever this was: a move number, a bad token, or a
        // token we just consumed
        while (*scan >= '0' && *scan <= '9') {
            scan++;
        }

        if (!bad && n >= 2) {
            *count = n;
            *is_capture = capture;
            return scan;
        }
        text = (scan > text) ? scan : text + 1;
    }
    return NULL;
}

// -----------------------------------------------------------------------------
// replaying a game onto the engine's board
// -----------------------------------------------------------------------------

// Is `end` one of the moves the piece on `start` can make?
static int hop_is_legal(long long p1, long long p2, long long p1k, long long p2k,
                        int start, int end, char* offsets, int only_jump) {
    int moves[8];
    int n = generate_moves(p1, p2, p1k, p2k, start, &moves[0], offsets, only_jump);
    if (n < 0) {
        // a jump exists but was not asked for, so this quiet move is not legal
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (moves[(i * 2) + 1] == end) {
            return 1;
        }
    }
    return 0;
}

// Depth first search for a capture chain from `start` to `end`, writing the hops
// it is made of into `hops`. Returns the number of hops, or -1 if the two squares
// are not connected by a chain of captures. Leaves the board as it found it.
//
// Needed because some sources compress `26x17x10x1` to `26x1`, dropping the
// intermediate landing squares. Slow, and allowed to be: it only runs when the
// listed pair is not a single legal hop.
static int find_capture_path(long long* p1, long long* p2, long long* p1k, long long* p2k,
                             int start, int end, int player, char* offsets,
                             short* hops, int depth) {
    if (depth >= MAX_HOPS) {
        return -1;
    }
    int moves[8];
    int n = generate_moves(*p1, *p2, *p1k, *p2k, start, &moves[0], offsets, True);
    for (int i = 0; i < n; i++) {
        int landing = moves[(i * 2) + 1];
        int piece_type = get_piece_at_location(*p1, *p2, *p1k, *p2k, start);
        int jumped = update_board(p1, p2, p1k, p2k, start, landing);
        hops[depth] = (short)((start << 8) | landing);

        int found = -1;
        if (landing == end) {
            found = depth + 1;
        } else if (get_next_board_state(*p1, *p2, *p1k, *p2k, start, landing,
                                        player, piece_type, offsets) == player) {
            found = find_capture_path(p1, p2, p1k, p2k, landing, end, player, offsets, hops, depth + 1);
        }

        undo_board_update(p1, p2, p1k, p2k, start, landing, jumped, piece_type);
        if (found >= 0) {
            return found;
        }
    }
    return -1;
}

// one recorded ply: the position before the move, and the move's first hop
typedef struct {
    long long p1, p2, p1k, p2k;
    int player;
    short move;
} ReplayPly;

// Replay a game's movetext, filling `plies` with the first `max_plies` moves.
// Returns how many replayed legally; a game that goes wrong simply stops there
// and contributes the plies before the problem.
//
// Every hop is checked against the move generator, which is what keeps the book
// honest. This file contains 19th century games played under the huffing rule,
// where declining a capture forfeited the piece rather than being illegal, plus
// ordinary transcription errors. Replaying those blindly does not just lose the
// game in question - it corrupts the board and every position after it in that
// game gets recorded as if it were real. The forced capture rule needs no special
// case here: generate_all_moves returns only captures when one is available, so a
// quiet move recorded in its place is simply not in the list.
static int replay_game(const char* movetext, ReplayPly* plies, int max_plies, char* offsets) {
    long long p1 = START_P1, p2 = START_P2, p1k = 0, p2k = 0;
    int player = 1;
    int ply = 0;

    struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);
    const char* cursor = movetext;
    int squares[MAX_HOPS + 2];
    int count = 0;
    int is_capture = 0;

    while (ply < max_plies) {
        cursor = next_move_token(cursor, squares, &count, &is_capture);
        if (cursor == NULL) {
            break;
        }

        int from = pdn_to_engine[squares[0]];
        int piece = get_piece_at_location(p1, p2, p1k, p2k, from);
        if (piece != player && piece != player + 2) {
            break;  // not this player's piece, or no piece at all
        }

        int legal_moves[96];
        int jump = 0;
        int num_legal = generate_all_moves(p1, p2, p1k, p2k, player, &legal_moves[0],
                                           piece_loc, offsets, &jump);
        if (num_legal == 0) {
            break;
        }
        if (is_capture != jump) {
            break;  // a capture recorded with none available, or a huffed move
        }

        // the position before the move is the one the book records, so keep it
        // before the hops start landing on the board
        long long before_p1 = p1, before_p2 = p2, before_p1k = p1k, before_p2k = p2k;
        short first_hop = 0;
        int hop_count = 0;
        int next_player = player;
        int ok = 1;

        // expand the token into single hops, playing each one as it is resolved so
        // that the next pair is checked against the board it is really played on
        for (int i = 0; ok && i + 1 < count; i++) {
            int a = pdn_to_engine[squares[i]];
            int b = pdn_to_engine[squares[i + 1]];
            short path[MAX_HOPS];
            int path_length;

            if (hop_is_legal(p1, p2, p1k, p2k, a, b, offsets, is_capture)) {
                path[0] = (short)((a << 8) | b);
                path_length = 1;
            } else if (is_capture) {
                path_length = find_capture_path(&p1, &p2, &p1k, &p2k, a, b, player, offsets, path, 0);
            } else {
                path_length = -1;
            }
            if (path_length < 0 || hop_count + path_length > MAX_HOPS) {
                ok = 0;
                break;
            }

            for (int h = 0; h < path_length; h++) {
                int start = (path[h] >> 8) & 0xFF;
                int end = path[h] & 0xFF;
                int piece_type = get_piece_at_location(p1, p2, p1k, p2k, start);
                update_board(&p1, &p2, &p1k, &p2k, start, end);
                update_piece_locations(start, end, piece_loc);
                next_player = get_next_board_state(p1, p2, p1k, p2k, start, end,
                                                   player, piece_type, offsets);
                if (hop_count == 0) {
                    first_hop = path[h];
                }
                hop_count++;
            }
        }

        // the chain has to be finished: a record that stops halfway through one
        // leaves a board no legal game could be in
        if (!ok || hop_count == 0 || next_player == player) {
            break;
        }

        plies[ply].p1 = before_p1;
        plies[ply].p2 = before_p2;
        plies[ply].p1k = before_p1k;
        plies[ply].p2k = before_p2k;
        plies[ply].player = player;
        plies[ply].move = first_hop;

        player = next_player;
        ply++;
    }

    free(piece_loc);
    return ply;
}

// -----------------------------------------------------------------------------
// the validation search
// -----------------------------------------------------------------------------

// One evaler for the whole run - see the note at the top of the file on why the
// transposition table only pays off if it is not reconstructed per position.
static struct board_evaler* EVALER = NULL;
static long long total_nodes = 0;

static void book_search_init(void) {
    EVALER = board_evaler_constructor(START_P1, START_P2, cfg_search_depth, cfg_seconds, clock());
#if USE_ENDGAME_DB
    {
        const char* db_dir = getenv("CHECKERS_DB_DIR");
        endgame_db_init(db_dir != NULL ? db_dir : "db", DB_MAX_TOTAL);
    }
#endif
}

// Point the shared evaler at a new root. Everything root relative is reset; the
// transposition table and the killer moves are deliberately not, since they are
// the reason one evaler is faster than many. Killers are only a move ordering
// hint and order_moves can only reorder moves that are already in the list, so
// carrying them between sibling positions is free and usually helps.
static void book_search_reset(long long p1, long long p2, long long p1k, long long p2k) {
    if (cfg_fresh_table) {
        // Ageing the table makes negmax ignore every entry the previous positions
        // wrote, which is what the engine does between the moves of a real game.
        // Measured over 100 positions it costs twice the time and twice the nodes
        // to arrive at the same depth, so it is not the default - see --help.
        EVALER->hash_table->age++;
    }
    EVALER->nodes = 0;
    EVALER->avg_depth = 0;
    EVALER->extended_depth = 0;
    EVALER->initial_piece_count_p1 = get_bits_set(p1 | p1k);
    EVALER->initial_piece_count_p2 = get_bits_set(p2 | p2k);
    EVALER->start_time = clock();
    EVALER->time_limit = cfg_seconds;
    EVALER->search_results->num_moves = 0;
    EVALER->search_results->best_move = NO_MOVE;
    EVALER->search_results->best_eval = 0;
    memset(EVALER->draw_table->table, 0,
           (size_t)EVALER->draw_table->size * sizeof(struct draw_table_entry));
#if USE_NNUE
    nnue_stack_reset(EVALER->nnue, p1, p2, p1k, p2k);
#endif
}

static double book_search_elapsed(void) {
    return ((double)(clock() - EVALER->start_time)) / CLOCKS_PER_SEC;
}

// The position's hash as the search wants it: piece placement, plus the side to
// move. Leaving the side out (as this file used to) makes the two sides of the
// same placement collide in the transposition table.
static unsigned long long book_hash(long long p1, long long p2, long long p1k, long long p2k, int player) {
    unsigned long long hash = get_hash(p1, p2, p1k, p2k, EVALER->hash_table);
    if (player == 2) {
        hash ^= EVALER->hash_table->piece_hash_diff[SIDE_KEY_INDEX];
    }
    return hash;
}

// Iterative deepening from the root, returning the best evaluation and, through
// the out parameters, the engine's preferred move and the depth that produced it.
static int book_root_search(long long p1, long long p2, long long p1k, long long p2k, int player,
                            struct set* piece_loc, unsigned long long hash,
                            short* best_move_out, int* depth_out) {
    int best_eval = 0;
    short best_move = NO_MOVE;
    int completed = 0;

    for (int depth = 1; depth <= cfg_search_depth; depth++) {
        EVALER->search_depth = depth;
        EVALER->max_depth = min(max(depth + 10, 5), cfg_search_depth);

        int eval = negmax(&p1, &p2, &p1k, &p2k, player, piece_loc, depth,
                          -INFINITY, INFINITY, EVALER, hash, 0, -1);

        // out of time: the last completed iteration stands
        if (eval == INFINITY || eval == -INFINITY) {
            break;
        }

        best_eval = eval;
        best_move = EVALER->search_results->best_move;
        completed = depth;

        if (book_search_elapsed() > EVALER->time_limit) {
            break;
        }
        // a proven result that fits inside the depth already searched: no shorter
        // win can be hiding past the horizon, so there is nothing left to find
        if ((eval > WIN_MIN || eval < -WIN_MIN) && win_distance(eval) <= depth) {
            break;
        }
        if (EVALER->search_results->num_moves <= 1) {
            break;
        }
    }

    *best_move_out = best_move;
    *depth_out = completed;
    return best_eval;
}

// Score one candidate move from the root, through the window [alpha, beta].
// Mirrors what negmax does to reach a child of the root, which is the only way to
// get a score on the same scale as the root search's.
//
// Callers pass a null window around the threshold, because the search is never
// asked what a move is worth, only which side of the line it falls on - and that
// is much cheaper to answer.
static int book_score_move(long long p1, long long p2, long long p1k, long long p2k, int player,
                           struct set* piece_loc, unsigned long long hash,
                           short move, int depth, int alpha, int beta, int* timed_out) {
    int move_start = (move >> 8) & 0xFF;
    int move_end = move & 0xFF;

    unsigned long long next_hash = update_hash(p1, p2, p1k, p2k, move_start, move_end, hash, EVALER);
    int initial_piece_type = get_piece_at_location(p1, p2, p1k, p2k, move_start);
    int jumped_piece_type = update_board(&p1, &p2, &p1k, &p2k, move_start, move_end);
    update_piece_locations(move_start, move_end, piece_loc);
    int player_next = get_next_board_state(p1, p2, p1k, p2k, move_start, move_end,
                                           player, initial_piece_type, EVALER->piece_offsets);
#if USE_NNUE && NNUE_INCREMENTAL
    nnue_stack_push(EVALER->nnue, 0, initial_piece_type, move_start, move_end, jumped_piece_type);
#endif

    int next_forced = -1;
    if (player_next == player) {
#if USE_FORCED_CONTINUATION
        next_forced = move_end;
        next_hash ^= EVALER->hash_table->piece_hash_diff[FORCED_KEY_OFFSET + move_end];
#endif
    } else {
        next_hash ^= EVALER->hash_table->piece_hash_diff[SIDE_KEY_INDEX];
    }
    add_draw_entry(EVALER->draw_table, next_hash);

    int flip = (player != player_next) ? -1 : 1;
    int next_alpha = (flip == 1) ? alpha : -beta;
    int next_beta = (flip == 1) ? beta : -alpha;

    // a multi jump is one move, so continuing the chain must not consume a ply
    int child_depth = depth - 1;
#if USE_FREE_JUMP_CHAIN
    if (player_next == player) {
        child_depth = depth;
    }
#endif

    int eval = negmax(&p1, &p2, &p1k, &p2k, player_next, piece_loc, child_depth,
                      next_alpha, next_beta, EVALER, next_hash, 1, next_forced);

    int score;
    if (eval == INFINITY || eval == -INFINITY) {
        // no answer in the time available. Reporting the top of the window keeps
        // the move: a human move already in the book is not thrown out on the
        // strength of a search that never finished.
        *timed_out = 1;
        score = beta;
    } else {
        score = eval * flip;
    }

    undo_piece_locations_update(move_start, move_end, piece_loc);
    undo_board_update(&p1, &p2, &p1k, &p2k, move_start, move_end, jumped_piece_type, initial_piece_type);
    remove_draw_entry(EVALER->draw_table, next_hash);
    return score;
}

// counters for the closing summary
static long long stat_depth_sum = 0;   // depth actually reached, summed
static int stat_depth_n = 0;
static int stat_moves_dropped = 0;      // human moves the search threw out
static int stat_moves_engine_only = 0;  // kept moves no master played
static int stat_candidates = 0;         // moves put to the search
static int stat_moves_timed_out = 0;
static int stat_best_added = 0;
static int stat_forced = 0;

// Search a position and reduce its move list to the moves that survive: those
// scoring within ACCEPTABLE_EVAL_LOSS of the best move and no worse than
// EVAL_CUTOFF. Both conditions are one test against one threshold, since
// `x >= a && x >= b` is `x >= max(a, b)`.
static void validate_book_position(int idx) {
    PositionEntry* pos = &positions[idx];
    long long p1 = pos->p1, p2 = pos->p2, p1k = pos->p1k, p2k = pos->p2k;
    int player = pos->player;

    struct set* piece_loc = get_piece_locations(p1, p2, p1k, p2k);
    book_search_reset(p1, p2, p1k, p2k);
    unsigned long long hash = book_hash(p1, p2, p1k, p2k, player);

    // a position with one legal move does not need an opinion
    int legal[96];
    int jump = 0;
    int num_legal = generate_all_moves(p1, p2, p1k, p2k, player, &legal[0],
                                       piece_loc, EVALER->piece_offsets, &jump);
    if (num_legal == 1) {
        short only = (short)((legal[0] << 8) | legal[1]);
        MoveEntry* m = book_move(idx, only);
        MoveEntry kept = *m;
        kept.eval = 0;
        pos->moves[0] = kept;
        pos->moveCount = 1;
        pos->validated = 1;
        stat_forced++;
        total_nodes += EVALER->nodes;
        free(piece_loc);
        return;
    }

    short engine_best = NO_MOVE;
    int depth_reached = 0;
    int best_eval = book_root_search(p1, p2, p1k, p2k, player, piece_loc, hash,
                                     &engine_best, &depth_reached);

    if (depth_reached == 0) {
        // not even one iteration finished, so there is no opinion to act on
        pos->validated = 1;
        total_nodes += EVALER->nodes;
        free(piece_loc);
        return;
    }

    stat_depth_sum += depth_reached;
    stat_depth_n++;

    int threshold = max(best_eval - cfg_eval_loss, cfg_eval_cutoff);

    // the candidates are scored at the depth the root reached, which is where
    // best_eval came from, so the two are comparable
    EVALER->search_depth = depth_reached;
    EVALER->max_depth = min(max(depth_reached + 10, 5), cfg_search_depth);
    EVALER->time_limit = book_search_elapsed() + cfg_seconds;

    // Every legal move is a candidate, not only the ones a master played.
    //
    // The game collection and the search know different things, and this is the
    // line between them. Which positions are worth having in the book is
    // something 22,000 master games know and a search cannot tell you - it is a
    // record of what actually gets played. Which move to make in one of them is
    // the opposite: the engine searching that position to depth 17 is the
    // stronger player, and the games are a 19th and 20th century record of humans
    // at the board. So the collection chooses the positions and the engine
    // chooses the moves; a move nobody happened to try is kept or thrown out on
    // exactly the same terms as one they did.
    //
    // The extra candidates are nearly free: the root search just finished looking
    // at every one of these moves, so the transposition table already holds them
    // at this depth and most of the tests below return without a search.
    MoveEntry candidates[MAX_LEGAL_MOVES];
    int candidate_count = 0;
    if (cfg_human_moves_only) {
        for (int i = 0; i < pos->moveCount && candidate_count < MAX_LEGAL_MOVES; i++) {
            candidates[candidate_count++] = pos->moves[i];
        }
    } else {
        for (int i = 0; i < num_legal && candidate_count < MAX_LEGAL_MOVES; i++) {
            MoveEntry entry;
            entry.move = (short)((legal[i * 2] << 8) | legal[(i * 2) + 1]);
            entry.eval = 0;
            entry.games = 0;
            entry.outcome.wins = 0;
            entry.outcome.losses = 0;
            entry.outcome.ties = 0;
            // carry over the human record, where there is one
            for (int j = 0; j < pos->moveCount; j++) {
                if (pos->moves[j].move == entry.move) {
                    entry = pos->moves[j];
                    break;
                }
            }
            candidates[candidate_count++] = entry;
        }
    }

    if (pos->moveCapacity < candidate_count) {
        pos->moves = realloc(pos->moves, sizeof(MoveEntry) * candidate_count);
        if (pos->moves == NULL) {
            fprintf(stderr, "out of memory growing a move list\n");
            exit(1);
        }
        pos->moveCapacity = candidate_count;
    }

    stat_candidates += candidate_count;

    int kept_count = 0;
    for (int i = 0; i < candidate_count; i++) {
        MoveEntry candidate = candidates[i];
        int score;

        if (candidate.move == engine_best) {
            score = best_eval;
        } else {
            int timed_out = 0;
            score = book_score_move(p1, p2, p1k, p2k, player, piece_loc, hash,
                                    candidate.move, depth_reached,
                                    threshold - 1, threshold, &timed_out);
            if (timed_out) {
                stat_moves_timed_out++;
            }
        }

        candidate.eval = score;
        if (score >= threshold) {
            if (candidate.games == 0) {
                stat_moves_engine_only++;
            }
            pos->moves[kept_count++] = candidate;
        } else {
            if (candidate.games > 0) {
                stat_moves_dropped++;
            }
            if (cfg_verbose && candidate.games > 0) {
                printf("  drop %02d-%02d  eval %d, best %d, %d human games\n",
                       (candidate.move >> 8) & 0xFF, candidate.move & 0xFF,
                       score, best_eval, candidate.games);
            }
        }
    }
    pos->moveCount = kept_count;

    // The engine's own choice always belongs in the book. This is also what keeps
    // a line from emptying out: when the best move itself is below EVAL_CUTOFF
    // every human move fails the threshold, and the position is left holding the
    // one move the engine would have played anyway.
    int have_best = 0;
    for (int i = 0; i < pos->moveCount; i++) {
        if (pos->moves[i].move == engine_best) {
            have_best = 1;
            break;
        }
    }
    if (!have_best && engine_best != NO_MOVE) {
        MoveEntry* m = book_move(idx, engine_best);
        m->eval = best_eval;
        stat_best_added++;
    }

    // Order: the engine's move first, then the moves masters actually played,
    // then the ones only the search likes. The format says nothing about order
    // and the GUI picks at random, so this is for a reader - and for any future
    // reader that takes the first move rather than choosing among them, which
    // should get the best corroborated ones first.
    for (int i = 1; i < pos->moveCount; i++) {
        if (pos->moves[i].move == engine_best) {
            MoveEntry tmp = pos->moves[0];
            pos->moves[0] = pos->moves[i];
            pos->moves[i] = tmp;
            break;
        }
    }
    for (int i = 1; i < pos->moveCount; i++) {
        if (pos->moves[i].games > 0) {
            MoveEntry played = pos->moves[i];
            int j = i;
            while (j > 1 && pos->moves[j - 1].games == 0) {
                pos->moves[j] = pos->moves[j - 1];
                j--;
            }
            pos->moves[j] = played;
        }
    }

    pos->validated = 1;
    total_nodes += EVALER->nodes;  // reset per position by book_search_reset
    free(piece_loc);
}

// -----------------------------------------------------------------------------
// output
// -----------------------------------------------------------------------------

// Write every validated position. Unvalidated ones are left out on purpose: the
// file is rewritten as a checkpoint while the run is still going, and an entry
// in it means "the engine has passed on these moves", which is also what makes
// --resume able to trust it.
static int write_book_to_file(const char* filename, int* moves_out) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: could not open %s for writing\n", filename);
        return 0;
    }
    char key[KEY_LENGTH];
    int written = 0;
    int moves = 0;
    for (int i = 0; i < positionsCount; i++) {
        if (!positions[i].validated || positions[i].moveCount == 0) {
            continue;
        }
        compute_position_key(key, positions[i].p1, positions[i].p2, positions[i].p1k, positions[i].p2k);
        fprintf(fp, "%s", key);
        for (int j = 0; j < positions[i].moveCount; j++) {
            fprintf(fp, " %04X", (unsigned short)positions[i].moves[j].move);
        }
        fprintf(fp, "\n");
        written++;
        moves += positions[i].moveCount;
    }
    fclose(fp);
    if (moves_out != NULL) {
        *moves_out = moves;
    }
    return written;
}

// Reload a previous run's output: any position it already holds is taken as
// validated and skipped. Positions in the file that are not in the book being
// built (a different --min-occurrence, say) are ignored.
static int load_book_file(const char* filename) {
    char* text = read_whole_file(filename, NULL);
    if (text == NULL) {
        return 0;
    }

    int loaded = 0;
    char* line = text;
    while (line != NULL && *line != '\0') {
        char* end = strchr(line, '\n');
        if (end != NULL) {
            *end = '\0';
        }

        unsigned long long p1, p2, p1k, p2k;
        if (sscanf(line, "%16llX_%16llX_%16llX_%16llX", &p1, &p2, &p1k, &p2k) == 4) {
            int idx = index_find((long long)p1, (long long)p2, (long long)p1k, (long long)p2k);
            if (idx >= 0) {
                positions[idx].moveCount = 0;
                const char* scan = line + (KEY_LENGTH - 1);
                unsigned int move;
                while (*scan == ' ' && sscanf(scan, " %4X", &move) == 1) {
                    book_move(idx, (short)move);
                    scan += 5;
                }
                if (positions[idx].moveCount > 0) {
                    positions[idx].validated = 1;
                    loaded++;
                }
            }
        }

        line = (end != NULL) ? end + 1 : NULL;
    }
    free(text);
    return loaded;
}

// -----------------------------------------------------------------------------
// driver
// -----------------------------------------------------------------------------

// shallowest first, so an interrupted run has finished the positions the book is
// asked about most often, and consecutive searches are near neighbours in the
// opening tree - which is what the shared transposition table feeds on
static int compare_positions(const void* a, const void* b) {
    const PositionEntry* x = (const PositionEntry*)a;
    const PositionEntry* y = (const PositionEntry*)b;
    if (x->ply != y->ply) {
        return x->ply - y->ply;
    }
    return x->seq - y->seq;
}

static void build_book_from_pdn(void) {
    long size = 0;
    char* text = read_whole_file(cfg_pdn, &size);
    if (text == NULL) {
        fprintf(stderr, "Error: could not open %s\n", cfg_pdn);
        exit(1);
    }
    printf("reading %s (%.1f MB)\n", cfg_pdn, (double)size / (1024.0 * 1024.0));

    char* offsets = compute_offsets();
    ReplayPly* plies = malloc(sizeof(ReplayPly) * (size_t)cfg_book_depth);
    if (offsets == NULL || plies == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    index_rebuild();

    int games = 0;
    int games_with_result = 0;
    long long recorded = 0;
    int short_games = 0;

    // games start at a [Event tag at the beginning of a line. Splitting on blank
    // lines instead (as this used to) breaks any game whose movetext contains one,
    // and silently drops the last game in the file.
    char* block = strstr(text, "[Event ");
    while (block != NULL && (cfg_max_games == 0 || games < cfg_max_games)) {
        char* next = strstr(block + 1, "\n[Event ");
        if (next != NULL) {
            *next = '\0';  // terminate this game at the newline
            next++;        // and start the next one at its '['
        }

        int result = RES_UNKNOWN;
        strip_pdn_noise(block, &result);
        int count = replay_game(block, plies, cfg_book_depth, offsets);

        games++;
        if (result != RES_UNKNOWN) {
            games_with_result++;
        }
        if (count < cfg_book_depth) {
            short_games++;
        }
        for (int i = 0; i < count; i++) {
            int idx = book_position(plies[i].p1, plies[i].p2, plies[i].p1k, plies[i].p2k,
                                    plies[i].player, i);
            if (idx >= 0) {
                book_record(idx, plies[i].move, plies[i].player, result);
                recorded++;
            }
        }

        block = next;
    }

    printf("%d games, %lld plies recorded, %d distinct positions\n",
           games, recorded, positionsCount);
    printf("%d games ended or stopped replaying before ply %d, %d had no readable result\n",
           short_games, cfg_book_depth, games - games_with_result);
    free(plies);
    free(offsets);
    free(text);
}

static void filter_by_occurrence(void) {
    int kept = 0;
    int ambiguous = 0;
    for (int i = 0; i < positionsCount; i++) {
        if (positions[i].ambiguous) {
            ambiguous++;
            free(positions[i].moves);
        } else if (positions[i].occurrence > cfg_min_occurrence) {
            positions[kept++] = positions[i];
        } else {
            free(positions[i].moves);
        }
    }
    printf("%d positions reached more than %d times (%d dropped)\n",
           kept, cfg_min_occurrence, positionsCount - kept);
    if (side_to_move_clashes > 0) {
        printf("%d positions dropped: also reached with the other side to move, "
               "which the key cannot distinguish (%d plies)\n",
               ambiguous, side_to_move_clashes);
    }
    positionsCount = kept;
}

// -----------------------------------------------------------------------------
// running the validation across several processes
// -----------------------------------------------------------------------------
//
// Processes, not threads. The search keeps its state in process globals - the
// transposition table is a singleton in hash_table.c, the endgame database
// slices and GAME_HISTORY are file scope arrays - and a hash table entry is 24
// bytes written without a lock, so two threads sharing one would eventually read
// an entry whose hash matched but whose move and score came from a different
// position. Making the engine thread safe is a much larger change than this file
// should be making. Separate processes share nothing but the read-only database
// files, so the split is safe as it stands.
//
// Each worker re-reads the PDN and rebuilds the position list, which costs a
// seventh of a second and means no data has to be handed between processes: the
// aggregation is deterministic, so every worker computes the same list in the
// same order and `--shard i/n` selects the same block in each. The cost is one
// transposition table per worker, about 100 MB.

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

#define MAX_JOBS 64
#define MAX_CHILD_ARGS 64

static int cpu_count(void) {
#ifdef _WIN32
    const char* n = getenv("NUMBER_OF_PROCESSORS");
    int count = (n != NULL) ? atoi(n) : 0;
    return (count > 0) ? count : 1;
#else
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    return (count > 0) ? (int)count : 1;
#endif
}

// where a worker writes its share of the book. Kept after the merge rather than
// cleaned up, because it is also what --resume reads to skip work already done.
static void part_path(char* buffer, size_t size, int shard) {
    snprintf(buffer, size, "%s.part%d", cfg_out, shard);
}

#ifdef _WIN32
// The Windows spawn functions build a command line by joining argv with spaces,
// so an argument containing one arrives at the child split in two - and this
// repository lives under "Checkers Engine". Anything with a space is quoted
// here; the returned string is always heap allocated so the caller can free it
// unconditionally.
static char* quote_arg(const char* arg) {
    size_t length = strlen(arg);
    if (strchr(arg, ' ') == NULL && strchr(arg, '\t') == NULL) {
        char* plain = malloc(length + 1);
        if (plain != NULL) {
            memcpy(plain, arg, length + 1);
        }
        return plain;
    }
    char* quoted = malloc(length + 3);
    if (quoted != NULL) {
        quoted[0] = '"';
        memcpy(quoted + 1, arg, length);
        quoted[length + 1] = '"';
        quoted[length + 2] = '\0';
    }
    return quoted;
}
#endif

// Start one worker. Returns a handle to wait on, or -1.
static long long spawn_worker(char** child_argv) {
#ifdef _WIN32
    char* quoted[MAX_CHILD_ARGS + 1];
    int n = 0;
    while (child_argv[n] != NULL && n < MAX_CHILD_ARGS) {
        quoted[n] = quote_arg(child_argv[n]);
        if (quoted[n] == NULL) {
            while (n-- > 0) {
                free(quoted[n]);
            }
            return -1;
        }
        n++;
    }
    quoted[n] = NULL;
    intptr_t handle = _spawnvp(_P_NOWAIT, child_argv[0], (const char* const*)quoted);
    for (int i = 0; i < n; i++) {
        free(quoted[i]);
    }
    return (long long)handle;
#else
    pid_t pid = fork();
    if (pid == 0) {
        execvp(child_argv[0], child_argv);
        _exit(127);
    }
    return (long long)pid;
#endif
}

// Wait for one worker. Returns its exit status, or -1 if it could not be waited on.
static int wait_worker(long long handle) {
#ifdef _WIN32
    int status = -1;
    if (_cwait(&status, (intptr_t)handle, 0) == -1) {
        return -1;
    }
    return status;
#else
    int status = 0;
    if (waitpid((pid_t)handle, &status, 0) < 0) {
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

// Concatenate the workers' part files into cfg_out. The shards are disjoint
// blocks of one sorted list, and the format is one independent line per
// position, so appending them in order is all a merge amounts to.
static int merge_parts(int jobs, int* positions_out, int* moves_out) {
    FILE* out = fopen(cfg_out, "w");
    if (out == NULL) {
        fprintf(stderr, "Error: could not open %s for writing\n", cfg_out);
        return 0;
    }
    int lines = 0;
    int moves = 0;
    for (int j = 0; j < jobs; j++) {
        char path[1024];
        part_path(path, sizeof(path), j);
        char* text = read_whole_file(path, NULL);
        if (text == NULL) {
            fprintf(stderr, "Error: worker %d wrote no %s\n", j, path);
            fclose(out);
            return 0;
        }
        for (char* line = text; *line != '\0'; ) {
            char* end = strchr(line, '\n');
            if (end != NULL) {
                *end = '\0';
            }
            if (*line != '\0') {
                fprintf(out, "%s\n", line);
                lines++;
                for (const char* c = line; *c != '\0'; c++) {
                    moves += (*c == ' ');
                }
            }
            if (end == NULL) {
                break;
            }
            line = end + 1;
        }
        free(text);
    }
    fclose(out);
    *positions_out = lines;
    *moves_out = moves;
    return 1;
}

// Split the run across `cfg_jobs` copies of this program and merge the results.
static int run_workers(int argc, char* argv[]) {
    int jobs = (cfg_jobs > 0) ? cfg_jobs : cpu_count();
    if (jobs > MAX_JOBS) {
        jobs = MAX_JOBS;
    }

    // everything this process was given, minus --jobs, is passed straight
    // through, so the workers cannot drift from the options asked for here
    char* base[MAX_CHILD_ARGS];
    int base_count = 0;
    base[base_count++] = argv[0];
    for (int i = 1; i < argc && base_count < MAX_CHILD_ARGS - 6; i++) {
        if (strcmp(argv[i], "--jobs") == 0) {
            i++;
            continue;
        }
        base[base_count++] = argv[i];
    }

    printf("%d worker processes, about %d MB of hash table each\n",
           jobs, (int)((1ll << 22) * sizeof(struct hash_table_entry) / (1024 * 1024)));
    printf("progress below is worker 0's share; the others run quiet\n\n");
    fflush(stdout);

    long long handles[MAX_JOBS];
    char shard_text[MAX_JOBS][32];
    char part_text[MAX_JOBS][1024];
    int started = 0;

    for (int j = 0; j < jobs; j++) {
        snprintf(shard_text[j], sizeof(shard_text[j]), "%d/%d", j, jobs);
        part_path(part_text[j], sizeof(part_text[j]), j);

        char* child[MAX_CHILD_ARGS + 1];
        int n = 0;
        for (int i = 0; i < base_count; i++) {
            child[n++] = base[i];
        }
        child[n++] = "--shard";
        child[n++] = shard_text[j];
        child[n++] = "--out";
        child[n++] = part_text[j];
        if (j > 0) {
            child[n++] = "--quiet";
        }
        child[n] = NULL;

        handles[j] = spawn_worker(child);
        if (handles[j] == -1) {
            fprintf(stderr, "could not start worker %d\n", j);
            break;
        }
        started++;
    }

    int failed = 0;
    for (int j = 0; j < started; j++) {
        int status = wait_worker(handles[j]);
        if (status != 0) {
            fprintf(stderr, "worker %d exited with status %d\n", j, status);
            failed++;
        }
    }
    if (started < jobs || failed > 0) {
        fprintf(stderr, "%d of %d workers failed; their part files are left in "
                        "place, so --resume will pick up where they stopped\n",
                (jobs - started) + failed, jobs);
        return 1;
    }

    int merged_positions = 0;
    int merged_moves = 0;
    if (!merge_parts(jobs, &merged_positions, &merged_moves)) {
        return 1;
    }
    printf("\n%s: %d positions, %d moves (%.2f per position)\n",
           cfg_out, merged_positions, merged_moves,
           (merged_positions > 0) ? (double)merged_moves / merged_positions : 0.0);
    printf("worker output kept as %s.part0 .. .part%d (delete them, or keep them "
           "for --resume)\n", cfg_out, jobs - 1);
    return 0;
}

static void print_usage(const char* program) {
    printf("usage: %s [options]\n\n", program);
    printf("  --pdn PATH         game collection to read (default: %s)\n", DEFAULT_PDN);
    printf("  --out PATH         book to write (default: %s)\n", DEFAULT_OUT);
    printf("  --book-depth N     plies of each game to record (default: %d)\n", BOOK_DEPTH);
    printf("  --depth N          depth cap for the validation search (default: %d)\n", SEARCH_DEPTH);
    printf("  --time SECONDS     search budget per position, per stage (default: %.1f)\n", SEARCH_SECONDS);
    printf("  --eval-loss N      how far below the best move a book move may score (default: %d)\n", ACCEPTABLE_EVAL_LOSS);
    printf("  --eval-cutoff N    worst score a book move may have (default: %d)\n", EVAL_CUTOFF);
    printf("  --min-occurrence N a position must be reached MORE times than this (default: %d)\n", REQUIRED_OCCURENCES);
    printf("  --jobs N           split the run across N worker processes and merge\n");
    printf("                     their output (0 = one per core, this machine has %d).\n", cpu_count());
    printf("                     Each worker holds its own hash table, so N is\n");
    printf("                     limited by memory as much as by cores\n");
    printf("  --shard I/N        validate only shard I of N, writing just that share.\n");
    printf("                     --jobs does this for you; use it by hand only to\n");
    printf("                     spread a run over more than one machine\n");
    printf("  --resume           keep the positions already validated in --out\n");
    printf("  --human-moves-only only consider moves a master actually played, rather\n");
    printf("                     than every legal move the engine approves of\n");
    printf("  --fresh-table      do not let a position's search reuse the previous\n");
    printf("                     position's transposition table. Measured over 100\n");
    printf("                     positions this costs 2x the time and 2x the nodes to\n");
    printf("                     reach the same depth, so it is off\n");
    printf("  --games N          read only the first N games (for a quick trial)\n");
    printf("  --limit N          validate only the first N positions (for a quick trial)\n");
    printf("  --no-validate      aggregate only, skip the search entirely\n");
    printf("  --verbose          print every move the search throws out\n");
    printf("\nthe search reads the endgame database from $CHECKERS_DB_DIR (default: db)\n");
}

static int parse_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        const char* value = (i + 1 < argc) ? argv[i + 1] : NULL;

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(arg, "--verbose") == 0) {
            cfg_verbose = 1;
        } else if (strcmp(arg, "--resume") == 0) {
            cfg_resume = 1;
        } else if (strcmp(arg, "--no-validate") == 0) {
            cfg_no_validate = 1;
        } else if (strcmp(arg, "--human-moves-only") == 0) {
            cfg_human_moves_only = 1;
        } else if (strcmp(arg, "--fresh-table") == 0) {
            cfg_fresh_table = 1;
        } else if (strcmp(arg, "--quiet") == 0) {
            cfg_quiet = 1;
        } else if (value == NULL) {
            fprintf(stderr, "%s needs a value\n", arg);
            return 0;
        } else if (strcmp(arg, "--pdn") == 0) {
            cfg_pdn = value; i++;
        } else if (strcmp(arg, "--out") == 0) {
            cfg_out = value; i++;
        } else if (strcmp(arg, "--book-depth") == 0) {
            cfg_book_depth = atoi(value); i++;
        } else if (strcmp(arg, "--depth") == 0) {
            cfg_search_depth = atoi(value); i++;
        } else if (strcmp(arg, "--time") == 0) {
            cfg_seconds = atof(value); i++;
        } else if (strcmp(arg, "--eval-loss") == 0) {
            cfg_eval_loss = atoi(value); i++;
        } else if (strcmp(arg, "--eval-cutoff") == 0) {
            cfg_eval_cutoff = atoi(value); i++;
        } else if (strcmp(arg, "--min-occurrence") == 0) {
            cfg_min_occurrence = atoi(value); i++;
        } else if (strcmp(arg, "--games") == 0) {
            cfg_max_games = atoi(value); i++;
        } else if (strcmp(arg, "--limit") == 0) {
            cfg_max_positions = atoi(value); i++;
        } else if (strcmp(arg, "--jobs") == 0) {
            cfg_jobs = atoi(value); i++;
        } else if (strcmp(arg, "--shard") == 0) {
            if (sscanf(value, "%d/%d", &cfg_shard, &cfg_shards) != 2
                || cfg_shards < 1 || cfg_shard < 0 || cfg_shard >= cfg_shards) {
                fprintf(stderr, "--shard wants I/N with 0 <= I < N\n");
                return 0;
            }
            i++;
        } else {
            fprintf(stderr, "unknown option %s (try --help)\n", arg);
            return 0;
        }
    }

    if (cfg_book_depth < 1 || cfg_search_depth < 1 || cfg_seconds <= 0.0) {
        fprintf(stderr, "--book-depth, --depth and --time must be positive\n");
        return 0;
    }
    if (cfg_jobs != 1 && cfg_shards > 1) {
        fprintf(stderr, "--jobs splits the run into shards itself; do not give "
                        "--shard as well\n");
        return 0;
    }
    if (cfg_jobs != 1 && cfg_no_validate) {
        // nothing to parallelise: aggregation is a seventh of a second
        cfg_jobs = 1;
    }
    return 1;
}

int main(int argc, char* argv[]) {
    if (!parse_args(argc, argv)) {
        return 1;
    }

    // --jobs: become the parent of a pool of workers rather than doing the work
    if (cfg_jobs != 1) {
        return run_workers(argc, argv);
    }

    // A worker's progress would interleave with every other worker's, so all but
    // the first are started with --quiet and drop stdout on the floor. Anything
    // that actually goes wrong is written to stderr, which is left alone.
    if (cfg_quiet) {
#ifdef _WIN32
        freopen("NUL", "w", stdout);
#else
        freopen("/dev/null", "w", stdout);
#endif
    }

    clock_t started = clock();

    build_book_from_pdn();
    filter_by_occurrence();
    qsort(positions, (size_t)positionsCount, sizeof(PositionEntry), compare_positions);
    index_rebuild();

    if (cfg_resume) {
        int loaded = load_book_file(cfg_out);
        printf("resuming: %d positions already validated in %s\n", loaded, cfg_out);
    }

    // --limit applies before the split, so it means the same number of positions
    // whether the run is one process or sixteen
    int total = positionsCount;
    if (cfg_max_positions > 0 && total > cfg_max_positions) {
        total = cfg_max_positions;
    }

    // the shard is a contiguous block of the sorted order, so neighbouring
    // positions stay in the same process and keep sharing its hash table
    int lo = 0;
    int hi = total;
    if (cfg_shards > 1) {
        lo = (int)((long long)total * cfg_shard / cfg_shards);
        hi = (int)((long long)total * (cfg_shard + 1) / cfg_shards);
        printf("shard %d of %d: positions %d..%d\n", cfg_shard, cfg_shards, lo, hi - 1);
    }

    if (cfg_no_validate) {
        for (int i = lo; i < hi; i++) {
            positions[i].validated = 1;
        }
    } else {
        int todo = 0;
        for (int i = lo; i < hi; i++) {
            todo += !positions[i].validated;
        }

        book_search_init();
        clock_t validation_started = clock();
        int done = 0;
        for (int i = lo; i < hi; i++) {
            if (positions[i].validated) {
                continue;  // resumed from a previous run
            }
            validate_book_position(i);
            done++;

            if (done % 20 == 0 || done == todo) {
                double spent = ((double)(clock() - validation_started)) / CLOCKS_PER_SEC;
                double rate = spent / done;
                printf("\r%d/%d positions  %.1fs each  %.0f min left       ",
                       done, todo, rate, (rate * (todo - done)) / 60.0);
                fflush(stdout);
            }
            if (done % CHECKPOINT_EVERY == 0) {
                write_book_to_file(cfg_out, NULL);
            }
        }
        printf("\n");
    }

    int moves_written = 0;
    int written = write_book_to_file(cfg_out, &moves_written);

    if (cfg_verbose) {
        char key[KEY_LENGTH];
        for (int i = 0; i < positionsCount; i++) {
            if (!positions[i].validated) {
                continue;
            }
            compute_position_key(key, positions[i].p1, positions[i].p2,
                                 positions[i].p1k, positions[i].p2k);
            printf("%s  ply %d, %d games\n", key, positions[i].ply, positions[i].occurrence);
            for (int j = 0; j < positions[i].moveCount; j++) {
                printf("    %02d-%02d  eval %5d  %d games: %dW %dL %dD\n",
                       (positions[i].moves[j].move >> 8) & 0xFF,
                       positions[i].moves[j].move & 0xFF,
                       positions[i].moves[j].eval,
                       positions[i].moves[j].games,
                       positions[i].moves[j].outcome.wins,
                       positions[i].moves[j].outcome.losses,
                       positions[i].moves[j].outcome.ties);
            }
        }
    }

    printf("\n%s: %d positions, %d moves (%.2f per position)\n",
           cfg_out, written, moves_written,
           (written > 0) ? (double)moves_written / written : 0.0);
    printf("candidate moves searched: %d\n", stat_candidates);
    printf("human moves dropped as unsound: %d\n", stat_moves_dropped);
    printf("moves kept that no master played: %d\n", stat_moves_engine_only);
    printf("engine best move added:   %d\n", stat_best_added);
    printf("positions with one legal move (not searched): %d\n", stat_forced);
    if (stat_depth_n > 0) {
        printf("mean depth reached: %.2f ply (cap %d)\n",
               (double)stat_depth_sum / stat_depth_n, cfg_search_depth);
    }
    if (stat_moves_timed_out > 0) {
        printf("moves kept unverified (out of time): %d\n", stat_moves_timed_out);
    }
    printf("%.1fM nodes searched, %.1f minutes total\n",
           (double)total_nodes / 1e6, ((double)(clock() - started)) / CLOCKS_PER_SEC / 60.0);

    for (int i = 0; i < positionsCount; i++) {
        free(positions[i].moves);
    }
    free(positions);
    free(index_table);
    if (EVALER != NULL) {
        end_board_search(EVALER);
    }
    return 0;
}
