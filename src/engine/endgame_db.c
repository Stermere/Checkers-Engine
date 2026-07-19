// Endgame win/loss/draw tablebase: indexing, loading and probing.
// Positions are grouped into material "slices" (m1,k1,m2,k2) = (p1 men, p1 kings,
// p2 men, p2 kings). Each slice is a packed 2-bit array indexed by the colex ranks
// of each piece group's squares plus the side to move. Values are from the
// perspective of the side to move. Only positions with no jump in progress are
// covered (a multi-jump is treated as a single complete move).
//
// File format: db/wld_<m1><k1><m2><k2>.bin - raw 2-bit packed values, 4 per byte.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DB_DRAW 0
#define DB_WIN 1
#define DB_LOSS 2
#define DB_BROKEN 3
#define DB_UNKNOWN 4  // only used during generation (byte arrays)

#define DB_MAX_TOTAL 6   // largest total piece count supported by the data structures
#define DB_MAX_GROUP 5   // most pieces one group can hold (5v1 is the widest split)

// packed slice buffers, indexed [m1][k1][m2][k2]; NULL = not loaded
static unsigned char* DB_SLICE[DB_MAX_GROUP + 1][DB_MAX_GROUP + 1][DB_MAX_GROUP + 1][DB_MAX_GROUP + 1];
// distance-to-win/loss in plies (one byte per position, 255 = none/unknown)
static unsigned char* DB_DTW[DB_MAX_GROUP + 1][DB_MAX_GROUP + 1][DB_MAX_GROUP + 1][DB_MAX_GROUP + 1];
static int DB_LOADED_MAX = 0;  // largest total piece count with at least one loaded slice
static int DB_INIT_DONE = 0;

// binomial coefficients C[n][k] for n<=32, k<=6
static long long DB_C[33][7];

static void db_init_binomials(){
    for (int n = 0; n <= 32; n++){
        DB_C[n][0] = 1;
        for (int k = 1; k <= 6; k++){
            DB_C[n][k] = (n == 0) ? 0 : DB_C[n - 1][k - 1] + DB_C[n - 1][k];
        }
    }
}

// map a 64-board bit index to the 0..31 playable-square index and back.
// dark squares: even rows use odd columns, odd rows use even columns.
static inline int db_bit64_to_sq32(int b){
    return (b >> 3) * 4 + ((b & 7) >> 1);
}
static inline int db_sq32_to_bit64(int s){
    int row = s >> 2;
    int col = ((s & 3) << 1) + ((row & 1) ? 0 : 1);
    return row * 8 + col;
}

// number of positions in a slice (men have 28 legal squares, kings 32)
static long long db_slice_size(int m1, int k1, int m2, int k2){
    return DB_C[28][m1] * DB_C[32][k1] * DB_C[28][m2] * DB_C[32][k2] * 2ll;
}

// colex rank of a strictly-increasing square list
static long long db_rank_group(const int* sq, int count){
    long long r = 0;
    for (int i = 0; i < count; i++){
        r += DB_C[sq[i]][i + 1];
    }
    return r;
}

// extract the sq32 indices of a bitboard group in increasing order.
// offset shifts the square index (4 for p1 men so rows 1-7 rank over 0..27),
// max_valid rejects squares a piece of this type cannot legally stand on
// (27 for men - a man on its promotion row would be a king - 31 for kings)
static int db_extract_group(unsigned long long bits, int* out, int offset, int max_valid){
    int n = 0;
    while (bits){
        int b = countTrailingZeros(bits);
        int s = db_bit64_to_sq32(b) - offset;
        if (s < 0 || s > max_valid) return -1;
        out[n++] = s;
        bits &= bits - 1;
    }
    return n;
}

// encode a position into its slice-local index; returns -1 if not encodable
static long long db_encode(long long p1, long long p2, long long p1k, long long p2k, int stm,
                           int m1, int k1, int m2, int k2){
    int g[4][DB_MAX_GROUP + 1];
    if (db_extract_group((unsigned long long)p1, g[0], 4, 27) != m1) return -1;
    if (db_extract_group((unsigned long long)p1k, g[1], 0, 31) != k1) return -1;
    if (db_extract_group((unsigned long long)p2, g[2], 0, 27) != m2) return -1;
    if (db_extract_group((unsigned long long)p2k, g[3], 0, 31) != k2) return -1;

    long long idx = db_rank_group(g[0], m1);
    idx = idx * DB_C[32][k1] + db_rank_group(g[1], k1);
    idx = idx * DB_C[28][m2] + db_rank_group(g[2], m2);
    idx = idx * DB_C[32][k2] + db_rank_group(g[3], k2);
    return idx * 2 + (stm - 1);
}

// read a packed 2-bit value
static inline int db_get_packed(const unsigned char* buf, long long idx){
    return (buf[idx >> 2] >> ((idx & 3) * 2)) & 3;
}
static inline void db_set_packed(unsigned char* buf, long long idx, int v){
    long long byte = idx >> 2;
    int shift = (int)(idx & 3) * 2;
    buf[byte] = (unsigned char)((buf[byte] & ~(3 << shift)) | (v << shift));
}

// probe the database. returns 1 on hit and stores DB_WIN/DB_LOSS/DB_DRAW
// (from the side-to-move's perspective) into *result, and the distance to the
// win/loss in plies into *dtw (255 when unavailable or drawn).
int probe_endgame_db(long long p1, long long p2, long long p1k, long long p2k, int player, int* result, int* dtw){
    int m1 = get_bits_set(p1), k1 = get_bits_set(p1k);
    int m2 = get_bits_set(p2), k2 = get_bits_set(p2k);
    if (m1 > DB_MAX_GROUP || k1 > DB_MAX_GROUP || m2 > DB_MAX_GROUP || k2 > DB_MAX_GROUP) return 0;

    unsigned char* buf = DB_SLICE[m1][k1][m2][k2];
    if (buf == NULL) return 0;

    long long idx = db_encode(p1, p2, p1k, p2k, player, m1, k1, m2, k2);
    if (idx < 0) return 0;

    int v = db_get_packed(buf, idx);
    if (v == DB_BROKEN) return 0;
    *result = v;
    unsigned char* dbuf = DB_DTW[m1][k1][m2][k2];
    *dtw = (dbuf != NULL) ? dbuf[idx] : 255;
    return 1;
}

// try to load every slice file up to max_total pieces from dir. safe to call once.
void endgame_db_init(const char* dir, int max_total){
    if (DB_INIT_DONE) return;
    DB_INIT_DONE = 1;
    db_init_binomials();
    memset(DB_SLICE, 0, sizeof(DB_SLICE));
    memset(DB_DTW, 0, sizeof(DB_DTW));

    int loaded = 0;
    for (int total = 2; total <= max_total && total <= DB_MAX_TOTAL; total++){
        for (int m1 = 0; m1 <= total && m1 <= DB_MAX_GROUP; m1++){
            for (int k1 = 0; m1 + k1 <= total && k1 <= DB_MAX_GROUP; k1++){
                if (m1 + k1 < 1 || m1 + k1 > total - 1) continue;
                for (int m2 = 0; m1 + k1 + m2 <= total && m2 <= DB_MAX_GROUP; m2++){
                    int k2 = total - m1 - k1 - m2;
                    if (k2 < 0 || k2 > DB_MAX_GROUP || m2 + k2 < 1) continue;

                    char path[512];
                    snprintf(path, sizeof(path), "%s/wld_%d%d%d%d.bin", dir, m1, k1, m2, k2);
                    FILE* f = fopen(path, "rb");
                    if (f == NULL) continue;

                    long long size = db_slice_size(m1, k1, m2, k2);
                    long long bytes = (size + 3) / 4;
                    unsigned char* buf = (unsigned char*)malloc((size_t)bytes);
                    if (buf != NULL && fread(buf, 1, (size_t)bytes, f) == (size_t)bytes){
                        DB_SLICE[m1][k1][m2][k2] = buf;
                        loaded++;
                        if (total > DB_LOADED_MAX) DB_LOADED_MAX = total;
                    } else {
                        free(buf);
                    }
                    fclose(f);

                    // matching distance file (optional but expected)
                    snprintf(path, sizeof(path), "%s/dtw_%d%d%d%d.bin", dir, m1, k1, m2, k2);
                    f = fopen(path, "rb");
                    if (f != NULL){
                        unsigned char* dbuf = (unsigned char*)malloc((size_t)size);
                        if (dbuf != NULL && fread(dbuf, 1, (size_t)size, f) == (size_t)size){
                            DB_DTW[m1][k1][m2][k2] = dbuf;
                        } else {
                            free(dbuf);
                        }
                        fclose(f);
                    }
                }
            }
        }
    }
    if (loaded > 0){
        printf("endgame db: loaded %d slices (up to %d pieces)\n", loaded, DB_LOADED_MAX);
    }
}

// largest piece count the loaded database can answer for
int endgame_db_max_pieces(){
    return DB_LOADED_MAX;
}
