// Endgame WLD tablebase generator.
// Builds db/wld_<m1><k1><m2><k2>.bin files for all material slices up to a piece
// count given on the command line (default 5). Reuses the engine's move generation
// (board_search.c) so the database rules match the search exactly.
//
// Method: slices are processed in order of (total pieces asc, total men asc).
// Capture moves and promotions always lead into already-finished slices, so a
// position with captures gets its final value immediately; quiet positions are
// resolved by value iteration within the slice. Unresolvable positions are draws.
//
// The iteration relaxes the distance-to-win as well as the win/loss/draw value,
// and both are run to a fixed point. Stopping at the value (as this used to do)
// is enough for a correct WLD bit but leaves the distance at whatever the first
// winning successor to resolve happened to offer, which is only an upper bound -
// see the comment on the relaxation loop in gen_slice.
//
// Build:  cl /O2 /Fe:dbgen.exe src/engine/db_gen.c
// Run:    dbgen.exe [max_pieces]        generate (resumes from existing files)
//         dbgen.exe check [max_pieces]  verify values and distances on disk

#include "board_search.c"
#include <direct.h>

static const char* DB_DIR = "db";

// ---------------------------------------------------------------------------
// decoding (unranking) - the inverse of db_encode in endgame_db.c
// ---------------------------------------------------------------------------

// colex unrank: fill out[0..count-1] with increasing square indices
static void db_unrank_group(long long rank, int count, int* out){
    for (int i = count; i >= 1; i--){
        int s = i - 1;
        while (s < 31 && DB_C[s + 1][i] <= rank){
            s++;
        }
        out[i - 1] = s;
        rank -= DB_C[s][i];
    }
}

// decode a slice-local index into bitboards; returns 0 if the position is invalid
// (two pieces on the same square)
static int db_decode(long long idx, int m1, int k1, int m2, int k2,
                     long long* p1, long long* p2, long long* p1k, long long* p2k, int* stm){
    *stm = (int)(idx & 1) + 1;
    long long r = idx >> 1;

    long long rk2 = r % DB_C[32][k2]; r /= DB_C[32][k2];
    long long rm2 = r % DB_C[28][m2]; r /= DB_C[28][m2];
    long long rk1 = r % DB_C[32][k1]; r /= DB_C[32][k1];
    long long rm1 = r;

    int g[DB_MAX_GROUP + 1];
    unsigned int used = 0;
    *p1 = 0; *p2 = 0; *p1k = 0; *p2k = 0;

    db_unrank_group(rm1, m1, g);
    for (int i = 0; i < m1; i++){
        int s = g[i] + 4; // p1 men rank over rows 1-7
        if (used & (1u << s)) return 0;
        used |= 1u << s;
        *p1 |= 1ll << db_sq32_to_bit64(s);
    }
    db_unrank_group(rk1, k1, g);
    for (int i = 0; i < k1; i++){
        int s = g[i];
        if (used & (1u << s)) return 0;
        used |= 1u << s;
        *p1k |= 1ll << db_sq32_to_bit64(s);
    }
    db_unrank_group(rm2, m2, g);
    for (int i = 0; i < m2; i++){
        int s = g[i]; // p2 men rank over rows 0-6
        if (used & (1u << s)) return 0;
        used |= 1u << s;
        *p2 |= 1ll << db_sq32_to_bit64(s);
    }
    db_unrank_group(rk2, k2, g);
    for (int i = 0; i < k2; i++){
        int s = g[i];
        if (used & (1u << s)) return 0;
        used |= 1u << s;
        *p2k |= 1ll << db_sq32_to_bit64(s);
    }
    return 1;
}

// ---------------------------------------------------------------------------
// value helpers
// ---------------------------------------------------------------------------

static inline int db_negate(int v){
    if (v == DB_WIN) return DB_LOSS;
    if (v == DB_LOSS) return DB_WIN;
    return v;
}

// preference order when choosing among move results: WIN > DRAW > LOSS
static inline int db_better(int a, int b){
    if (a == DB_WIN || b == DB_WIN) return DB_WIN;
    if (a == DB_DRAW || b == DB_DRAW) return DB_DRAW;
    return DB_LOSS;
}

// bit set on a val[] byte during generation to mark a position whose value and
// distance are already final: an invalid index, or a position with a capture
// available. Captures are mandatory and every capture resolves entirely into a
// smaller (already generated) slice, so nothing inside this slice can change it
// and the relaxation loop must not re-evaluate it as if it were quiet.
// DB_UNKNOWN is 4, so the flag has to sit above the low three bits.
#define DB_FINAL 8
#define DB_VAL_OF(b) ((b) & 7)

// results are passed around encoded as (value << 8) | dtw, where dtw is the
// distance to the win/loss in plies (255 for draws / not applicable)
#define DB_ENC(v, d) (((v) << 8) | ((d) > 254 ? 254 : (d)))
#define DB_VAL(e) ((e) >> 8)
#define DB_DIST(e) ((e) & 0xFF)

// negate an encoded result into the parent's perspective, adding one ply
static inline int db_negate_enc(int e){
    int v = DB_VAL(e), d = DB_DIST(e);
    if (v == DB_WIN) return DB_ENC(DB_LOSS, d + 1);
    if (v == DB_LOSS) return DB_ENC(DB_WIN, d + 1);
    return DB_ENC(v, 255);
}

// preference between two encoded results for the side choosing the move:
// WIN > DRAW > LOSS; among wins prefer the shortest, among losses the longest
static inline int db_better_enc(int a, int b){
    int va = DB_VAL(a), vb = DB_VAL(b);
    if (va != vb){
        if (va == DB_WIN) return a;
        if (vb == DB_WIN) return b;
        if (va == DB_DRAW) return a;
        return b;
    }
    if (va == DB_WIN) return (DB_DIST(a) <= DB_DIST(b)) ? a : b;
    if (va == DB_LOSS) return (DB_DIST(a) >= DB_DIST(b)) ? a : b;
    return a;
}

// look up a finished position in an already-generated (packed) slice.
// stm = the side to move in that position. returns an encoded result.
static int db_lookup_final(long long p1, long long p2, long long p1k, long long p2k, int stm){
    long long own = (stm == 1) ? (p1 | p1k) : (p2 | p2k);
    if (own == 0) return DB_ENC(DB_LOSS, 0); // side to move has no pieces: it has lost

    int m1 = get_bits_set(p1), k1 = get_bits_set(p1k);
    int m2 = get_bits_set(p2), k2 = get_bits_set(p2k);
    unsigned char* buf = DB_SLICE[m1][k1][m2][k2];
    if (buf == NULL){
        printf("FATAL: dependency slice %d%d%d%d missing\n", m1, k1, m2, k2);
        exit(1);
    }
    long long idx = db_encode(p1, p2, p1k, p2k, stm, m1, k1, m2, k2);
    int v = db_get_packed(buf, idx);
    unsigned char* dbuf = DB_DTW[m1][k1][m2][k2];
    return DB_ENC(v, dbuf != NULL ? dbuf[idx] : 254);
}

// resolve every continuation of a multi-jump in progress (piece at pos must keep
// jumping); returns the best achievable encoded result for `player`
static int db_resolve_continuation(long long p1, long long p2, long long p1k, long long p2k,
                                   int player, int pos, char* offsets){
    int buf[8];
    int n = generate_moves(p1, p2, p1k, p2k, pos, buf, offsets, True);
    int best = DB_ENC(DB_LOSS, 0);
    for (int i = 0; i < n; i++){
        long long a = p1, b = p2, c = p1k, d = p2k;
        int ptype = get_piece_at_location(a, b, c, d, buf[i * 2]);
        update_board(&a, &b, &c, &d, buf[i * 2], buf[i * 2 + 1]);
        int pnext = get_next_board_state(a, b, c, d, buf[i * 2], buf[i * 2 + 1], player, ptype, offsets);
        int v;
        if (pnext == player){
            v = db_resolve_continuation(a, b, c, d, player, buf[i * 2 + 1], offsets);
        } else {
            v = db_negate_enc(db_lookup_final(a, b, c, d, pnext));
        }
        best = db_better_enc(best, v);
    }
    return best;
}

// encoded value of a position where the side to move has at least one capture.
// every capture fully resolves into a smaller (finished) slice.
static int db_resolve_captures(long long p1, long long p2, long long p1k, long long p2k,
                               int player, char* offsets){
    long long own = (player == 1) ? (p1 | p1k) : (p2 | p2k);
    int buf[8];
    int best = DB_ENC(DB_LOSS, 0);
    unsigned long long bits = (unsigned long long)own;
    while (bits){
        int b = countTrailingZeros(bits);
        bits &= bits - 1;
        int n = generate_moves(p1, p2, p1k, p2k, b, buf, offsets, True);
        for (int i = 0; i < n; i++){
            long long a1 = p1, b1 = p2, c1 = p1k, d1 = p2k;
            int ptype = get_piece_at_location(a1, b1, c1, d1, buf[i * 2]);
            update_board(&a1, &b1, &c1, &d1, buf[i * 2], buf[i * 2 + 1]);
            int pnext = get_next_board_state(a1, b1, c1, d1, buf[i * 2], buf[i * 2 + 1], player, ptype, offsets);
            int v;
            if (pnext == player){
                v = db_resolve_continuation(a1, b1, c1, d1, player, buf[i * 2 + 1], offsets);
            } else {
                v = db_negate_enc(db_lookup_final(a1, b1, c1, d1, pnext));
            }
            best = db_better_enc(best, v);
        }
    }
    return best;
}

// does `player` have any capture available?
static int db_has_capture(long long p1, long long p2, long long p1k, long long p2k, int player, char* offsets){
    long long own = (player == 1) ? (p1 | p1k) : (p2 | p2k);
    int buf[8];
    unsigned long long bits = (unsigned long long)own;
    while (bits){
        int b = countTrailingZeros(bits);
        bits &= bits - 1;
        if (generate_moves(p1, p2, p1k, p2k, b, buf, offsets, True) > 0) return 1;
    }
    return 0;
}

// evaluate a quiet position from its successors. val/dtw = this slice's arrays.
// returns an encoded result, or (DB_UNKNOWN << 8) when not yet determined.
static int db_eval_quiet(long long p1, long long p2, long long p1k, long long p2k, int player,
                         unsigned char* val, unsigned char* dtw,
                         int m1, int k1, int m2, int k2, char* offsets){
    long long own = (player == 1) ? (p1 | p1k) : (p2 | p2k);
    int buf[8];
    int any_unknown = 0;
    int any_draw = 0;
    int any_move = 0;
    int best_win_d = 255;   // shortest distance among winning replies
    int worst_loss_d = -1;  // longest distance among losing replies
    unsigned long long bits = (unsigned long long)own;
    while (bits){
        int b = countTrailingZeros(bits);
        bits &= bits - 1;
        int n = generate_moves(p1, p2, p1k, p2k, b, buf, offsets, False);
        for (int i = 0; i < n; i++){
            any_move = 1;
            long long a1 = p1, b1 = p2, c1 = p1k, d1 = p2k;
            int ptype = get_piece_at_location(a1, b1, c1, d1, buf[i * 2]);
            update_board(&a1, &b1, &c1, &d1, buf[i * 2], buf[i * 2 + 1]);
            int opp = player ^ 0x3;

            int v, d;
            int promoted = (ptype <= 2) && (buf[i * 2 + 1] < 8 || buf[i * 2 + 1] > 55);
            if (promoted){
                // different slice (one fewer man, one more king) - already finished
                int e = db_lookup_final(a1, b1, c1, d1, opp);
                v = DB_VAL(e);
                d = DB_DIST(e);
            } else {
                long long idx = db_encode(a1, b1, c1, d1, opp, m1, k1, m2, k2);
                v = DB_VAL_OF(val[idx]);
                d = dtw[idx];
            }
            if (v == DB_LOSS && d + 1 < best_win_d) best_win_d = d + 1;
            if (v == DB_UNKNOWN) any_unknown = 1;
            else if (v == DB_DRAW) any_draw = 1;
            else if (v == DB_WIN && d + 1 > worst_loss_d) worst_loss_d = d + 1;
        }
    }
    if (best_win_d < 255) return DB_ENC(DB_WIN, best_win_d);
    if (!any_move) return DB_ENC(DB_LOSS, 0); // no moves: side to move loses
    if (any_unknown) return DB_UNKNOWN << 8;
    if (any_draw) return DB_ENC(DB_DRAW, 255);
    return DB_ENC(DB_LOSS, worst_loss_d < 0 ? 254 : worst_loss_d);
}

// ---------------------------------------------------------------------------
// slice generation
// ---------------------------------------------------------------------------

static void gen_slice(int m1, int k1, int m2, int k2, char* offsets){
    long long S = db_slice_size(m1, k1, m2, k2);
    clock_t t0 = clock();
    unsigned char* val = (unsigned char*)malloc((size_t)S);
    unsigned char* dtw = (unsigned char*)malloc((size_t)S);
    memset(dtw, 255, (size_t)S);

    long long p1, p2, p1k, p2k;
    int stm;

    // pass 0: broken positions, capture positions (final), terminal losses
    for (long long idx = 0; idx < S; idx++){
        if (!db_decode(idx, m1, k1, m2, k2, &p1, &p2, &p1k, &p2k, &stm)){
            val[idx] = DB_BROKEN | DB_FINAL;
            continue;
        }
        if (db_has_capture(p1, p2, p1k, p2k, stm, offsets)){
            int e = db_resolve_captures(p1, p2, p1k, p2k, stm, offsets);
            val[idx] = (unsigned char)(DB_VAL(e) | DB_FINAL);
            dtw[idx] = (unsigned char)DB_DIST(e);
        } else {
            val[idx] = DB_UNKNOWN;
        }
    }

    // Relaxation over the quiet positions, to a fixed point on the DISTANCE and
    // not just on the win/loss/draw value.
    //
    // A position cannot simply be frozen the first time it resolves. db_eval_quiet
    // takes the minimum over the winning successors that are known *at that
    // moment* and ignores the ones still unresolved, so the distance it returns is
    // an upper bound, not the answer. Two things stop "first resolved" from also
    // meaning "shortest":
    //   - pass 0 seeds every capture position with a distance carried in from a
    //     smaller slice, which can be any value at all. A pass number therefore
    //     does not correspond to a distance layer, so the usual retrograde
    //     argument (layer k resolves exactly the wins at distance k) does not hold.
    //   - the sweep reads and writes val/dtw in place, so a chain of many plies
    //     can collapse inside a single pass whenever the index order happens to
    //     follow it, while a shorter win elsewhere waits for the next pass.
    // Freezing therefore locked in whichever winning line the sweep order reached
    // first, and the error compounded through every slice built on top of it.
    //
    // Re-evaluating instead is safe and converges: the win/loss/draw value of a
    // resolved position never changes (a winning successor stays lost for the
    // opponent, and a loss is only recorded once every successor is known), while
    // both a win's minimum and a loss's maximum move monotonically downward as
    // successors resolve and their own distances improve. Distances are bounded
    // below by zero, so the loop terminates.
    //
    // Sweep direction alternates: this is Bellman-Ford style relaxation, where a
    // correction travels along the sweep and stalls against it, so alternating
    // roughly halves the number of passes on chains that run backwards in index
    // order.
    int pass = 0;
    long long changed = 1;
    int forward = 1;
    while (changed){
        changed = 0;
        pass++;
        for (long long i = 0; i < S; i++){
            long long idx = forward ? i : (S - 1 - i);
            if (val[idx] & DB_FINAL) continue;
            db_decode(idx, m1, k1, m2, k2, &p1, &p2, &p1k, &p2k, &stm);
            int e = db_eval_quiet(p1, p2, p1k, p2k, stm, val, dtw, m1, k1, m2, k2, offsets);
            if (DB_VAL(e) == DB_UNKNOWN) continue;
            if (val[idx] == DB_UNKNOWN){
                val[idx] = (unsigned char)DB_VAL(e);
                dtw[idx] = (unsigned char)DB_DIST(e);
                changed++;
            } else if (DB_DIST(e) < dtw[idx]){
                // same value, shorter proof: a successor resolved or got shorter
                dtw[idx] = (unsigned char)DB_DIST(e);
                changed++;
            }
        }
        forward = !forward;
    }

    // whatever never resolves is a draw; pack and count
    long long bytes = (S + 3) / 4;
    unsigned char* packed = (unsigned char*)calloc((size_t)bytes, 1);
    long long wins = 0, losses = 0, draws = 0, broken = 0;
    for (long long idx = 0; idx < S; idx++){
        int v = DB_VAL_OF(val[idx]);
        if (v == DB_UNKNOWN){ v = DB_DRAW; dtw[idx] = 255; }
        if (v == DB_WIN) wins++;
        else if (v == DB_LOSS) losses++;
        else if (v == DB_DRAW){ draws++; dtw[idx] = 255; }
        else broken++;
        db_set_packed(packed, idx, v);
    }
    free(val);

    char path[512];
    snprintf(path, sizeof(path), "%s/wld_%d%d%d%d.bin", DB_DIR, m1, k1, m2, k2);
    FILE* f = fopen(path, "wb");
    fwrite(packed, 1, (size_t)bytes, f);
    fclose(f);
    snprintf(path, sizeof(path), "%s/dtw_%d%d%d%d.bin", DB_DIR, m1, k1, m2, k2);
    f = fopen(path, "wb");
    fwrite(dtw, 1, (size_t)S, f);
    fclose(f);

    DB_SLICE[m1][k1][m2][k2] = packed;
    DB_DTW[m1][k1][m2][k2] = dtw;

    double secs = ((double)(clock() - t0)) / CLOCKS_PER_SEC;
    printf("slice %d%d%d%d: %lld pos, %d passes, W/L/D %lld/%lld/%lld, %.1fs\n",
           m1, k1, m2, k2, S, pass, wins, losses, draws, secs);
    fflush(stdout);
}

// try to load an existing slice (wld + dtw files) for resume; returns 1 on success
static int try_load_slice(int m1, int k1, int m2, int k2){
    char path[512];
    long long S = db_slice_size(m1, k1, m2, k2);

    snprintf(path, sizeof(path), "%s/wld_%d%d%d%d.bin", DB_DIR, m1, k1, m2, k2);
    FILE* f = fopen(path, "rb");
    if (f == NULL) return 0;
    long long bytes = (S + 3) / 4;
    unsigned char* buf = (unsigned char*)malloc((size_t)bytes);
    int ok = (buf != NULL) && (fread(buf, 1, (size_t)bytes, f) == (size_t)bytes);
    fclose(f);
    if (!ok){
        free(buf);
        return 0;
    }

    snprintf(path, sizeof(path), "%s/dtw_%d%d%d%d.bin", DB_DIR, m1, k1, m2, k2);
    f = fopen(path, "rb");
    if (f == NULL){
        free(buf);
        return 0;
    }
    unsigned char* dbuf = (unsigned char*)malloc((size_t)S);
    ok = (dbuf != NULL) && (fread(dbuf, 1, (size_t)S, f) == (size_t)S);
    fclose(f);
    if (!ok){
        free(buf);
        free(dbuf);
        return 0;
    }

    DB_SLICE[m1][k1][m2][k2] = buf;
    DB_DTW[m1][k1][m2][k2] = dbuf;
    return 1;
}

// verify a slice: every stored value AND distance must satisfy the minimax
// relation over its successors (all read back from the final packed data).
//
// For a win the relation dtw(P) = 1 + min{ dtw(S) : S a successor the opponent
// loses } has a single solution, because distances are grounded in positions
// with no moves - so a distance that merely looks self-consistent cannot be
// wrong, and any mismatch here is a real error. Distances are what the search
// uses to pick a conversion move, so they are checked as strictly as the values.
//
// returns error count.
static long long check_slice(int m1, int k1, int m2, int k2, char* offsets, long long sample_step,
                             long long* dist_errors){
    long long S = db_slice_size(m1, k1, m2, k2);
    unsigned char* packed = DB_SLICE[m1][k1][m2][k2];
    unsigned char* dist = DB_DTW[m1][k1][m2][k2];
    long long errors = 0;
    long long p1, p2, p1k, p2k;
    int stm;
    int buf[8];

    for (long long idx = 0; idx < S; idx += sample_step){
        int stored = db_get_packed(packed, idx);
        if (stored == DB_BROKEN) continue;
        if (!db_decode(idx, m1, k1, m2, k2, &p1, &p2, &p1k, &p2k, &stm)){
            printf("  ERR %d%d%d%d idx %lld: stored %d but position invalid\n", m1, k1, m2, k2, idx, stored);
            errors++;
            continue;
        }

        int expected, expected_d = -1;
        if (db_has_capture(p1, p2, p1k, p2k, stm, offsets)){
            int e = db_resolve_captures(p1, p2, p1k, p2k, stm, offsets);
            expected = DB_VAL(e);
            expected_d = DB_DIST(e);
        } else {
            long long own = (stm == 1) ? (p1 | p1k) : (p2 | p2k);
            int any_move = 0, any_draw = 0;
            int best_win = 255, worst_loss = -1;
            unsigned long long bits = (unsigned long long)own;
            while (bits){
                int b = countTrailingZeros(bits);
                bits &= bits - 1;
                int n = generate_moves(p1, p2, p1k, p2k, b, buf, offsets, False);
                for (int i = 0; i < n; i++){
                    any_move = 1;
                    long long a1 = p1, b1 = p2, c1 = p1k, d1 = p2k;
                    update_board(&a1, &b1, &c1, &d1, buf[i * 2], buf[i * 2 + 1]);
                    int e = db_lookup_final(a1, b1, c1, d1, stm ^ 0x3);
                    int v = DB_VAL(e), d = DB_DIST(e);
                    if (v == DB_LOSS && d + 1 < best_win) best_win = d + 1;
                    if (v == DB_WIN && d + 1 > worst_loss) worst_loss = d + 1;
                    if (v == DB_DRAW) any_draw = 1;
                }
            }
            if (best_win < 255){ expected = DB_WIN; expected_d = best_win; }
            else if (!any_move){ expected = DB_LOSS; expected_d = 0; }
            else if (any_draw){ expected = DB_DRAW; expected_d = 255; }
            else { expected = DB_LOSS; expected_d = (worst_loss < 0) ? 254 : worst_loss; }
        }

        if (expected != stored){
            if (errors < 5){
                printf("  ERR %d%d%d%d idx %lld: stored %d expected %d (stm %d)\n",
                       m1, k1, m2, k2, idx, stored, expected, stm);
            }
            errors++;
        } else if (stored != DB_DRAW && dist != NULL && dist[idx] != expected_d){
            if (*dist_errors < 5){
                printf("  ERR %d%d%d%d idx %lld: dtw %d expected %d (%s, stm %d)\n",
                       m1, k1, m2, k2, idx, dist[idx], expected_d,
                       stored == DB_WIN ? "win" : "loss", stm);
            }
            (*dist_errors)++;
        }
    }
    return errors;
}

static void run_check(int maxp, char* offsets){
    long long total_errors = 0, total_dist_errors = 0;
    for (int total = 2; total <= maxp; total++){
        for (int men_total = 0; men_total <= total; men_total++){
            for (int m1 = 0; m1 <= men_total && m1 <= DB_MAX_GROUP; m1++){
                int m2 = men_total - m1;
                if (m2 > DB_MAX_GROUP) continue;
                for (int k1 = 0; k1 <= total - men_total && k1 <= DB_MAX_GROUP; k1++){
                    int k2 = total - men_total - k1;
                    if (k2 > DB_MAX_GROUP) continue;
                    if (m1 + k1 < 1 || m2 + k2 < 1) continue;
                    if (m1 + k1 > DB_MAX_GROUP || m2 + k2 > DB_MAX_GROUP) continue;
                    if (DB_SLICE[m1][k1][m2][k2] == NULL){
                        if (!try_load_slice(m1, k1, m2, k2)){
                            printf("slice %d%d%d%d: MISSING\n", m1, k1, m2, k2);
                            continue;
                        }
                    }
                    long long S = db_slice_size(m1, k1, m2, k2);
                    long long step = (S > 2000000) ? 17 : 1; // sample large slices
                    long long de = 0;
                    long long e = check_slice(m1, k1, m2, k2, offsets, step, &de);
                    printf("check %d%d%d%d: %lld value errors, %lld distance errors%s\n",
                           m1, k1, m2, k2, e, de, step > 1 ? " (sampled)" : "");
                    total_errors += e;
                    total_dist_errors += de;
                }
            }
        }
    }
    printf("total: %lld value errors, %lld distance errors\n", total_errors, total_dist_errors);
}

int main(int argc, char** argv){
    if (argc > 1 && strcmp(argv[1], "check") == 0){
        int maxp = (argc > 2) ? atoi(argv[2]) : 5;
        db_init_binomials();
        memset(DB_SLICE, 0, sizeof(DB_SLICE));
    memset(DB_DTW, 0, sizeof(DB_DTW));
        run_check(maxp, compute_offsets());
        return 0;
    }

    int maxp = (argc > 1) ? atoi(argv[1]) : 5;
    if (maxp > DB_MAX_TOTAL) maxp = DB_MAX_TOTAL;

    db_init_binomials();
    memset(DB_SLICE, 0, sizeof(DB_SLICE));
    memset(DB_DTW, 0, sizeof(DB_DTW));
    _mkdir(DB_DIR);
    char* offsets = compute_offsets();

    clock_t t0 = clock();
    // slices ordered by total pieces, then total men, so captures and promotions
    // always resolve into finished slices
    for (int total = 2; total <= maxp; total++){
        for (int men_total = 0; men_total <= total; men_total++){
            for (int m1 = 0; m1 <= men_total && m1 <= DB_MAX_GROUP; m1++){
                int m2 = men_total - m1;
                if (m2 > DB_MAX_GROUP) continue;
                for (int k1 = 0; k1 <= total - men_total && k1 <= DB_MAX_GROUP; k1++){
                    int k2 = total - men_total - k1;
                    if (k2 > DB_MAX_GROUP) continue;
                    if (m1 + k1 < 1 || m2 + k2 < 1) continue;
                    if (m1 + k1 > DB_MAX_GROUP || m2 + k2 > DB_MAX_GROUP) continue;
                    if (try_load_slice(m1, k1, m2, k2)){
                        printf("slice %d%d%d%d: loaded from disk\n", m1, k1, m2, k2);
                        continue;
                    }
                    gen_slice(m1, k1, m2, k2, offsets);
                }
            }
        }
    }
    printf("done in %.1fs\n", ((double)(clock() - t0)) / CLOCKS_PER_SEC);
    return 0;
}
