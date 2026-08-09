#!/usr/bin/env bash
# Build the engine for the browser.
#
# This is a separate script rather than a mode in src/python/Package_engine.py
# on purpose. emcc is not a compiler setuptools can drive through Extension(),
# so a --wasm flag there would end up shelling out from inside setup(), fighting
# both setuptools' model and its staleness logic. Keeping Package_engine.py
# untouched is also the strongest available guarantee that none of this affects
# the desktop .pyd - and it needs no edit, because engine_dependencies() globs
# src/engine/*.c (so wasm_api.c correctly triggers a .pyd rebuild) while
# `sources` stays hardcoded to board_search.c (so it is never compiled into one).
#
# Two artifacts come out of one flag set:
#
#   marcher.js/.wasm         SIMD128. What almost everyone gets.
#   marcher-scalar.js/.wasm  No vectors. Safari below 16.4, and anything else
#                            where the runtime probe in engine-worker.js says
#                            SIMD is unavailable. Substantially slower, but it
#                            plays.
#
# Usage:  bash src/wasm/build_wasm.sh [--debug]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/src/wasm/dist"
mkdir -p "$OUT"

if ! command -v emcc >/dev/null 2>&1; then
    echo "emcc not found on PATH." >&2
    echo "Activate the SDK first, e.g.:" >&2
    echo "  source ~/emsdk/emsdk_env.sh        (bash)" >&2
    echo "  ~/emsdk/emsdk_env.ps1              (PowerShell)" >&2
    exit 1
fi
echo "using $(emcc --version | head -n1)"

COMMON=(
    "$ROOT/src/engine/board_search.c"
    -O3 -flto
    # there is no main(): board_search.c's was commented out long ago, and this
    # is a library, so emcc must be told not to look for an entry point
    --no-entry
    -DNDEBUG
    -DWASM_API=1

    # nothing is listening on stdout in a browser, and the engine writes a
    # progress line per deepening iteration. wasm_on_iteration replaces it.
    -DPRINT_OUTPUT=0

    # 2^21 entries x 16 bytes = 32MB, half the desktop table. A tab also holds
    # the tablebase and the page itself, and at ~1s per move the search visits
    # far too few nodes to fill even this.
    -DTT_ENTRIES_LOG2=21

    # the host fetches tablebase slices and hands them over; no fopen, no getenv,
    # which is what makes -sFILESYSTEM=0 below honest
    -DENDGAME_DB_MEM_LOAD=1
    -DENDGAME_DB_NO_FILES=1

    -sMODULARIZE=1
    -sEXPORT_NAME=createMarcher
    # deliberately NOT EXPORT_ES6: a classic worker importScripts() this file and
    # needs `var createMarcher` at worker global scope. See engine-worker.js for
    # why the worker is classic.
    -sENVIRONMENT=worker,node

    # the bitboards genuinely use bit 63, which a JS Number cannot hold, so
    # u64 arguments cross as BigInt
    -sWASM_BIGINT=1
    -sFILESYSTEM=0

    -sALLOW_MEMORY_GROWTH=1
    -sINITIAL_MEMORY=32MB
    -sMAXIMUM_MEMORY=512MB

    # NOT optional. Emscripten's default stack is 64KB. negmax recurses to 50
    # frames at the "master" setting, each carrying int moves[96] plus locals,
    # and the leaf adds a 1KB aligned NNUE accumulator - roughly 35KB, inside
    # 64KB but with no margin at all. A wasm stack overflow under ASSERTIONS=0
    # is silent memory corruption, not a trap.
    -sSTACK_SIZE=4MB

    -sEXPORTED_FUNCTIONS=_wasm_search,_wasm_result_ptr,_wasm_result_fields,_wasm_perft,_wasm_nnue_eval,_wasm_nnue_simd,_wasm_db_max_pieces,_endgame_db_alloc_slice,_malloc,_free
    -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32
)

if [ "${1:-}" = "--debug" ]; then
    # the build to run the game suite through once: it traps a stack overflow
    # and a bad heap access instead of corrupting memory quietly
    COMMON+=(-sASSERTIONS=1 -sSTACK_OVERFLOW_CHECK=2 -sSAFE_HEAP=1)
    echo "debug build: assertions, stack and heap checks on (slow)"
else
    COMMON+=(-sASSERTIONS=0)
fi

# -msse2 makes clang define __SSE2__, which selects the NNUE_SIMD 1 path already
# in nnue.c; -msimd128 lets those intrinsics lower to wasm SIMD128. Every one
# the SSE2 path uses maps to a single wasm instruction - _mm_madd_epi16 is
# exactly i32x4.dot_i16x8_s - so there is no emulation cost.
#
# -mavx2 is deliberately NOT used. Emscripten can emulate it, but wasm has no
# 256 bit vectors, so __m256i becomes two v128s for no gain.
echo "building SIMD..."
emcc "${COMMON[@]}" -msimd128 -msse2 -o "$OUT/marcher.js"

echo "building scalar fallback..."
emcc "${COMMON[@]}" -DNNUE_NO_SIMD=1 -o "$OUT/marcher-scalar.js"

# A third artifact that exists only to be tested. The two shipping builds use a
# 32MB transposition table, so their search results legitimately differ from the
# desktop engine's 64MB one - same tree shape, different transposition hits.
# This one matches the desktop table exactly, which turns "does the browser
# engine search the same tree as the .pyd" into a question with a yes/no answer.
# src/wasm/verify_wasm.mjs drives it.
echo "building verification build (desktop-sized table)..."
emcc "${COMMON[@]/-DTT_ENTRIES_LOG2=21/-DTT_ENTRIES_LOG2=22}" \
     -msimd128 -msse2 -o "$OUT/marcher-verify.js"

echo
ls -la "$OUT"
echo
echo "built into $OUT"
