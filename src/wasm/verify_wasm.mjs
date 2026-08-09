// Hold the WebAssembly engine to the same standard as the desktop .pyd.
//
// src/python/verify_identical.py proves that two *native* builds behave
// identically. This is the other half: it replays that exact probe sequence
// through the browser engine and demands the same answers.
//
// Why this can be an equality test rather than a similarity test: there is no
// floating point anywhere in the search or the evaluation. The network is
// integer arithmetic on purpose (see the comment at Package_engine.py:122), the
// only doubles in the engine are the time limit and clock(), and the probe runs
// with a time budget so large that no clock check can fire. So the result is a
// pure function of the input, and "close enough" is not a category that needs
// to exist here.
//
// Two things have to line up for the comparison to be meaningful:
//
//   * **Table size.** The shipping builds use a 32MB transposition table and the
//     desktop engine uses 64MB, which legitimately changes which positions get
//     a transposition hit. build_wasm.sh emits marcher-verify.js at the desktop
//     size for exactly this reason - that is the build this script loads.
//
//   * **No database.** The wasm build has no filesystem, so it starts with an
//     empty tablebase. The reference must therefore come from a run with
//     CHECKERS_DB_DIR pointing somewhere that does not exist, which is what this
//     script asks for when it generates one.
//
//   * **Order.** The transposition table persists across searches, so a result
//     depends on every search before it. The replay order here mirrors
//     run_probes() in verify_identical.py exactly.
//
// Usage:
//   node src/wasm/verify_wasm.mjs
//   node src/wasm/verify_wasm.mjs --ref some_reference.json
//   node src/wasm/verify_wasm.mjs --module dist/marcher.js   (expect search diffs)

import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const require = createRequire(import.meta.url);
const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');

// ---------------------------------------------------------------------------
// arguments
// ---------------------------------------------------------------------------
const argv = process.argv.slice(2);
function opt(name, fallback) {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : fallback;
}
const MODULE_PATH = path.resolve(HERE, opt('--module', 'dist/marcher-verify.js'));
const REF_PATH = opt('--ref', null);

// ---------------------------------------------------------------------------
// reference
// ---------------------------------------------------------------------------
function pythonExe() {
  const venv = path.join(ROOT, '.venv', 'Scripts', 'python.exe');
  if (fs.existsSync(venv)) return venv;
  const venvPosix = path.join(ROOT, '.venv', 'bin', 'python');
  if (fs.existsSync(venvPosix)) return venvPosix;
  return process.platform === 'win32' ? 'python' : 'python3';
}

function generateReference() {
  const out = path.join(HERE, 'dist', 'reference.json');
  fs.mkdirSync(path.dirname(out), { recursive: true });
  console.log('generating reference from the native engine (no database)...');
  const res = spawnSync(pythonExe(), [
    path.join('src', 'python', 'verify_identical.py'),
    '--module', 'search_engine',
    '--out', out,
    // must match what the wasm build can actually do; see the header comment
    '--perft-depth', '6',
    '--search-depth', '9',
  ], {
    cwd: ROOT,
    encoding: 'utf8',
    // the wasm engine has no tablebase, so the reference must not have one either
    env: { ...process.env, CHECKERS_DB_DIR: path.join(ROOT, '__no_such_db__') },
  });
  if (res.status !== 0) {
    console.error(res.stdout || '', res.stderr || '');
    throw new Error(`reference generation failed (exit ${res.status})`);
  }
  return out;
}

const refPath = REF_PATH ? path.resolve(REF_PATH) : generateReference();
const ref = JSON.parse(fs.readFileSync(refPath, 'utf8'));
console.log(`reference: ${path.relative(ROOT, refPath)} (${ref.module})`);

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------
if (!fs.existsSync(MODULE_PATH)) {
  console.error(`no wasm build at ${MODULE_PATH}\nRun: bash src/wasm/build_wasm.sh`);
  process.exit(1);
}
const createMarcher = require(MODULE_PATH);
const Module = await createMarcher();

const simd = Module._wasm_nnue_simd();
console.log(`engine:    ${path.relative(ROOT, MODULE_PATH)}  `
          + `simd=${simd} (${{ 0: 'scalar', 1: 'SIMD128', 2: 'AVX2' }[simd] ?? '?'})`);

if (simd !== ref.nnue_info.simd) {
  console.log(`note: reference was built with simd=${ref.nnue_info.simd}. The vector `
            + `width must not change any result - that is what makes this test worth `
            + `running - so a difference below is a real bug, not an expected one.`);
}

const NO_TIME_LIMIT = 1e9;
const FIELDS = Module._wasm_result_fields();

function readResult() {
  // HEAP32 is re-read on every call rather than cached: ALLOW_MEMORY_GROWTH
  // detaches the old view whenever the heap grows, and a search allocates.
  const ptr = Module._wasm_result_ptr();
  return Array.from(Module.HEAP32.subarray(ptr >> 2, (ptr >> 2) + FIELDS));
}

function search(pos, forced, depth) {
  Module._wasm_search(BigInt(pos.p1), BigInt(pos.p2), BigInt(pos.p1k), BigInt(pos.p2k),
                      pos.stm, NO_TIME_LIMIT, depth, forced);
  const r = readResult();
  return {
    move: [r[0], r[1]],
    // rebuilt into the same shape as the Python stats tuple:
    // (depth, max_ply, nodes, hash_entries, eval, evals)
    stats: [r[3], r[4],
            (r[6] >>> 0) * 4294967296 + (r[5] >>> 0),
            (r[8] >>> 0) * 4294967296 + (r[7] >>> 0),
            r[2],
            (r[10] >>> 0) * 4294967296 + (r[9] >>> 0)],
  };
}

// ---------------------------------------------------------------------------
// replay
// ---------------------------------------------------------------------------
const diffs = [];
let checked = 0;

function expect(section, key, got, want) {
  checked++;
  const a = JSON.stringify(got), b = JSON.stringify(want);
  if (a !== b) diffs.push(`${section}[${key}]\n    native: ${b}\n    wasm:   ${a}`);
}

const D = ref.depths;

// nnue_eval first: an evaluation difference should be reported as one, not as a
// downstream node-count mystery
for (const pos of ref.positions) {
  const got = Module._wasm_nnue_eval(BigInt(pos.p1), BigInt(pos.p2),
                                     BigInt(pos.p1k), BigInt(pos.p2k), pos.stm);
  expect('nnue_eval', pos.label, got, ref.nnue_eval[pos.label]);
}

// perft: pure move generation, independent of everything else
for (const pos of ref.positions) {
  for (let d = 1; d <= D.perft; d++) {
    const got = Module._wasm_perft(BigInt(pos.p1), BigInt(pos.p2),
                                   BigInt(pos.p1k), BigInt(pos.p2k), pos.stm, d);
    expect('perft', `${pos.label}@${d}`, got, ref.perft[`${pos.label}@${d}`]);
  }
}

// search: same flat ordered pass as run_probes(), because the table carries over
for (const pos of ref.positions) {
  for (const forced of pos.forced) {
    for (let d = 1; d <= D.search; d++) {
      const got = search(pos, forced, d);
      expect('search', `${pos.label}|f${forced}@${d}`, got,
             ref.search[`${pos.label}|f${forced}@${d}`]);
    }
  }
}

// ---------------------------------------------------------------------------
// speed
// ---------------------------------------------------------------------------
const start = ref.positions.find((p) => p.label === 'start/p1') ?? ref.positions[0];
const t0 = performance.now();
Module._wasm_search(BigInt(start.p1), BigInt(start.p2), BigInt(start.p1k),
                    BigInt(start.p2k), start.stm, NO_TIME_LIMIT, 13, -1);
const elapsed = (performance.now() - t0) / 1000;
const r = readResult();
const nodes = (r[6] >>> 0) * 4294967296 + (r[5] >>> 0);

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------
console.log();
console.log(`speed: depth 13 from the start position, ${nodes.toLocaleString()} nodes `
          + `in ${elapsed.toFixed(2)}s = ${(nodes / elapsed / 1e6).toFixed(2)}M nodes/s`);
console.log();

if (diffs.length) {
  console.log(`${diffs.length} DIFFERENCE(S) in ${checked} probes:`);
  for (const d of diffs.slice(0, 30)) console.log('  ' + d);
  if (diffs.length > 30) console.log(`  ... and ${diffs.length - 30} more`);
  console.log('\nFAIL: the WebAssembly engine does not match the native engine.');
  process.exit(1);
}
console.log(`PASS: ${checked} probes identical to the native engine.`);
