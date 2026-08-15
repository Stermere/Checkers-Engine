# Checkers-Engine (Marcher Engine)

A strong checkers engine written in C, with an nnue and an endgame
tablebase. It runs natively as a Python extension, and in the browser as
WebAssembly.

**[Play it here](https://stermere.github.io/Marcher_Engine_GUI/)** — no server,
no install. The engine is compiled to WebAssembly and runs entirely on your
machine.

There is also a desktop app for offline play, which bundles the endgame
tablebase — see [the GUI repo](https://github.com/Stermere/Marcher_Engine_GUI#desktop-app).

## Strength

About **-50 Elo against Kingsrow (x64) 1.19e at 0.5s per move**, which is a very
strong reference. Both engines were configured to use a single thread and a 64 MB
transposition table when this measurement was made. The browser build measures **1.8x slower than the native build** (1.7M vs 3.1M nodes/s) so expect slightly lower strength in the browser.

## Training data

The shipped NNUE learned from about 22M positions, and Kingsrow(x64) 1.19e was
used as an opponent in the training pipeline.

No Kingsrow code is included or redistributed here — `src/python/nnue/kingsrow.py`
loads it at runtime through the public CheckerBoard engine DLL API, from a local
CheckerBoard install you supply yourself.

## Play locally

```bash
git clone https://github.com/Stermere/Checkers-Engine
cd Checkers-Engine
initVenv.bat        # virtualenv + requirements
build.bat           # compile the C into a Python module
run.bat             # serve the web app on localhost
```

Local play is stronger than the web app especially if given a larger time budget by modifying `Marcher_Engine_GUI\flask-server\server.py` line 8.

## Build for the browser

Needs the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html).

```bash
source ~/emsdk/emsdk_env.sh
bash src/wasm/build_wasm.sh     # SIMD build, scalar fallback, verification build
node src/wasm/verify_wasm.mjs   # must print PASS
bash src/wasm/copy_to_gui.sh    # drop it into the GUI checkout to run it
```

Pushing to `main` builds, verifies and publishes this automatically — see
`.github/workflows/engine-wasm.yml`.

## Testing

Two gates, both equality tests rather than similarity tests:

```bash
python src/python/verify_identical.py --full   # two native builds must agree
node src/wasm/verify_wasm.mjs                  # wasm must agree with native
```

Both compare perft, static evaluation, and fixed-depth search — including
**node counts**, which is the load-bearing part. Equal node counts at equal
depth mean two builds walked the identical tree, cutoffs and all.

Before changing the engine, build a baseline to compare against *first*:

```bash
python src/python/Package_engine.py build --force --name search_engine_ref
```

`src/python/bot_vs_bot.py` runs engine-vs-engine matches for Elo.

## Layout

```
src/engine/     the engine (C)
src/python/     build script, tooling, and the NNUE training pipeline
src/wasm/       WebAssembly build and verification
db/             endgame tablebase (generated, not in git)
Marcher_Engine_GUI/   the web app (submodule)
```

## License

MIT — see [LICENSE](LICENSE). Use it in anything, including commercially; just
keep the copyright notice. The GUI submodule is MIT as well.
