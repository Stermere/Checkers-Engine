#!/usr/bin/env bash
# Put a local wasm build where the dev server can serve it.
#
# CI does this from release assets; this is the same thing for working on the
# browser engine by hand. Everything it writes is gitignored in the GUI repo -
# these are build outputs, not sources.
#
# After running this:
#     cd Marcher_Engine_GUI/client && REACT_APP_ENGINE=wasm npm start
#
# (plain `npm start` stays on Flask - see src/engine/EngineClient.js.)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST="$ROOT/src/wasm/dist"
GUI="$ROOT/Marcher_Engine_GUI/client/public/engine"

if [ ! -f "$DIST/marcher.js" ]; then
    echo "no build in $DIST - run: bash src/wasm/build_wasm.sh" >&2
    exit 1
fi

mkdir -p "$GUI"
cp "$DIST"/marcher.js "$DIST"/marcher.wasm "$GUI/"
cp "$DIST"/marcher-scalar.js "$DIST"/marcher-scalar.wasm "$GUI/"
echo "engine    -> $GUI"

# The tablebase is optional: without it the engine plays fine and is only
# weaker in endings, so a missing db/ is a note rather than a failure.
if [ -d "$ROOT/db" ]; then
    mkdir -p "$GUI/db"
    cp "$ROOT"/db/*.bin "$GUI/db/" 2>/dev/null || true
    python "$ROOT/src/wasm/make_db_manifest.py" "$GUI/db"
    echo "tablebase -> $GUI/db"
else
    echo "no db/ at $ROOT - skipping tablebase (the engine will still play)"
fi
