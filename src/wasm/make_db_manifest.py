"""Write the index the browser needs to fetch endgame tablebase slices.

The engine used to find its slices by trying to fopen every material
combination and ignoring the misses. A browser cannot do that cheaply - every
miss is a real HTTP request and a 404 - so the web build is told up front what
exists.

Each entry carries the material split parsed out of the filename and the exact
byte count. The byte count is not decoration: engine-worker.js passes it to
endgame_db_alloc_slice, which refuses a slice whose length disagrees with what
the indexing says it must be. That is what stops a truncated download from
being indexed as though it were whole and answering "proven win" out of
uninitialised memory.

Usage:
    python src/wasm/make_db_manifest.py [db_dir]      # writes db_dir/manifest.json
"""

import json
import os
import re
import sys

NAME_RE = re.compile(r'^(wld|dtw)_(\d)(\d)(\d)(\d)\.bin$')


def build(db_dir):
    entries = []
    for name in sorted(os.listdir(db_dir)):
        m = NAME_RE.match(name)
        if not m:
            continue
        kind, m1, k1, m2, k2 = m.groups()
        entries.append({
            "name": name,
            "dtw": kind == "dtw",
            "m1": int(m1), "k1": int(k1), "m2": int(m2), "k2": int(k2),
            "bytes": os.path.getsize(os.path.join(db_dir, name)),
        })
    return entries


def main():
    db_dir = sys.argv[1] if len(sys.argv) > 1 else "db"
    if not os.path.isdir(db_dir):
        raise SystemExit(f"no such directory: {db_dir}")

    entries = build(db_dir)
    out = os.path.join(db_dir, "manifest.json")
    with open(out, "w") as f:
        json.dump(entries, f, separators=(",", ":"))

    wld = sum(e["bytes"] for e in entries if not e["dtw"])
    dtw = sum(e["bytes"] for e in entries if e["dtw"])
    print(f"{len(entries)} slices -> {out}")
    print(f"  wld {wld / 1e6:.2f} MB (loaded first)")
    print(f"  dtw {dtw / 1e6:.2f} MB (background)")


if __name__ == "__main__":
    main()
