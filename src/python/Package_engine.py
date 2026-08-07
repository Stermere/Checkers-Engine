import glob
import os
import sys
from setuptools import setup, Extension

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "engine")


def engine_dependencies():
    """Every file that goes into the build, not just the one source."""
    return sorted(glob.glob(os.path.join(ENGINE_DIR, "*.c"))
                  + glob.glob(os.path.join(ENGINE_DIR, "*.h")))


def invalidate_stale(module_name, depends):
    """Delete build outputs older than any dependency."""
    newest = max((os.path.getmtime(p) for p in depends if os.path.exists(p)),
                 default=0.0)

    def stale_outputs(pattern):
        return [p for p in glob.glob(pattern) if os.path.getmtime(p) < newest]

    removed = []

    # object files are pure build artifacts and cost seven seconds to rebuild
    for obj in stale_outputs(os.path.join("build", "temp.*", "Release", "src",
                                          "engine", "board_search.obj")):
        os.remove(obj)
        removed.append(obj)

    for pyd in stale_outputs(os.path.join("build", "lib.*",
                                          f"{module_name}.*.pyd")):
        backup = pyd + ".prev"
        try:
            if os.path.exists(backup):
                os.remove(backup)
            os.replace(pyd, backup)
        except OSError as e:
            # held open by a running process is the common case; say so rather
            # than building something that is not what was asked for
            raise SystemExit(
                f"{pyd} is out of date but could not be replaced ({e}).\n"
                f"Something is holding it open - a data generation run, a "
                f"Python session that imported it, or Dropbox mid-sync. Close "
                f"it, or build under another name with --name.")
        removed.append(f"{pyd} -> {os.path.basename(backup)}")

    return removed


def main():
    # Pass --old to build the extension as "search_engine_old" instead of "search_engine"
    old_build = "--old" in sys.argv
    if old_build:
        sys.argv.remove("--old")

    module_name = "search_engine_old" if old_build else "search_engine"

    # --name NAME builds the same source under a different module name, so a
    # variant can be built and imported alongside the engine that is currently
    # in use - measuring a change without stopping a running data generation, or
    # keeping a slow NNUE_VERIFY_INCREMENTAL build next to the fast one. The
    # .pyd lands in the same build directory, so engine_iface finds it too.
    renamed = False
    if "--name" in sys.argv:
        i = sys.argv.index("--name")
        module_name = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        renamed = True

    # --define NAME=VALUE passes an extra preprocessor definition through to the
    # compiler (repeatable). NNUE_VERIFY_INCREMENTAL=1 is the one that matters.
    extra_defines = []
    while "--define" in sys.argv:
        i = sys.argv.index("--define")
        extra_defines.append(sys.argv[i + 1])
        del sys.argv[i:i + 2]

    # --no-avx2 drops the AVX2 target for one build, which is how the SSE2 path
    # gets tested without editing this file.
    want_avx2 = "--no-avx2" not in sys.argv
    if not want_avx2:
        sys.argv.remove("--no-avx2")

    # Profile guided optimization was implemented here as a --pgo-gen/--pgo-use
    # two pass build, measured, and removed: it was worth -2% (1.23x vs 1.25x
    # against the same baseline, depth 15 over 12 openings), i.e. nothing outside
    # the noise. The reason appears to be that there is nothing left for it to
    # learn - the engine is a single translation unit already built with /GL and
    # /LTCG, so the optimizer sees the whole program, and what the search waits on
    # is memory and dependency chains rather than mispredicted branches. See
    # src/python/nnue/STATUS.md for the numbers.

    # Two flag sets, because which compiler setuptools reaches for is a property
    # of the platform. The engine is a single translation unit, so the flags that
    # matter are whole-program optimization and the vector target; the rest is
    # the ordinary release set (setuptools' own Release build already implies
    # most of it, but being explicit means a build is reproducible from here).
    #
    # Targeting AVX2 is what lets src/engine/nnue.c use 256 bit integer vectors:
    # a compiler only defines __AVX2__ when told the target may use AVX2, and
    # without it the network falls back to the 128 bit SSE2 path - still
    # bit-identical, just half the lanes in the feature transformer, which is the
    # hot loop. The cost is that the result will not run on a CPU older than
    # roughly 2013 (Haswell / Excavator); it dies with an illegal instruction
    # rather than running slowly. Pass --no-avx2 if that matters, and check
    # nnue_info()["simd"] for which path was actually compiled
    # (2 = AVX2, 1 = SSE2, 0 = scalar).
    if sys.platform == "win32":
        common_flags = [
            "/O2", "/Oi", "/Ot", "/Oy", "/GL", "/Gw", "/GS-",
            "/fp:fast", "/DNDEBUG",
            "/D", "PYTHON",
        ]
        if want_avx2:
            common_flags.append("/arch:AVX2")
        common_link = ["/LTCG"]
        def define(d):
            return ["/D", d]
    else:
        # gcc / clang. No /fp:fast equivalent is needed: the network is integer
        # arithmetic on purpose, precisely so that no floating point model can
        # change what it computes.
        common_flags = [
            "-O3", "-fomit-frame-pointer", "-flto", "-DNDEBUG", "-DPYTHON",
        ]
        if want_avx2:
            common_flags.append("-mavx2")
        common_link = ["-flto"]
        def define(d):
            return [f"-D{d}"]

    # OLD_ENGINE selects the reference build: the previous search behavior and the
    # handcrafted evaluation. The default build is the current engine, which uses
    # the network in src/engine/nnue.c.
    if old_build:
        common_flags.extend(define("OLD_ENGINE"))
    if renamed:
        # board_search.c derives both the module name string and the PyInit_
        # symbol from this, so the two always agree
        common_flags.extend(define(f"ENGINE_MODULE_NAME={module_name}"))
    for d in extra_defines:
        common_flags.extend(define(d))

    depends = engine_dependencies()
    for out in invalidate_stale(module_name, depends):
        print(f"stale, removed: {out}")

    setup(
        name=module_name,
        version="2.0.0",
        description="Python interface for searching a given checkers board",
        author="Collin Kees",
        author_email="Collin@kees.net",
        ext_modules=[Extension(
            module_name,
            ["src/engine/board_search.c"],
            depends=depends,
            extra_compile_args=common_flags,
            extra_link_args=common_link,
        )]
    )

if __name__ == "__main__":
    main()
