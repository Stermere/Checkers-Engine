import sys
from setuptools import setup, Extension

def main():
    # Pass --old to build the extension as "search_engine_old" instead of "search_engine"
    old_build = "--old" in sys.argv
    if old_build:
        sys.argv.remove("--old")

    module_name = "search_engine_old" if old_build else "search_engine"

    # MSVC optimization flags. setuptools' default Release build already includes /O2,
    # but we make the optimizations explicit and add a few that meaningfully help
    # this engine (whole-program-opt, intrinsic generation, fast FP, NDEBUG).
    common_flags = [
        "/O2", "/Oi", "/Ot", "/Oy", "/GL", "/Gw", "/GS-",
        "/fp:fast", "/DNDEBUG",
        "/D", "PYTHON",
    ]
    if old_build:
        common_flags.extend(["/D", "OLD_ENGINE"])

    common_link = ["/LTCG"]

    setup(
        name=module_name,
        version="2.0.0",
        description="Python interface for searching a given checkers board",
        author="Collin Kees",
        author_email="Collin@kees.net",
        ext_modules=[Extension(
            module_name,
            ["src/engine/board_search.c"],
            extra_compile_args=common_flags,
            extra_link_args=common_link,
        )]
    )

if __name__ == "__main__":
    main()
