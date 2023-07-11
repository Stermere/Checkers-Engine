from setuptools import setup, Extension

def main():
    setup(
        name="search_engine",
        version="1.1.3",
        description="Python interface for searching a given checkers board",
        author="Collin Kees",
        author_email="Collin@kees.net",
        ext_modules=[Extension("search_engine", ["src/engine/board_search.c"])]
    )

if __name__ == "__main__":
    main()