from distutils.core import setup, Extension

def main():
    setup(name="board_search",
          version="1.0.0",
          description="Python interface for searching a given checkers board",
          author="Collin Kees",
          author_email="Collin@kees.net",
          ext_modules=[Extension("board_search", ["board_search.c"])])

if __name__ == "__main__":
    main()