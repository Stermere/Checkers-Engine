from distutils.core import setup, Extension

def main():
    setup(name="checkers_NN",
          version="1.0.0",
          description="Python interface for interfacing with the neural net",
          author="Collin Kees",
          author_email="Collin@kees.net",
          ext_modules=[Extension("chechers_NN", ["Checkers_NN.c"])])

if __name__ == "__main__":
    main()