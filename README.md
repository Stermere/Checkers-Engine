# Checkers-Engine
A checkers engine that uses the Minmax algorithm with alpha-beta pruning to find the best move for a given board position. This implementation of the engine also uses Monty-Carlo search to break ties and provide more variation when playing against it. 

The program also accepts two command-line arguments if you wish to have the bot play against itself.
The arguments default to '1 human' where 1 is the search time and human can be either bot or human.

This branch has the purpose of replacing the evaluation function with a neural network evaluation to improve its positional play, along with this the moty carlo search aspect of the program will also be removed as it causes more issues than it solves, while I am at it some refactoring of Main is probably not a bad idea.
