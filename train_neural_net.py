# Author: Collin Kees

# allows the bot to play against itself and collect data for training
# continusly trains the bot each game and saves each game's data
# to a file for later use in training the neural network
# note: this will use alot of RAM and cpu power so make sure you have enough

# imports
from cmath import log10
from random import randint
import sys
import os
from time import process_time
from bitboard_converter import convert_bit_move, convert_matrix_move, convert_to_bitboard, convert_to_matrix
from Board_opperations import Board, check_jump_required, update_board, check_win, check_tie, generate_all_options
import multiprocessing as mp

# import the search engine
sys.path.insert(0, os.getcwd() + "/build/lib.win-amd64-3.9/")
import search_engine


# function to turn a game in to the file type needed for training
# input: game data
# output: file type
def game_to_file(game, file_name = "game_data.ds"):
    # create a file
    file = open(file_name, "w")
    # write the game data to the file
    file.write(f"{game[-1]} {game[-2]}\n")
    for i in range(len(game) - 2):
        # calculate the true eval at this point
        if game[-1] == 1:
            eval_ = 1
            eval_ = eval_ - ((1 / game[-2]) * (game[-2] - (i + 1)))
        elif game[-1] == 2:
            eval_ = -1
            eval_ = eval_ + ((1 / game[-2]) * (game[-2] - (i + 1)))
        else:
            eval_ = 0

        # write the game data to the file
        file.write(f"{game[i][0]} {game[i][1]} {game[i][2]} {game[i][3]} {eval_}\n")
    # close the file
    file.close()


# plays one game engine vs engine and returns the game data
# first 3 moves are random to allow the bot to learn from the game
def play_game():
    # create a board
    board = Board()
    search_time = 2.5
    # create a list to store the game data
    game = []
    num_moves = 0
    # a final state of 0 is tie 1 is player 1 win 2 is player 2 win
    final_state = -1

    player = 1

    # give an indication of the game being played
    print("Game in progress...")

    # while the game is not over
    while final_state == -1:
        game.append(convert_to_bitboard(board.board))
        
        num_moves += 1

        # convert the board to a bitboard
        p1, p2, p1k, p2k = convert_to_bitboard(board.board)

        # since we want the first three moves random we check if the game is less than 3 moves
        if num_moves <= 3:
            # if it is we generate a random move
            moves = generate_all_options(board.board, player, False)
            # generate a random move
            best_move = randint(0, len(moves) - 1)
            best_move = moves[best_move]
        
        # if the game is more than 3 moves we use the search engine to find the best move
        else:
            # get the best move
            best_move = search_engine.search_position(p1, p2, p1k, p2k, player, search_time, 25)

            # convert the best move to a matrix move
            best_move = convert_bit_move(best_move[-2])

        # update the board
        update_board(best_move[0], best_move[1], board.board)

        # check if the move is a jump and if that piece can jump
        if (abs(best_move[0][0] - best_move[1][0])) == 2:
            if check_jump_required(board.board, player, best_move[1]):
                continue
        
        # if another jump is not required then check if the game is over and change the player
        player = 3 - player

        # see if the game is over
        win = check_win(board.board, player)
        tie = check_tie(game)
        if win == 1:
            final_state = 1
        elif win == 2:
            final_state = 2
        elif tie:
            final_state = 0

    # add the final state, and the number of moves to the game data
    game.append(num_moves)
    game.append(final_state)

    # return the game data
    print("Game concluded saving data...")
    return game


# main function
def main():
    # generate a game data file by playing a game
    #for i in range(1000):
    #    game = play_game()
    #    game_to_file(game, "data_set/pre/pre_training_data" + str(i) + ".ds")
#
    # for playing one game
    #game = play_game()
    # convert the game data to a file
    #game_to_file(game)
    #print("Data generated! exiting...")

    # train the neural network using the data set
    print("Training the neural network...")
    
    

# call the main function
if __name__ == "__main__":
    main()

