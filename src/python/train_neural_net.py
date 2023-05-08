# Author: Collin Kees

# allows the bot to play against itself and collect data for training
# continusly trains the bot each game and saves each game's data
# to a file for later use in training the neural network
# note: this will use alot of RAM and cpu power so make sure you have enough

# imports
from random import randint, shuffle
from copy import deepcopy 
import sys
import os
from time import process_time
from bitboard_converter import convert_bit_move, convert_matrix_move, convert_to_bitboard, convert_to_matrix
from Board_opperations import Board, check_jump_required, update_board, check_win, check_tie, generate_all_options
import multiprocessing as mp

# import the search engine
sys.path.insert(0, os.getcwd() + "/build/lib.win-amd64-3.9/")
import search_engine
import checkers_NN


# function to turn a game in to the file type needed for training
# input: game data
# output: file type
def game_to_file(game, file_name = "game_data.ds"):
    # create a file
    file_ds = open(file_name, "w")
    # write the game data to the file
    file_ds.write(f"{game[-1]} {game[-2]}\n")
    game = deepcopy(game[:-2])
    
    # write the game data to the file
    for i in range(len(game)):
        file_ds.write(f"{game[i][0]} {game[i][1]} {game[i][2]} {game[i][3]} {game[i][4]}\n")
    # close the file
    file_ds.close()


# function to train the neural network
def train_neural_net(neural_net_file, data_set_file, epochs, batch_size, learning_rate):
    error = checkers_NN.train_net(neural_net_file, data_set_file, epochs, batch_size, learning_rate)

    # print the error
    #print("error: " + str(error))
    return error;
    
def test_neural_net(neural_net_file, data_set_file):
    error = checkers_NN.test_net(neural_net_file, data_set_file)
    return error

# plays one game engine vs engine and returns the game data
# first 6 moves are random to allow the bot to learn from the game
def play_game(save_list):
    # create a board
    board = Board()
    search_time = 0.2
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
        num_moves += 1

        # convert the board to a bitboard
        p1, p2, p1k, p2k = convert_to_bitboard(board.board)

        if num_moves <= 5:
            # if it is we generate a random move
            moves = generate_all_options(board.board, player, False)
            # generate a random move
            best_move = randint(0, len(moves) - 1)
            best_move = moves[best_move]
        
        # if the game is more than 4 moves we use the search engine to find the best move
        else:
            # get the best move
            best_move = search_engine.search_position(p1, p2, p1k, p2k, player, search_time, 50)
            eval_ = best_move[-1][-1]
            depth = best_move[-1][1]

            # convert the best move to a matrix move
            best_move = convert_bit_move(best_move[-2])

            # append the move to the game data
            if depth > 1 and (eval_ > -200 and eval_ < 200):
                game.append(list(convert_to_bitboard(board.board)))
                game[-1].append(eval_ + 100)

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
        # also check if the length of the game is greater than 200 if so call it a tie
        if num_moves > 200:
            tie = True
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
    print("Game concluded...")
    save_list.append(game)
    return game


def conv_gamefiles_to_ds(game_file, data_set_file, num_files, make_testingset = False):
    # make a training and testing file
    train_file = open(data_set_file + "_train.ds", "w")
    if make_testingset:
        test_file = open(data_set_file + "_test.ds", "w")
    test_list = []
    train_list = []

    file_list = []
    for i in range(num_files):
        try:
            file_list.append(open(game_file + str(i) + ".ds", "r"))
        # if for some reason the file does not exist then skip it by creating a blank ds file
        except FileNotFoundError:
            file_list.append(open(game_file + str(i) + ".ds", "w"))
            file_list[i].write("0 0\n")
            file_list[i].close()
            file_list[i] = open(game_file + str(i) + ".ds", "r")


    # get the total number of moves in each file
    num_moves = []
    for i in range(num_files):
        num_moves.append(int(file_list[i].readline().split()[-1]))

    train_moves = 0
    test_moves = 0
    # write every fourth move to the test file
    for i in range(num_files):
        # loop for the number of moves in the file
        for j in range(num_moves[i]):
            # if the move is the fourth move then write it to the test file
            if make_testingset:
                if j % 4 == 3:
                    test_list.append(file_list[i].readline())
                    test_moves += 1
                else:
                    train_list.append(file_list[i].readline())
                    train_moves += 1
            # if the move is not the fourth move then write it to the train file
            else:
                train_list.append(file_list[i].readline())
                train_moves += 1

    # now the number of moves is known write the number of moves to the file
    train_file.write(f"0 {train_moves}\n")
    if make_testingset:
        test_file.write(f"0 {test_moves}\n")

    # write the training data to the file
    for i in range(len(train_list)):
        train_file.write(train_list[i])

    # write the testing data to the file
    if make_testingset:
        for i in range(len(test_list)):
            test_file.write(test_list[i])

    # close the files
    train_file.close()
    if make_testingset:
        test_file.close()

    # for each file in the list
    for i in range(num_files):
        # close the file
        file_list[i].close()

# play a number of games and save the game data to a file (for bulk data creation)
def play_n_games(n):
    # play 8 games at a time and save the data to a file
    manager = mp.Manager()
    
    # variable to store the index of the file being written to
    file_index = 2370

    for i in range(n // 10):
        # create a list to store the game data
        save_list = manager.list()
        # create a list of processes
        processes = []

        # play 10 games
        for j in range(10):
            # create a process to play the game
            p = mp.Process(target=play_game, args=(save_list,))
            # start the process
            p.start()
            # add the process to the list
            processes.append(p)

        # wait for all the processes to finish
        for p in processes:
            p.join()
            
        # save the game data to a file
        for j in range(len(save_list)):
            # only save the game if it is not a tie
            game_to_file(save_list[j], "data_set/old_eval/training_data" + str(file_index) + ".ds")
            file_index += 1


# function to allow the bot to play against itself and continusly train the neural network
def unsupervised_training(neural_net_file, data_set_file, epochs, learning_rate):
    # create a list to store the game data
    manager = mp.Manager()
    training_list = []
    p1_win = 0
    p2_win = 0
    tie = 0

    # we want 10 games per batch of training (only include two tie game) and 4 games of test data every 4 batches
    while True:
        games = []
        # prepare 8 processes to play the games
        game_list = manager.list()
        for i in range(8):
            p = mp.Process(target=play_game, args=(game_list,))
            p.start()
            games.append(p)

        # wait for the processes to finish
        for i in range(8):
            games[i].join()
        
        # check the outcome of the game and add it to the list if it the kind we want
        for i in range(8):
            game = game_list[i]
            if game[-1] == 1 and p1_win < 4:
                p1_win += 1
                training_list.append(game)
            elif game[-1] == 2 and p2_win < 4:
                p2_win += 1
                training_list.append(game)
            elif game[-1] == 0 and tie < 2:
                tie += 1
                training_list.append(game)
            if p1_win + p2_win + tie == 10:
                break

        # if we have enough games to train on then train the neural network
        if (p1_win + p2_win + tie) >= 10:
            # first write the training data to game files
            temp_file = "data_set/temp/temp"
            for i in range(len(training_list)):
                game_to_file(training_list[i], temp_file + str(i) + ".ds")

            # now convert the game files to a data set
            conv_gamefiles_to_ds(temp_file, data_set_file, len(training_list))

            # now test the neural network on the data set
            error_end = abs(test_neural_net(neural_net_file, data_set_file + "_train"))

            print(f"Error: {error_end}\n")

            print("training neural network...")
            train_neural_net(neural_net_file, data_set_file + "_train", epochs, learning_rate)

            # now reset the game list and the number of games
            training_list = []
            p1_win = 0
            p2_win = 0
            tie = 0

            # append the error to a file so we can see how it is changing
            with open("error.txt", "a") as f:
                f.write(f"{error_end}\n")
                f.close()


# main function
def main():

    # for playing one game
    #game = play_game()
    # convert the game data to a file
    #game_to_file(game)
    #print("Data generated! exiting...")

    # write the games to a training and testing file
    #conv_gamefiles_to_ds("data_set/old_eval/training_data", "data_set/data_set", 2080, True)
    #exit(0)

    # train the neural network on itself
    #unsupervised_training("neural_net/neural_net", "data_set/self_play_data/game_data", 100, 0.01)
    #print("training sucessful exiting...")
    #exit(1)

    # play a number of games and save the game data to a file (for bulk data creation)
    #play_n_games(10000)
    #exit(0)

    # test the neural network 
    print("testing neural network...")
    error_start = abs(test_neural_net("neural_net/neural_net", "data_set/data_set_test"))

    # train the neural network using the data set
    print("training neural network...")
    epochs_out = 32
    epochs = 32
    batch_size = 32
    learning_rate = 0.1
    for i in range(epochs_out):
        train_neural_net("neural_net/neural_net", "data_set/data_set_train", epochs, batch_size, learning_rate)
        print(abs(test_neural_net("neural_net/neural_net", "data_set/data_set_test")))

    # test the neural network on the test data set
    print("testing neural network...")
    error_end = abs(test_neural_net("neural_net/neural_net", "data_set/data_set_test"))

    # test the neural network on the training data set
    print("testing neural network...")
    error_end_training = abs(test_neural_net("neural_net/neural_net", "data_set/data_set_train"))

    print()
    print("error test_init: " + str(error_start))
    print("error test_end: " + str(error_end))
    print("error train_end: " + str(error_end_training))

    


# call the main function
if __name__ == "__main__":
    main()

