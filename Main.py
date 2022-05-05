# The driver program that interacts with the board search algorithms and GUI 
# uses minimax and monty carlo tree searches to find a optimial move
# developed by Collin Kees


import sys
import os
from time import process_time
from bitboard_converter import convert_bit_move, convert_matrix_move, convert_to_bitboard, convert_to_matrix
from Board_opperations import Board, check_jump_required, update_board, check_win, check_tie
import multiprocessing as mp

# import the search engine
sys.path.insert(0, os.getcwd() + "/build/lib.win-amd64-3.9/")
import search_engine


# calls the board search algorithm and parses the results to the GUI 
def start_search(board : list, player : int, p_time : int, return_dict): 
    p1, p2, p1k, p2k = convert_to_bitboard(board)
    # a depth of 25 is not likly to be hit by the search at the time of writing but if we do hit it
    # we will need to increase the depth here
    p_depth = 25;
    results = search_engine.search_position(p1, p2, p1k, p2k, player, p_time, p_depth)

    #print(results)

    # update the return dict (stops montycarlo if a search is terminated before the time constraint)
    return_dict["depth"] = results[-1][0]
    return_dict["leafs"] = results[-1][1]
    return_dict["eval"] = results[-1][3]
    return_dict["hashes"] = results[-1][2]

    # save the object in a touple interpretation to sent it back to the main thread
    return_dict["minmax"] = results

    print(results)


# start the processing of the minimax tree search on a new thread to allow the GUI to run
def start_processing(board : list, state : int, p_time, gui: object):
    manager = mp.Manager()
    return_dict = manager.dict()

    # initialize some values that will be needed
    return_dict["minmax"] = None
    return_dict["best_move"] = None
    return_dict["depth"] = 0
    return_dict["eval"] = 0
    return_dict["leafs"] = 0
    return_dict["hashes"] = 0

    # prepare the process
    minmax = mp.Process(target=start_search, args=(board, state, p_time, return_dict, ))

    # start the processes
    minmax.start()

    # update the values on the board as they are processed and continue to draw the board
    while return_dict["minmax"] == None:
        clock = pygame.time.Clock()
        # make another pygame loop for while the bot is thinking
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            gui.draw()
            clock.tick(60)
        gui.draw()
    # once minimax search ends montycarlo should end as well
    minmax.join()

    gui.update_params(return_dict)

    # update hashes, leaves, and prunned branches
    return return_dict["minmax"]


# parses the args and returns the correct data
def parse_args(args):
    time = 1
    bot = False
    if len(args) == 1:
        return bot, time
    search_time = args[1]
    mode = args[2]
    if mode == "bot":
        bot = True
    if search_time != "0":
        time = float(search_time)

    return bot, time


# start the main loop of the game
def main(args) -> None:
    size = (1000, 800)
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Checkers')
    pygame.init()
    player = 1
    board = Board()
    gui = Gui(board.board, size, clock, screen, 1) # must be initialized regardless of if a human is playing or not

    # variables to keep track of some data that is integral to the game (allows for tie detection and data collection for training the NN)
    p1_wins = 0
    p2_wins = 0
    turns = 0
    moves_at_turn = []
    board_at_turn = []
    pieces_at_turn = []

    # pars args
    BOT_PLAYING, P_TIME = parse_args(args)

    # main loop
    while True:
        # check for quit 
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

        # update any variables for data gathering : TODO add more
        turns += 1

        # player 1's turn
        if player == 1:

            # Note: only have one ennabled or bad stuff happens
            # for bot on bot
            if BOT_PLAYING:
                # the bot
                player3 = start_processing(board.board, 1, P_TIME, gui)

                # update the board with the bots chosen move
                chosen_move = convert_bit_move(player3[-2])
                turn = update_board(chosen_move[0], chosen_move[1], board.board)
                if turn == True and check_jump_required(board.board, player, chosen_move[1]):
                    turn = True
                else:
                    turn = False

                gui.blue_blocks = [chosen_move[0], chosen_move[1]]
            # for human player
            else:
                turn, chosen_move = gui.choose_action()
                gui.blue_blocks = [chosen_move[0], chosen_move[1]]

            if turn:
                continue

        # player 2's turn
        else:
            # the bot
            player2 = start_processing(board.board, 2, P_TIME, gui)

            # update the board with the bots chosen move
            chosen_move = convert_bit_move(player2[-2])
            turn = update_board(chosen_move[0], chosen_move[1], board.board)
            if turn == True and check_jump_required(board.board, player, chosen_move[1]):
                turn = True
            else:
                turn = False

            # highlight the move that the bot chose
            gui.red_blocks += chosen_move[0], chosen_move[1]

            if turn:
                continue

        # change the current player
        if player == 1:
            player = 2
            gui.red_blocks = []
        else:
            player = 1
            gui.blue_blocks = []


        # check for a win
        win = check_win(board.board, player)
        tie = check_tie([])
        if win == 1:
            # make another pygame loop for showing the win message
            start_time = process_time()
            gui.win_messsage = "Player one wins!"
            while process_time() - start_time < 3:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                gui.draw()
                clock.tick(60)
            print('player one wins')
            board.reset_board(gui)
            gui = Gui(board.board, size, clock, screen, 1)
            player = 1
            p1_wins += 1
            turns = 0

        elif win == 2:
            # make another pygame loop for showing the win message
            start_time = process_time()
            gui.win_messsage = "Player two wins!"
            while process_time() - start_time < 3:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                gui.draw()
                clock.tick(60)
            print('player two wins')
            board.reset_board(gui)
            gui = Gui(board.board, size, clock, screen, 1)
            player = 1
            p2_wins += 1
            turns = 0


# TODO put the logic to start the threads in its own file so this is not gross and bad
# this seems to fix the problem with threads doing weird things when the program is packaged with pyinstaller
mp.freeze_support()

if __name__ == '__main__':
    # import modules needed for the main thread
    import pygame
    from Gui import Gui

    main(sys.argv)