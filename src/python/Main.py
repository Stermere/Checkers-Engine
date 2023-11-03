# The driver program that interacts with the board search algorithms and GUI 
# uses minimax search with alpha beta pruning to find the best move for the current player
# developed by Collin Kees

import sys
import os
from time import process_time
from bitboard_converter import convert_bit_move, convert_matrix_move, convert_to_bitboard, convert_to_matrix
from Board_opperations import Board, check_jump_required, update_board, check_win, check_tie
import multiprocessing as mp

# import the search engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib.win-amd64-cpython-310')))
import search_engine


# calls the board search algorithm and parses the results to the GUI 
def start_search(board : list, player : int, p_time : int, ply : int, return_dict): 
    p1, p2, p1k, p2k = convert_to_bitboard(board)
    results = search_engine.search_position(p1, p2, p1k, p2k, player, p_time, ply)

    # update the return dict
    return_dict["depth"] = results[1][0]
    return_dict["depth_extended"] = results[1][1]
    return_dict["leafs"] = results[1][2]
    return_dict["eval"] = results[1][4]
    return_dict["hashes"] = results[1][3]

    # save the object in a touple interpretation to sent it back to the main thread
    return_dict["minmax"] = results


# start the processing of the minimax tree search on a new thread to allow the GUI to run
def start_processing(board : list, state : int, p_time, gui: object, ply : int):
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
    minmax = mp.Process(target=start_search, args=(board, state, p_time, ply, return_dict, ))

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
    ply = 50
    if len(args) == 1:
        return bot, time, ply
    search_time = args[2]
    mode = args[1]
    ply = int(args[3])
    if mode == "bot":
        bot = True
    if search_time != "0":
        time = float(search_time)

    return bot, time, ply


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
    turns = 0
    game_history = []

    # pars args
    BOT_PLAYING, P_TIME, PLY = parse_args(args)

    # main loop
    while True:
        # check for quit 
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

        # update any variables for data gathering
        turns += 1
        game_history.append(convert_to_bitboard(board.board))

        # player 1's turn
        if player == 1:

            # for bot on bot
            if BOT_PLAYING:
                # the bot
                player3 = start_processing(board.board, 1, P_TIME, gui, PLY)

                # update the board with the bots chosen move
                chosen_move = convert_bit_move(player3[0])
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
            player2 = start_processing(board.board, 2, P_TIME, gui, PLY)

            # update the board with the bots chosen move
            chosen_move = convert_bit_move(player2[0])
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
        tie = check_tie(game_history)
        if win == 1:
            # make another pygame loop for showing the win message
            start_time = process_time()
            gui.win_messsage = "Player one wins!"
            while process_time() - start_time < 2:
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
            turns = 0
            game_history = []

        elif win == 2:
            # make another pygame loop for showing the win message
            start_time = process_time()
            gui.win_messsage = "Player two wins!"
            while process_time() - start_time < 2:
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
            turns = 0
            game_history = []

        elif tie:
            # make another pygame loop for showing the win message
            start_time = process_time()
            gui.win_messsage = "Tie!"
            while process_time() - start_time < 2:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                gui.draw()
                clock.tick(60)
            print('Tie game')
            board.reset_board(gui)
            gui = Gui(board.board, size, clock, screen, 1)
            player = 1
            turns = 0
            game_history = []


if __name__ == '__main__':
    # import modules needed for the main thread
    import pygame
    from Gui import Gui

    main(sys.argv)