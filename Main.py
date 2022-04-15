# The driver program that interacts with the board search algorithms and GUI 
# uses minimax and monty carlo tree searches to find a optimial move
# developed by Collin Kees

from copy import deepcopy
from time import process_time
from bitboard_converter import convert_bit_move, convert_to_bitboard
from Monty_carlo import Monty_carlo
from Board_opperations import Board, check_jump_required, update_board, check_win, generate_all_options
import search_engine

if __name__ == '__main__':
    import pygame
    import multiprocessing as mp
    from Gui import Gui

# allows processing of wins until time runs out
def dynamic_win(board : list, last_move : tuple, state : int, return_dict, p_time : float):
    # get the child that is the new board state
    time = process_time()
    montycarlo = Monty_carlo(state, board)
    
    wins_per_search = 10
    while process_time() - time < p_time and return_dict["done"] != True:
        prevp1 = montycarlo.total_p1_win
        prevp2 = montycarlo.total_p2_win
        montycarlo.find_n_wins(wins_per_search)
        return_dict["wins1"] += montycarlo.total_p1_win - prevp1
        return_dict["wins2"] += montycarlo.total_p2_win - prevp2

    # save the object in a touple interpretation to sent it back to the main thread
    return_data = return_dict["montycarlo"]
    return_data.append((last_move, montycarlo.total_p1_win, montycarlo.total_p2_win))
    return_dict["montycarlo"] = return_data

# allows for processing to be stopped at a precice time without losing speed 
def dynamic_depth(board : list, player : int, p_time : int, return_dict): 
    p1, p2, p1k, p2k = convert_to_bitboard(board)
    # a depth of 25 is almost impossible to reach so we will limit it to that if it does reach it
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
    return_dict["done"] = True


# merge the weights from both three searches to resolve any ties in the minimax search
def merge_monty_and_minmax(montycarlo, minmax, player):
    # if the player is 2 then the weights are reversed
    options = []
    best_move = minmax[0][0]
    # get the best moves from the minmax search
    for i in range(len(minmax)):
        if minmax[i][0] >= best_move:
            best_move = minmax[i][0]
            options.append((minmax[i][0], minmax[i][1], minmax[i][1]))

    # merge the weights from both searches
    for i in range(len(options)):
        for j in range(len(montycarlo)):
            if montycarlo[j][1] == convert_bit_move((options[i][1], options[i][2]))[0] and montycarlo[j][0][0] == convert_bit_move((options[i][1], options[i][2]))[0]:

                # merge logic 
                if player == 2:
                    ratio = (montycarlo[j][2] + 1) / (montycarlo[j][1] + 1)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_move = (options[i][1], montycarlo[i][2])
                else:
                    ratio = (montycarlo[j][1] + 1) / (montycarlo[j][2] + 1)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_move = (options[i][1], montycarlo[i][2])

    # update the best move attribute
    minmax[-2] = best_move


# start the processing of the minimax and Monty Carlo tree search
def start_processing(board : list, state : int, p_time, gui: object):
    manager = mp.Manager()
    return_dict = manager.dict()

    # initialize some values that will be needed
    return_dict["minmax"] = None
    return_dict["best_move"] = None
    return_dict["montycarlo"] = []
    return_dict["depth"] = 0
    return_dict["eval"] = 0
    return_dict["leafs"] = 0
    return_dict["hashes"] = 0
    return_dict["wins1"] = 0
    return_dict["wins2"] = 0

    # generate the first depth of the montycarlo tree to split up processing
    invert_state = lambda x : 1 if x == 2 else 2
    jump = False
    if check_jump_required(board, state):
        next_layer_boards = generate_all_options(board, state, True)
        jump = True
        
    else:
        next_layer_boards = generate_all_options(board, state, False)      

    # prepare the processes
    minmax = mp.Process(target=dynamic_depth, args=(board, state, p_time, return_dict, ))
    montycarlo_list = []

    for child in next_layer_boards:
        board_ = deepcopy(board)
        update_board(child[0], child[1], board_)
        state_ = check_jump_required(board_, state, child[1])
        if state_ and jump:
            state_ = state
        else:
            state_ = invert_state(state) 
        montycarlo_list.append(mp.Process(target=dynamic_win, args=(board_, child, state_, return_dict, p_time, )))

    # start the processes
    return_dict["done"] = False
    minmax.start()
    for child in montycarlo_list:
        child.start()

    # update the values on the board as they are processed and continue to draw the board
    while return_dict["done"] == False:
        # make another pygame loop for while the bot is thinking
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            gui.draw()
    # once minimax search ends montycarlo should end as well
    minmax.join()
    for child in montycarlo_list:
        child.join()

    gui.update_params(return_dict)

    merge_monty_and_minmax(return_dict["montycarlo"], return_dict["minmax"], state)

    # update hashes, leaves, and prunned branches
    return return_dict["minmax"], (return_dict["wins1"], return_dict["wins2"])


def main() -> None:
    size = (1000, 800)
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Checkers')
    pygame.init()
    player = 1
    board = Board()
    gui = Gui(board.board, size, clock, screen, 1) # must be initialized regardless of if a human is playing or not

    # variables to keep track of some data that is integral to the game (allows for moves to not be repeated more than twice)
    p1_wins = 0
    p2_wins = 0
    turns = 0
    moves_at_turn = []
    board_at_turn = []
    pieces_at_turn = []


    # True to have the game play against itself
    BOT_PLAYING = False
    P_TIME = 2

    while True:
        # check for quit 
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

        # update any variables for data gathering : TODO add more
        turns += 1

        # allow the player that's turn it is to make its move
        if player == 1:

            # Note: only have one ennabled or bad stuff happens
            # for bot on bot
            if BOT_PLAYING:
                # the bot
                player3, monty_carlo = start_processing(board.board, 1, P_TIME, gui)

                # update the board with the bots chosen move
                chosen_move = convert_bit_move(player3[-2])
                turn = update_board(chosen_move[0], chosen_move[1], board.board)
                if turn == True and check_jump_required(board.board, player, chosen_move[1]):
                    turn = True
                else:
                    turn = False

                gui.limited_options = [chosen_move[0], chosen_move[1]]
            # for human player
            else:
                turn, chosen_move = gui.choose_action()

            if turn:
                continue

        else:
            # the bot
            player2, monty_carlo = start_processing(board.board, 2, P_TIME, gui)

            # update the board with the bots chosen move
            chosen_move = convert_bit_move(player2[-2])
            turn = update_board(chosen_move[0], chosen_move[1], board.board)
            if turn == True and check_jump_required(board.board, player, chosen_move[1]):
                turn = True
            else:
                turn = False

            gui.red_blocks += chosen_move[0], chosen_move[1]
            # print some stats
            print('-'*50 + '\n')
            print(f'Hashes Generated: {player2[-1][2]}')
            print(f'Boards Evaluated: {player2[-1][1]}')
            print(f'ends found with montycarlo search: {monty_carlo[0] + monty_carlo[1]}')
            print(f'predicted board state: {round(player2[-1][3], 4)}')
            print(f'depth reached: {player2[-1][0]}\n')
            print('-'*50 + '\n')

            if turn:
                continue

        # change the current player
        if player == 1:
            player = 2
            gui.red_blocks = []
        else:
            player = 1

        # check for a win
        win = check_win(board.board, player)
        if win == 1:
            # make another pygame loop for showing the win message
            start_time = process_time()
            gui.win_messsage = "Player one wins!"
            while process_time() - start_time < 3:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                gui.draw()
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
                        quit()
                gui.draw()
            print('player two wins')
            board.reset_board(gui)
            gui = Gui(board.board, size, clock, screen, 1)
            player = 1
            p2_wins += 1
            turns = 0

if __name__ == '__main__':
    main()