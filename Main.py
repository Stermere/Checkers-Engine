# The driver program that interacts with the board search algorithms and GUI 
# uses minimax and monty carlo tree searches to find a optimial move
# developed by Collin Kees

from copy import deepcopy
from time import process_time, sleep
from Transposition_table import Transposition
from Minmax_bitboard import Minmax
from Monty_carlo import Monty_carlo
from Board_opperations import Board, check_jump_required, update_board, check_win, generate_all_options

if __name__ == '__main__':
    import pygame
    import multiprocessing as mp
    from Gui import Gui
    from Minmax_bitboard import convert_bit_move


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
def dynamic_depth(board : list, state : int, p_time, return_dict): 
    minmax = Minmax(board)
    minmax.iterative_depth(p_time, state, output=return_dict)

    # update the return dict (stops montycarlo if a search is terminated before the time constraint)
    return_dict["minmax"] = minmax 
    return_dict["done"] = True


# merge the weights from both three searches to resolve any ties in the minimax search
def merge_monty_and_minmax(montycarlo, minmax, player):
    move_options = [] + [[minmax.move_list[2][0][0], minmax.move_list[2][0][1]]]
    highest_eval = minmax.move_list[2][0][0]
    for item in minmax.move_list[2]:
        if item[0] == None:
            continue
        if player == 1:
            if item[0] > highest_eval:
                move_options = [[item[0], item[1]]]
                highest_eval = item[0]
            elif item[0] == highest_eval:
                move_options.append([item[0], item[1]])
        if player == 2:
            if item[0] < highest_eval:
                move_options = [[item[0], item[1]]]
                highest_eval = item[0]
            elif item[0] == highest_eval:
                move_options.append([item[0], item[1]])

    best_move = move_options[0][1]
    best_ratio = 0
    for child in montycarlo:
        for move in move_options:
            # monty carlo list is in the format [move, p1_wins, p2_wins]
            if child[0] == convert_bit_move(move[1]):
                if player == 2:
                    ratio = (child[2] + 1) / (child[1] + 1)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_move = move[1]
                else:
                    ratio = (child[1] + 1) / (child[2] + 1)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_move = move[1]

    # update the best move attribute
    minmax.best_move = best_move
    
    print(minmax.best_move, minmax.eval)


# start the processing of the minimax and Monty Carlo tree search
def start_processing(board : list, state : int, p_time, gui: object):
    manager = mp.Manager()
    return_dict = manager.dict()

    # initialize some values that will be needed
    return_dict["minmax"] = None
    return_dict["best_move"] = None
    return_dict["eval"] = 0
    return_dict["depth"] = 0
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
    return_dict["montycarlo"] = []

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
        gui.update_params(return_dict)

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

    # variables to keep track of some data that will be needed to train the neral net (also nice to display some stats while playing if you want)
    p1_wins = 0
    p2_wins = 0
    turns = 0
    board_at_turn = []
    pieces_at_turn = []


    # True to have the game play against itself
    BOT_PLAYING = False
    P_TIME = 3

    while True:
        # check for quit 
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

        # update any variables for data gathering : TODO add more
        turns += 1
        
        # update the screen
        gui.draw()

        # allow the player that's turn it is to make its move
        if player == 1:

            # Note: only have one ennabled or bad stuff happens
            # for bot on bot
            if BOT_PLAYING:
                player3, monty_carlo = start_processing(board.board, 1, P_TIME, gui)
                            # update the board with the bots chosen move
                chosen_move = convert_bit_move(player3.best_move)
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
            chosen_move = convert_bit_move(player2.best_move)
            turn = update_board(chosen_move[0], chosen_move[1], board.board)
            if turn == True and check_jump_required(board.board, player, chosen_move[1]):
                turn = True
            else:
                turn = False

            gui.red_blocks += chosen_move[0], chosen_move[1]
            print(f'time allowed for processing {P_TIME} seconds')
            print(f'Total branches Prunned: {0}')
            print(f'Hashes Generated: {0}')
            print(f'Boards Evaluated: {player2.nodes_traversed}')
            print(f'ends found with montycarlo search: {monty_carlo[0] + monty_carlo[1]}')
            if player2.highest_depth == 0:
                print('predicted board state: Unknown')
            else:
                print(f'predicted board state: {round(player2.eval, 4)}')
            print(f'depth reached: {player2.highest_depth}\n')
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
            gui.draw()
            print('player one wins')
            sleep(3)
            board.reset_board(gui, player2)
            gui = Gui(board.board, size, clock, screen, 1)
            player = 1
            p1_wins += 1
            turns = 0

        elif win == 2:
            gui.draw()
            print('player two wins')
            sleep(3)
            board.reset_board(gui, player2)
            gui = Gui(board.board, size, clock, screen, 1)
            player = 1
            p2_wins += 1
            turns = 0

if __name__ == '__main__':
    main()