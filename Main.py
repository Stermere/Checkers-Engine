# a checkers engine using the minimax algorithm
# developed by Collin Kees

from copy import deepcopy
from time import process_time, sleep
from Transposition_table import Transposition
from Minmax import Minmax
from Monty_carlo import Monty_carlo
from Board_opperations import Board, check_jump_required, generate_options, update_board, check_win, generate_all_options

if __name__ == '__main__':
    import pygame
    import multiprocessing as mp


class Player(): # class to deal with the visual elements for the human player
    C1 = (186,45,30)
    C2 = (0,0,0)
    C3 = (126,25,10)
    C4 = (40,40,40)
    CB = (179,149,96)
    CD = (125,107,79)
    CP = (227,116,52)
    CS = (167,229,211)
    CM = (174,181,161)
    CR = (174,141,121)
    CL = (174,181,181)

    def __init__(self, board: list, size : tuple, clock : object, screen : object, type_ : int) -> None:
        self.type = type_
        if self.type == 1:
            self.king_type = 3
        else:
            self.king_type = 4
        self.board = board
        self.selected_block = None
        self.highlighted_blocks = []
        self.limited_options = []
        self.red_blocks = []
        self.size = size
        self.clock = clock
        self.screen = screen
        self.draw()

    # the main loop that runs when the human player is choosing its next move (yes it is a horribly complicated function)
    def choose_action(self) -> list:
        self.limited_options = check_jump_required(self.board, self.type)
        running = True
        while running:
            mouse_button_click = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_button_click.append(event.button)
            # GUI loop starts here
            if mouse_button_click: # handle a click
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for y, row in enumerate(self.board):
                    for x, spot in enumerate(row):
                        # check if the player tryed to move
                        for block in self.highlighted_blocks:
                            if self.size[0] / 8 * x < mouse_x < self.size[0] / 8 * (x + 1)\
                                and self.size[1] / 8 * y < mouse_y < self.size[1] / 8 * (y + 1):
                                if (x, y) == block:
                                    hopped = update_board(tuple(self.selected_block), (x, y), self.board)
                                    self.selected_block = None
                                    self.highlighted_blocks = []
                                    self.limited_options = []
                                    self.red_blocks = []
                                    if hopped:
                                        if not generate_options((x,y), self.board, only_jump=True) == []:
                                            self.draw()
                                            return True, None
                                        else:
                                            self.draw()
                                            running = False
                                            break
                                    else:
                                        self.draw()
                                        running = False
                                        break
                        # check if the player selected a block
                        if self.size[0] / 8 * x < mouse_x < self.size[0] / 8 * (x + 1)\
                             and self.size[1] / 8 * y < mouse_y < self.size[1] / 8 * (y + 1)\
                                 and (spot == self.type or spot == self.king_type):
                            if self.limited_options == []:
                                self.selected_block = [x, y]
                                self.highlighted_blocks = generate_options((x,y), self.board)
                            else:
                                for i in self.limited_options:
                                    if i == (x, y):
                                        self.selected_block = [x, y]
                                        self.highlighted_blocks = generate_options((x,y), self.board, only_jump=True)
            self.draw()
        return False, None

    # draws the board
    def draw(self) -> None:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        block_width = self.size[0]/8
        block_height = self.size[1]/8
        color = lambda x : Player.CB if x % 2 == 0 else Player.CD 
        for y, row in enumerate(self.board):
            for x, spot in enumerate(row):
                i = x + y
                c = color(i)
                rect = (block_width * x,
                        block_height * y,
                        block_width * (x + 1),
                        block_height * (y + 1))
                for piece in self.red_blocks:
                    if (x, y) == piece:
                        c = Player.CR
                for piece in self.highlighted_blocks:
                    if (x, y) == piece:
                        c = Player.CM
                for piece in self.limited_options:
                    if (x, y) == piece:
                        c = Player.CL
                if rect[0] < mouse_x < rect[2] and rect[1] < mouse_y < rect[3]:
                    c = Player.CP
                elif [x, y] == self.selected_block:
                    c = Player.CS
                pygame.draw.rect(self.screen, c, rect)
                
        for y, row in enumerate(self.board):
            for x, spot in enumerate(row):
                spot_on_screen = (x * block_width + block_width * .5,
                    y * block_height + block_height * .5)
                if self.board[y][x] == 1 or self.board[y][x] == 3:
                    pygame.draw.circle(self.screen, Player.C1, spot_on_screen, block_width / 2 - 10)
                    if self.board[y][x] == 3:
                        pygame.draw.circle(self.screen, Player.C3, spot_on_screen, block_width / 2 - 20)
                elif self.board[y][x] == 2 or self.board[y][x] == 4:
                    pygame.draw.circle(self.screen, Player.C2, spot_on_screen, block_width / 2 - 10)
                    if self.board[y][x] == 4:
                        pygame.draw.circle(self.screen, Player.C4, spot_on_screen, block_width / 2 - 20)
        pygame.display.update()

# allows processing of wins until time runs out
def dynamic_win(board : list, state : int, return_dict, p_time : float):
    # get the child that is the new board state
    time = process_time()
    montycarlo = Monty_carlo(state, board)
    
    wins_per_search = 10
    while process_time() - time < p_time:
        montycarlo.find_n_wins(wins_per_search)
        return_dict["wins"] += wins_per_search

    # save the object in a touple interpretation to sent it back to the main thread
    return_data = return_dict["montycarlo"]
    return_data.append((montycarlo.board, montycarlo.total_p1_win, montycarlo.total_p2_win))
    return_dict["montycarlo"] = return_data

# allows for processing to be stopped at a precice time without losing speed 
def dynamic_depth(depth : int, board : list, state : int, p_time, return_dict):
    hashes = 0
    leafs = 0
    prunned = 0
    time = process_time()

    Minmax.MAX_DEPTH = 1
    board = deepcopy(board)
    t = Transposition()
    m = Minmax(board, 0, state, -1000, 1000, start_time=time, processing_time=p_time, transposition=t, hash_=t.initial_hash)
    full_minmax = m
    # if there is only one move in the board tree Imidiatly return since there is no point in looking at future boards in that case
    if len(m.board_tree) == 1:
        return_dict["minmax"] = full_minmax
        return_dict["depth"] = 0
        return_dict["leafs"] = leafs
        return_dict["prunned"] = prunned
        return_dict["hashes"] = hashes
        return

    # continue to process until time constraints are realized
    while process_time() - time < p_time:
        depth += 1
        Minmax.MAX_DEPTH = depth
        full_minmax = m
        leafs, prunned = leafs + full_minmax.lp[0], prunned + full_minmax.lp[1]
        best_moves = m.get_list_best_moves()
        m = Minmax(board, 0, state, -1000, 1000, start_time=time, best_moves=best_moves, processing_time=p_time, transposition=t, hash_=t.initial_hash)

    hashes = t.length()

    # update the return dict
    return_dict["minmax"] = full_minmax
    return_dict["depth"] = depth
    return_dict["leafs"] = leafs
    return_dict["prunned"] = prunned
    return_dict["hashes"] = hashes


# merge the weights from both three searches to get the 'ideal' move
def merge_monty_and_minmax(montycarlo, minmax):
    for child in montycarlo:
        for mmchild in minmax.board_tree:
            # monty carlo list is in the format [board, p1_wins, p2_wins]
            if child[0] == mmchild.board:
                # multiply the ratio of wins to loses by the boards state found by minmax
                if child[1] > child[2]:
                    if mmchild.board_state < 0:
                        new_eval = mmchild.board_state * 0.1 / ((child[1] + 1) / (child[2] + 1))
                    else:
                        new_eval = mmchild.board_state * ((child[1] + 1) / (child[2] + 1))

                else:
                    if mmchild.board_state < 0:
                        new_eval = mmchild.board_state * ((child[2] + 1) / (child[1] + 1))
                    else:
                        new_eval = mmchild.board_state / ((child[2] + 1) / (child[1] + 1))

                print("p1 " , child[1] , " p2 " , child[2] , " origninal state " , mmchild.board_state, " new state ", new_eval)
                
                # update the eval
                mmchild.board_state = new_eval


# start the processing of the minimax and Monty Carlo tree search
def start_processing(depth : int, board : list, state : int, p_time):
    global hashes, leafs, prunned
    manager = mp.Manager()
    return_dict = manager.dict()

    # generate the first depth of the montycarlo tree to split up processing
    invert_state = lambda x : 1 if x == 2 else 2
    if check_jump_required(board, state):
        next_layer_boards = generate_all_options(board, state, True)
        
    else:
        next_layer_boards = generate_all_options(board, state, False)      

    # prepare the processes
    minimax = mp.Process(target=dynamic_depth, args=(depth, board, state, p_time, return_dict, ))
    montycarlo_list = []
    return_dict["montycarlo"] = []
    return_dict["wins"] = 0

    for child in next_layer_boards:
        board_ = deepcopy(board)
        update_board(child[0], child[1], board_)
        state_ = check_jump_required(board_, state, child[1])
        if state_:
            state_ = state
        else:
            state_ = invert_state(state) 
        montycarlo_list.append(mp.Process(target=dynamic_win, args=(board_, state_, return_dict, p_time, )))

    # start the processes
    return_dict["done"] = False
    minimax.start()
    for child in montycarlo_list:
        child.start()

    # once minimax search ends montycarlo should end as well
    minimax.join()
    return_dict["done"] = True
    for child in montycarlo_list:
        child.join()

    merge_monty_and_minmax(return_dict["montycarlo"], return_dict["minmax"])

    # update hashes, leaves, and prunned branches
    hashes = return_dict["hashes"]
    leafs = return_dict["leafs"]
    prunned = return_dict["prunned"]
    return return_dict["minmax"], return_dict["depth"], return_dict["wins"]


# print the trace leading to the initial board eval of a search (used for debugging)
def print_board_tree(player):
    def print_board(board):
        for row in board:
            print(row)
    board_instance = player
    while (board_instance.board_tree != []):
        if board_instance.board_tree == None:
            print('Forced win found')
            return
        for child in board_instance.board_tree:
            if child.board_state == board_instance.board_state:
                print_board(child.board)
                print(f'Board state: {board_instance.state}')
                print(f'number of childen: {len(board_instance.board_tree)}')
                board_instance = child
                break
                
                

def main() -> None:
    global leafs, branches, prunned, hashes
    prunned = 0
    leafs = 0
    branches = 0
    hashes = 0
    size = (800, 800)
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Checkers')
    player = 1
    board = Board()
    player1 = Player(board.board, size, clock, screen, 1) # must be initialized regardless of if a human is playing or not

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

        # update the screen
        player1.draw()

        global board_check
        board_check = []

        # allow the player that's turn it is to make its move
        if player == 1:

            # Note: only have one ennabled or bad stuff happens
            # for bot on bot
            if BOT_PLAYING:
                player3, depth_reached, monty_carlo = start_processing(4, board.board, 1, P_TIME)
                player3.board = board.board
                turn, i = player3.choose_action()
                player1.limited_options = [i[0], i[1]]
                branches, leafs, prunned = 0, 0, 0
            # for human player
            else:
                turn, i = player1.choose_action()

            if turn:
                continue

        else:
            # the bot
            player2, depth_reached, monty_carlo = start_processing(4, board.board, 2, P_TIME)

            # uncomment to see the trace of boards leading to the expected state
            #print_board_tree(player2)

            # update the board to the board to the main board
            player2.board = board.board
            turn, i = player2.choose_action()
            player1.red_blocks += i[0], i[1]
            print(f'time allowed for processing {P_TIME} seconds')
            print(f'Total branches Prunned: {prunned}')
            print(f'Hashes Generated: {hashes}')
            print(f'Boards Evaluated: {leafs}')
            print(f'ends found with montycarlo search: {monty_carlo}')
            if depth_reached == 0:
                print('predicted board state: Unknown')
            else:
                print(f'predicted board state: {round(player2.board_state, 4)}')
            print(f'depth reached: {depth_reached}\n')
            print('-'*50 + '\n')

            leafs = 0
            branches = 0
            prunned = 0

            if turn:
                continue

        # change the current player
        if player == 1:
            player = 2
            player1.red_blocks = []
        else:
            player = 1

        # check for a win
        win = check_win(board.board, player)
        if win == 1:
            player1.draw()
            print('player one wins')
            sleep(3)
            board.reset_board(player1, player2)
            player1 = Player(board.board, size, clock, screen, 1)
            player = 1

        elif win == 2:
            player1.draw()
            print('player two wins')
            sleep(3)
            board.reset_board(player1, player2)
            player1 = Player(board.board, size, clock, screen, 1)
            player = 1

if __name__ == '__main__':
    main()