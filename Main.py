# a checkers engine using the minimax algorithm
# developed by Collin Kees

from time import process_time, sleep
from copy import deepcopy
from random import randrange
import pygame
from Static_board_eval import get_eval
from Transposition_table import Transposition
from Board_opperations import *


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


class Minmax:
    MAX_DEPTH = 5

    def __init__(self, board : list, depth : int, state : int, alpha : float, beta : float,
     last_move=None, start_time=None, best_moves=None, processing_time=1, only_jump=False, 
     transposition=None, last_board=None, last_hash=None) -> None:
        global leafs
        global branches
        self.start_time = start_time
        self.transposition = transposition
        self.last_move = last_move
        self.best_moves = best_moves
        self.processing_time = processing_time
        self.board = board
        self.hash = transposition.hash_from_hash(self.board, last_board, last_hash, last_move)
        self.alpha = alpha
        self.beta = beta
        self.depth = depth
        self.state = state
        self.board_state = None
        self.board_tree = None
        if process_time() - start_time > self.processing_time:
            self.board_state = 0
            return
        # recursive part of the init function
        if check_win(self.board, self.state) != 0:
            win_state = check_win(self.board, self.state)
            # ensure that the shortest path to a win is taken rather than a long way
            if win_state == 2:
                self.board_state = -100 + self.depth
            else:
                self.board_state = 100 - self.depth
            leafs += 1
            return

        if self.depth < Minmax.MAX_DEPTH:
            branches += 1
            self.board_tree = self.gen_tree(only_jump=only_jump)
            if self.board_tree:
                self.board_state = self.get_best_node()
                branches += 1
        else:
            self.board_tree = self.gen_tree(only_jump=True)
            if not self.board_tree:
                self.board_state = self.eval_board()
                leafs += 1
            else:
                self.board_state = self.get_best_node()
                branches += 1
                
    # function to generate a tree of all the moves sorted in order from best to worst
    # this works in corilation with alpha beta prunning to allow dynamic depth
    def get_list_best_moves(self):
        reverse_sort = lambda x : True if x == 1 else False
        moves = []
        moves.append(self.board_state)
        moves.append(self.last_move)
        moves.append([])
        moves.append([])
        if self.board_tree == None:
            return moves
        for i in self.board_tree:
            moves[2].append((i.last_move, i.board_state))
            move = i.get_list_best_moves()
            moves[3].append(move)
        # sort the moves from best to worst
        moves[2].sort(key=lambda x : x[1], reverse=reverse_sort(self.state))
        for x, m in enumerate(moves[2]):
            moves[2][x] = tuple(m[0])

        return moves
        
    # choose one of the actions that yields the highest board value
    def choose_action(self) -> tuple:
        actions = []
        if self.state == 2:
            best_node = 1000
            for board in self.board_tree:
                if board.board_state < best_node:
                    best_node = board.board_state
        else:
            best_node = -1000
            for board in self.board_tree:
                if board.board_state > best_node:
                    best_node = board.board_state
        for board in self.board_tree:
            if board.board_state == best_node:
                actions.append(board.last_move)
        if len(actions) == 0:
            return False, None
        move = actions[randrange(0, len(actions))]
        jumped = update_board(move[0], move[1], self.board)
        # allow the bot another turn if another jump by the same piece is possible
        if jumped and check_jump_required(self.board, self.state, move[1]):
            return True, move
        return False, move
             
    # function that handles generation the branches of the current board state
    def gen_tree(self, only_jump=False) -> list:
        global prunned
        invert_state = lambda x : 1 if x == 2 else 2
        branches = []

        # determine if the player must jump
        if not only_jump:
            only_jump = check_jump_required(self.board, self.state)
            if only_jump:
                only_jump = True
            else:
                only_jump = False

        # generate moves or get from sorted move list of last search
        gen_moves = True
        if self.best_moves:
            if self.best_moves[3]:
                moves = self.best_moves[2]
                moves_set = set(moves)
                moves_all = generate_all_options(self.board, self.state, only_jump)
                for item in moves_all:
                    if not item in moves_set:
                        moves.append(item)
                gen_moves = False
        if gen_moves:
            moves = generate_all_options(self.board, self.state, only_jump)

        min_eval = 1000
        max_eval = -1000
        # create the children of this board state
        for move in moves:
            board = deepcopy(self.board)
            jumped = update_board(move[0], move[1], board)

            # if the same piece that just moved jumped and can jump again let it!
            check_jump = False
            if jumped:
                check_jump = check_jump_required(board, self.state, move[1])

            if self.best_moves:
                next_best_moves = self.get_next_best_moves(move, self.best_moves)
            else:
                next_best_moves = []

            # if the piece that just jumped can jump again pass the same state to the next board if not invert the state
            if check_jump:
                child = Minmax(board, self.depth + 1, self.state, self.alpha, self.beta, last_move=[move[0], move[1]],\
                     start_time=self.start_time, best_moves=next_best_moves, processing_time=self.processing_time,\
                        transposition=self.transposition, last_hash=self.hash, last_board=self.board)
            else:
                child = Minmax(board, self.depth + 1, invert_state(self.state), self.alpha, self.beta, last_move=[move[0], move[1]],\
                     start_time=self.start_time, best_moves=next_best_moves, processing_time=self.processing_time,\
                        transposition=self.transposition, last_hash=self.hash, last_board=self.board)
                     
            branches.append(child)

            # alpha beta prunning
            if self.state == 1:
                max_eval = max(max_eval, child.board_state)
                self.alpha = max(self.alpha, max_eval)
                if self.beta < self.alpha:
                    prunned += 1
                    break
            if self.state == 2:
                min_eval = min(min_eval, child.board_state)
                self.beta = min(self.beta, min_eval)
                if self.beta < self.alpha:
                    prunned += 1
                    break
        return branches

    def eval_board(self) -> float: # gives the board state a numerical value
        score = self.transposition.get_value(self.hash)
        if score == None:
            score = get_eval(self.board)
            self.transposition.store_hash(self.hash, score)
        return score

    def get_best_node(self) -> float:
        if self.state == 2:
            best_node = 1000
            for board in self.board_tree:
                if board.board_state < best_node:
                    best_node = board.board_state
        else:
            best_node = -1000
            for board in self.board_tree:
                if board.board_state > best_node:
                    best_node = board.board_state
        return best_node

    def get_next_best_moves(self, move, next_moves):
        for item in next_moves[3]:
            if item != None:
                if move == tuple(item[1]):
                    if len(item[3]) == 0:
                        return None
                    return item
        return None


# allows for processing to be stopped at a precice time without losing speed 
def dynamic_depth(start_time : float, depth : int, board : list, state : int, p_time) -> tuple:
    global branches, leafs, prunned, hashes
    default_depth = Minmax.MAX_DEPTH
    b, r, p = 0, 0, 0
    hashes = 0

    Minmax.MAX_DEPTH = depth
    t = Transposition()
    m = Minmax(board, 0, state, -1000, 1000, start_time=start_time, processing_time=p_time, transposition=t)
    full_minmax = m
    # if there is only one move in the board tree Imidiatly return since there is no point in looking at future boards in that case
    if len(m.board_tree) == 1:
        return full_minmax, depth

    # continue to process until time constraints are realized
    while process_time() - start_time < p_time:
        depth += 1
        Minmax.MAX_DEPTH = depth
        full_minmax = m
        b, r, p = branches + b, leafs + r, prunned + p
        best_moves = m.get_list_best_moves()
        m = Minmax(board, 0, state, -1000, 1000, start_time=start_time, best_moves=best_moves, processing_time=p_time, transposition=t)
    branches, leafs, prunned = b, r, p
    hashes = t.length()
    Minmax.MAX_DEPTH = default_depth

    return full_minmax, depth

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
                time = process_time()
                player3, depth_reached = dynamic_depth(time, 4, board.board, 1, P_TIME)
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
            time = process_time()
            player2, depth_reached = dynamic_depth(time, 4, board.board, 2, P_TIME)

            # uncomment to see the trace of boards leading to the expected state
            #print_board_tree(player2)

            turn, i = player2.choose_action()
            player1.red_blocks += i[0], i[1]
            print(f'processing time: {process_time() - time} seconds')
            print(f'Total branches Prunned: {prunned}')
            print(f'Hashes Generated: {hashes}')
            print(f'Boards Evaluated: {leafs}')
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
