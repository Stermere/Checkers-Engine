# class that handels the minimax tree searching

# really needs to be refactored aka rewriten for effeciency
# this code is old and bad like really really bad it should be replced in the next major update

from time import process_time
from random import randrange
from Static_board_eval import get_eval
from Board_opperations import *

class Minmax:
    MAX_DEPTH = 0

    def __init__(self, board : list, depth : int, state : int, alpha : float, beta : float,
     last_move=None, start_time=None, best_moves=None, processing_time=1, only_jump=False, 
     transposition=None, hash_=None, lp=[0, 0]) -> None:
        global leafs
        global branches
        self.start_time = start_time
        self.lp = lp
        self.transposition = transposition
        self.last_move = last_move
        self.best_moves = best_moves
        self.next_moves = []
        self.processing_time = processing_time
        self.board = board
        self.hash = hash_
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
            lp[0] += 1
            return

        if self.depth < Minmax.MAX_DEPTH:
            self.board_tree = self.gen_tree(only_jump=only_jump)
            if self.board_tree:
                self.board_state = self.get_best_node()
        else:
            self.board_tree = self.gen_tree(only_jump=True)
            if not self.board_tree:
                self.board_state = self.eval_board()
                lp[0] += 1
            else:
                self.board_state = self.get_best_node()
                
    # function to generate a tree of all the moves sorted in order from best to worst
    # this works in corilation with alpha beta prunning to allow dynamic depth
    def get_list_best_moves(self):
        reverse_sort = lambda x : True if x == 1 else False
        moves = []
        moves.append(self.board_state)
        moves.append(self.last_move)
        moves.append([])
        moves.append([])
        all_possible_moves = set(self.next_moves)
        if self.board_tree == None:
            return moves
        # add the explored moves in the the list
        for i in self.board_tree:
            moves[2].append((i.last_move, i.board_state))
            move = i.get_list_best_moves()
            moves[3].append(move)
            all_possible_moves.remove(tuple(i.last_move))
        # sort the moves from best to worst
        moves[2].sort(key=lambda x : x[1], reverse=reverse_sort(self.state))

        # add the prunned moves as well just after the sort so they are checked last as if they where prunned that indicated a bad move
        for i in all_possible_moves:
            moves[2].append((i, 0))
        # convert moves back to a usable form
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
                gen_moves = False
        if gen_moves:
            moves = generate_all_options(self.board, self.state, only_jump)

        # save the moves so that in the next depth iteration there is no need to regenerate any moves
        self.next_moves = moves

        min_eval = 1000
        max_eval = -1000
        # create the children of this board state
        for move in moves:
            # update the board and remember some data needed for move reversal
            jumped_piece = None
            jumped_piece_loc = None
            moving_piece_start_type = self.board[move[0][1]][move[0][0]]
            if abs(move[0][0] - move[1][0]) == 2:
                jumped_piece_loc = ((move[0][0] + move[1][0]) // 2, (move[0][1] + move[1][1]) // 2)
                jumped_piece = self.board[jumped_piece_loc[1]][jumped_piece_loc[0]]
            jumped = update_board(move[0], move[1], self.board)

            # make the hash for the following board
            new_hash = self.transposition.hash_from_hash(self.board, jumped_piece, self.hash, move)


            # if the same piece that just moved jumped and can jump again let it!
            check_jump = False
            if jumped:
                check_jump = check_jump_required(self.board, self.state, move[1])

            if self.best_moves:
                next_best_moves = self.get_next_best_moves(move, self.best_moves)
            else:
                next_best_moves = []

            # if the piece that just jumped can jump again pass the same state to the next board if not invert the state
            if check_jump:
                child = Minmax(self.board, self.depth + 1, self.state, self.alpha, self.beta, last_move=[move[0], move[1]],\
                     start_time=self.start_time, best_moves=next_best_moves, processing_time=self.processing_time,\
                        transposition=self.transposition, hash_=new_hash, lp=self.lp)
            else:
                child = Minmax(self.board, self.depth + 1, invert_state(self.state), self.alpha, self.beta, last_move=[move[0], move[1]],\
                     start_time=self.start_time, best_moves=next_best_moves, processing_time=self.processing_time,\
                        transposition=self.transposition, hash_=new_hash, lp=self.lp)
            branches.append(child)

            # print board if child.board_state is none
            if child.board_state == None:
                for item in self.board:
                    print(item)



            # undo the move that was made to the board
            undo_update_board(move, jumped_piece, jumped_piece_loc, moving_piece_start_type, self.board)
            
            # alpha beta prunning
            if self.state == 1:
                max_eval = max(max_eval, child.board_state)
                self.alpha = max(self.alpha, max_eval)
                if self.beta < self.alpha:
                    self.lp[1] += 1
                    break
            if self.state == 2:
                min_eval = min(min_eval, child.board_state)
                self.beta = min(self.beta, min_eval)
                if self.beta < self.alpha:
                    self.lp[1] += 1
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