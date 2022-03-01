# class that handles the monty carlo tree search algorithm

from distutils.command import check
from Board_opperations import *
from random import randrange
from copy import deepcopy


class Monty_carlo:
    reverse_state = lambda a : 1 if a == 2 else 2

    def __init__(self, state, board):
        self.total_p1_win = 0
        self.total_p2_win = 0
        self.child = None
        self.state = state
        self.board = board

    # the start to finding wins where n is the number of wins that will be found before returning
    def find_n_wins(self, n):
        for i in range(n):
            board = deepcopy(self.board)
            self.find_win()
            self.board = board
  
    # generate a board that could follow this one and continue until a win is found
    def find_win(self):
        # check for a win
        win = check_win(self.board, self.state)
        if win == 1:
            self.total_p1_win = 1
            return
        elif win == 2:
            self.total_p2_win = 1
            return

        # get the list of legal moves for this node
        moves = check_jump_required(self.board, self.state)
        if not moves:
            moves = generate_all_options(self.board, self.state, False)
        else:
            moves = generate_all_options(self.board, self.state, True)
        if len(moves) == 0:
            self.update_params()
            return
        
        # make the child of this board
        rand = randrange(0, len(moves))
        move = moves[rand]
        jumped = update_board(move[0], move[1], self.board)

        # check if the state should be keep the same or updated
        if jumped:
            if not check_jump_required(self.board, self.state, move[1]):
                self.state = Monty_carlo.reverse_state(self.state)
        else:
                self.state = Monty_carlo.reverse_state(self.state)


        # make the child and call its find_win function then update your params
        self.child = Monty_carlo(self.state, self.board)
        self.child.find_win()
        self.update_params()

    # update the parameters of this node
    def update_params(self):
        self.total_p1_win += self.child.total_p1_win
        self.total_p2_win += self.child.total_p2_win