# evaluates a bit board in the current state


class Board_evaler:
    def __init__(self):
        self.piece_pos = self.compute_piece_pos()
        self.king_pos = self.compute_king_pos()

    def eval_board(self, piece_loc, p1bits, p1kbits, p2bits, p2kbits):
        eval_ = self.numerical_eval(piece_loc, p1bits, p1kbits, p2bits, p2kbits)
        return eval_

    # returns the material based eval along with the king positions and piece positions factored in
    def numerical_eval(self, piece_loc, p1bits, p1kbits, p2bits, p2kbits):
        eval_ = 0
        for loc in piece_loc:
            # get the piece that is at the location
            if p1bits >> loc & 1:
                eval_ += 3
                eval_ += self.eval_piece_pos(loc)
            elif p2bits >> loc & 1:
                eval_ -= 3
                eval_ -= self.eval_piece_pos(loc)
            elif p1kbits >> loc & 1:
                eval_ += 5
                eval_ += self.eval_king_pos(loc)
                eval_ += self.eval_piece_pos(loc)
            elif p2kbits >> loc & 1:
                eval_ -= 5
                eval_ -= self.eval_king_pos(loc)
                eval_ -= self.eval_piece_pos(loc)

        return eval_ 
            

    def eval_king_pos(self, loc):
        return self.king_pos[loc] / 10

    def eval_piece_pos(self, loc):
        return self.piece_pos[loc] / 10

    # populates a hashmap that holds the weights of being in a position
    def compute_piece_pos(self):
        # heat map of where it is generaly good to have pieces
        pos_dict = dict()
        heat_map = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        for i in range(64):
            pos_dict[i] = heat_map[i//8][i%8]

        return pos_dict

    def compute_king_pos(self):
        # heat map of where it is generaly good to have your king
        pos_dict = dict()
        heat_map = [[1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 2, 2, 2, 2, 2, 1, 0],
                    [0, 1, 3, 3, 3, 3, 1, 0],
                    [0, 1, 3, 4, 4, 3, 1, 0],
                    [0, 1, 3, 4, 4, 3, 1, 0],
                    [0, 1, 3, 3, 3, 3, 1, 0],
                    [0, 1, 2, 2, 2, 2, 2, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1]]
        for i in range(64):
            pos_dict[i] = heat_map[i//8][i%8]   
        return pos_dict