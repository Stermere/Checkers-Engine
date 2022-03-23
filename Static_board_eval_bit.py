# evaluates a bit board in the current state


from secrets import randbits


class Board_evaler:
    def __init__(self):
        self.piece_pos = self.compute_piece_pos()
        self.king_pos = self.compute_king_pos()
        self.num_hashes = 0
        self.hashes = dict()

    # get the eval of a board either but geting the hashed value or computing it and hashing it
    def eval_board(self, piece_loc, p1bits, p1kbits, p2bits, p2kbits):
        board_hash = self.compute_hash(p1bits, p1kbits, p2bits, p2kbits)
        hashed_eval = self.hashes.get(board_hash)
        # if the hash was found return it
        if hashed_eval:
            return hashed_eval
        # if the hash was not found populate the hash map with it
        eval_to_hash = self.numerical_eval(piece_loc, p1bits, p1kbits, p2bits, p2kbits)
        self.hashes[board_hash] = eval_to_hash
        self.num_hashes += 1
        return eval_to_hash

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
            
    # return the value in the precomputed dict
    def eval_king_pos(self, loc):
        return self.king_pos[loc]

    # return the value in the precomputed dict
    def eval_piece_pos(self, loc):
        return self.piece_pos[loc]

    # create the hash of a board (takes 4 ints returns one semi-unique int)
    def compute_hash(self, p1bits, p1kbits, p2bits, p2kbits):
        p1kbits = p1kbits >> 1
        p2kbits = p2kbits >> 1
        return (p1bits ^ p2bits) ^ (p1kbits ^ p2kbits)

    # populates a hashmap that holds the weights of being in a position
    def compute_piece_pos(self):
        # heat map of where it is generaly good to have pieces
        pos_dict = dict()
        heat_map = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 2, 2, 1, 1, 0],
                    [0, 1, 1, 2, 2, 1, 1, 0],
                    [0, 1, 1, 2, 2, 1, 1, 0],
                    [0, 1, 1, 2, 2, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        for i in range(64):
            pos_dict[i] = heat_map[i//8][i%8] / 10

        return pos_dict

    # return a dict of the score to be given to a king piece at a given position
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
            pos_dict[i] = heat_map[i//8][i%8] / 10
        return pos_dict