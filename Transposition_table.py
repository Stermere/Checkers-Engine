# impliments a transposition table that can be easily interacted with

from secrets import randbits


class Transposition():
    def __init__(self):
        self.table = dict()
        # using a 64 bit hash should be fast and prevent almost all hash collisions
        self.initial_hash = randbits(64)
    
    def get_value(self, board_hash):
        value = self.table.get(board_hash)
        return value

    def store_hash(self, board_hash, value):
        self.table[board_hash] = value

    def hash_from_hash(self, board, last_board, last_hash, move):
        # if there is no previus move return the initial hash since after the first move is generated
        # there should always be a previus move and hash combo
        if move == None or last_hash == None or last_board == None:
            return self.initial_hash
        # update the hash
        piece = board[move[1][1]][move[1][0]]
        move_dist = abs(move[0][0] - move[1][0])
        hash_value = last_hash

        # update the hash
        hash_value = self.change_bin_value(piece, self.board_val(move[1]), hash_value)
        hash_value = self.change_bin_value(piece, self.board_val(move[0]), hash_value)

        # if the last move was a jump update the position inbetween
        if move_dist == 2:
            y = int((move[0][1] + move[1][1]) / 2)
            x = int((move[0][0] + move[1][0]) / 2)
            jumped_piece = last_board[y][x]
            hash_value = self.change_bin_value(jumped_piece, self.board_val((x, y)), hash_value)
        return hash_value

    def change_bin_value(self, type, val, hash_value):
        if type == 1:
            place = 0
        elif type == 3:
            place = 8
        elif type == 2:
            place = 32
        else:
            place = 24
        
        hash_value = hash_value ^ (1 << (val + place))
        return hash_value


    def board_val(self, loc):
        val = (loc[1] * 4) - 1
        if loc[1] % 2 == 1:
            x = int((loc[0] + 1) / 2)
        else:
            x = (loc[0] / 2)
        val -= x 
        return int(val)

    def length(self):
        return len(self.table)