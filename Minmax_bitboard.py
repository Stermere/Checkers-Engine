# uses a bit board representation of a board to search moves much faster than a standard 2d matrix
# this part of the program is mostly self contained as the rest of the program uses the 2d matrix representation 


from time import process_time
from Static_board_eval_bit import Board_evaler

# temp
from Board_opperations import generate_all_options, check_jump_required, undo_update_board, update_board


# converts a 2d matrix representation in to a bit board
def convert_to_bitboard(board):
    # initialze the integers
    p1bits = 0
    p1kbits = 0
    p2bits = 0
    p2kbits = 0
    # loop through the elements of the board
    num = 0
    for row in board:
        for item in row:
            # if player2 king
            if item == 4:
                p2kbits = p2kbits ^ (1 << (num))
            # if player1 king
            elif item == 3:
                p1kbits = p1kbits ^ (1 << (num))
            # if player 2 piece
            elif item == 2:
                p2bits = p2bits ^ (1 << (num))
            # if player 1 piece
            elif item == 1:
                p1bits = p1bits ^ (1 << (num))
            # incriment the count
            num += 1
    # return the new board
    return (p1bits, p1kbits, p2bits, p2kbits)


# convert a bitboard move to a matrix board move
def convert_bit_move(move):
    # move is returned with (x, y format)
    return ((move[0] % 8, move[0]//8), (move[1] % 8, move[1]//8))


# compute the offsets that are legal for each position
def compute_offsets():
    move_offset = dict()
    for i in range(64):
        offsets = [9, 7, -9, -7]
        if (i + 1) % 8 == 0:
            if offsets.__contains__(9):
                offsets.remove(9)
            if offsets.__contains__(-7):
                offsets.remove(-7)
        if i % 8 == 0:
            if offsets.__contains__(-9):
                offsets.remove(-9)
            if offsets.__contains__(7):
                offsets.remove(7)
        if i > 55:
            if offsets.__contains__(7):
                offsets.remove(7)
            if offsets.__contains__(9):
                offsets.remove(9)
        if i < 8:
            if offsets.__contains__(-7):
                offsets.remove(-7)
            if offsets.__contains__(-9):
                offsets.remove(-9)
        move_offset[i] = tuple(offsets)
    return move_offset

# get the indicies of each piece on a board
def get_piece_locations(p1bits, p1kbits, p2bits, p2kbits):
    piece_loc = set()
    for i in range(64):
        if p1bits >> i & 1 or p2bits >> i & 1 or p1kbits >> i & 1 or p2kbits >> i & 1:
            piece_loc.add(i)
    return piece_loc

# update the set of where the pieces are located on the board
def update_piece_locations(move, locations):
    locations.add(move[1])
    locations.remove(move[0])
    if abs(move[0] - move[1]) > 10:
        locations.remove((move[0] + move[1]) // 2)

def undo_update_piece_location(move, locations):
    locations.remove(move[1])
    locations.add(move[0])
    if abs(move[0] - move[1]) > 10:
        locations.add((move[0] + move[1]) // 2)
    
    
# generates all legal moves for a given board
def generate_all_moves(player, piece_loc, p1bits, p1kbits, p2bits, p2kbits, offsets, only_jump=False):
    # set the king type for this generation
    if player == 2:
        king = 4
    else:
        king = 3
    # set a flag to say to look at only jumps or all moves
    jump = only_jump
    options = []

    # loop thought the known piece positions and generate there moves
    for pos in piece_loc:
        piece = get_piece_at_pos(pos, p1bits, p1kbits, p2bits, p2kbits)
        if piece == player or piece == king:
            moves = generate_moves(pos, player, p1bits, p1kbits, p2bits, p2kbits, offsets, only_jump=jump)
            for move in moves:
                if jump == False and abs(move[0]-move[1]) > 10:
                    jump = True
                    moves = generate_moves(pos, player, p1bits, p1kbits, p2bits, p2kbits, offsets, only_jump=jump)
                    options = moves
                    break
                options.append(move)
    return options
            

def generate_moves(pos, player, p1bits, p1kbits, p2bits, p2kbits, offsets, only_jump=False):
    # set the king type for this generation
    if player == 2:
        king = 4
    else:
        king = 3
    
    moves = []
    offset = offsets[pos]
    piece = get_piece_at_pos(pos, p1bits, p1kbits, p2bits, p2kbits)
    for i in offset:
        # see if the offset can be done by the piece type
        if piece == 2 and i < 0:
            continue
        if piece == 1 and i > 0:
            continue

        new_pos = pos + i
        type_ = get_piece_at_pos(new_pos, p1bits, p1kbits, p2bits, p2kbits)
        # if the spot is unocuppied it is a valid move
        if type_ == 0 and only_jump == False:
            moves.append((pos, new_pos))
        # check if the type of the neghboring piece if it is your own do nothing
        elif type_ == player or type_ == king:
            continue
        elif type_ != 0 and offsets[new_pos].__contains__(i):
            # if it is a enemy look to see if the next spot is clear if so add that as a move
            new_pos = new_pos + i
            type_ = get_piece_at_pos(new_pos, p1bits, p1kbits, p2bits, p2kbits)
            if type_ == 0:
                moves.append((pos, new_pos))

    return moves


# return the piece at the board position given
def get_piece_at_pos(pos, p1bits, p1kbits, p2bits, p2kbits):
    if p1bits >> pos & 1:
        return 1
    if p2bits >> pos & 1:
        return 2
    if p1kbits >> pos & 1:
        return 3
    if p2kbits >> pos & 1:
        return 4
    else:
        return 0 

# extract's a list of moves from the sorted move tree and return then in the order they occured
def extract_moves(move_tree):
    moves = []
    for item in move_tree[2]:
        moves.append(item[1])
    return moves


# makes a move on the board and returns a new board
def make_move(move, p1bits, p1kbits, p2bits, p2kbits):
    piece = get_piece_at_pos(move[0], p1bits, p1kbits, p2bits, p2kbits)
    if piece == 1:
        p1bits = p1bits ^ (1 << (move[0]))
        p1bits = p1bits ^ (1 << (move[1]))
    elif piece == 2:
        p2bits = p2bits ^ (1 << (move[0]))
        p2bits = p2bits ^ (1 << (move[1]))
    elif piece == 3:
        p1kbits = p1kbits ^ (1 << (move[0]))
        p1kbits = p1kbits ^ (1 << (move[1]))
    else:
        p2kbits = p2kbits ^ (1 << (move[0]))
        p2kbits = p2kbits ^ (1 << (move[1]))
    # if the piece should become a king move the bit to the king bits
    if (piece == 1 or piece == 2) and (move[1] < 8 or move[1] > 55):
        if piece == 1:
            p1bits = p1bits ^ (1 << (move[1]))
            p1kbits = p1kbits ^ (1 << (move[1]))
        else:
            p2bits = p2bits ^ (1 << (move[1]))
            p2kbits = p2kbits ^ (1 << (move[1]))
    # if the piece jumped update the location it jumped
    if abs(move[0] - move[1]) > 10:
        loc = ((move[0] + move[1]) // 2)
        piece = get_piece_at_pos(loc, p1bits, p1kbits, p2bits, p2kbits)
        if piece == 1:
            p1bits = p1bits ^ (1 << loc)
        elif piece == 2:
            p2bits = p2bits ^ (1 << loc)
        elif piece == 3:
            p1kbits = p1kbits ^ (1 << loc)
        else:
            p2kbits = p2kbits ^ (1 << loc)
    # return the new board bits
    return p1bits, p1kbits, p2bits, p2kbits


class Minmax:
    reverse_state = lambda a : 1 if a == 2 else 2
    reverse_sort = lambda x : True if x == 1 else False
    def __init__(self, board):
        self.initial_board = convert_to_bitboard(board)
        self.offsets = compute_offsets()
        self.evaluator = Board_evaler()
        self.transposition = None 
        self.nodes_traversed = 0
        self.highest_depth = 0
        self.eval = 0
        self.best_move = None
        self.move_list = None 
        self.p_time = 0
        self.s_time = 1000000

    # search to a specified depth and save the transpositions (this should idealy never be used on its own)
    def search(self, player, alpha, beta, depth, p1bits, p1kbits, p2bits, p2kbits, piece_loc, offsets, best_moves, only_jump=False):
        # check if time constraints should be realized
        if process_time() - self.s_time > self.p_time:
            return 0, [None, None, []]

        # set up some variables for optimization
        min_eval = 1000
        max_eval = -1000
        move_search_order = [None, None, []]

        # if depth is 0 start a new search that only looks at captures
        if depth < 0:
            only_jump = True

        # use ordered moves from previous search to increase prunning and decrease number of times moves are generated in dynamic depth style search
        if best_moves[2]:
            moves = extract_moves(best_moves)
        else:
            moves = generate_all_moves(player, piece_loc, p1bits, p1kbits, p2bits, p2kbits, offsets, only_jump=only_jump)

        # if there are no moves check for a win evaluate the board and return
        if not moves:
            # if there are no legal moves the current player must have lost
            # so return the eval of losing
            if only_jump == False:
                if player == 2:
                    return 1000 + depth, move_search_order
                else:
                    return -1000 - depth, move_search_order

            self.nodes_traversed += 1
            return self.evaluator.eval_board(piece_loc, p1bits, p1kbits, p2bits, p2kbits), move_search_order

        # loop through moves until alpha beta kicks then add remaining moves to the end of the sorted return list
        prunned = False
        index = 0
        for move in moves:
            # if prunning happened just add the moves to the list
            if prunned:
                move_search_order[2].append([None, move, []])
                continue
            # get the next best moves for the move
            if best_moves[2]:
                next_best_moves = best_moves[2][index]
            else:
                next_best_moves = [None, None, []]

            # updte the board and search further
            bits = make_move(move, p1bits, p1kbits, p2bits, p2kbits)
            # check which player is going to go next
            if  abs(move[0]-move[1]) > 10\
                and generate_moves(move[1], player, bits[0], bits[1], bits[2], bits[3], offsets, only_jump=True):
                player_ = player
            else:
                player_ = Minmax.reverse_state(player)
            # update the piece set
            update_piece_locations(move, piece_loc)
            # search deaper boards here
            child_eval, search_order = self.search(player_, alpha, beta, depth-1, \
                    bits[0], bits[1], bits[2], bits[3], piece_loc, offsets, next_best_moves)
            # undo the piece set update
            undo_update_piece_location(move, piece_loc)

            # add the searchorder to this layers search order
            search_order[0] = child_eval
            search_order[1] = move
            move_search_order[2].append(search_order)

            # alpha beta prunning
            if player == 1:
                max_eval = max(max_eval, child_eval)
                alpha = max(alpha, max_eval)
                if beta < alpha:
                    prunned = True
                    move_search_order[2] = sorted(move_search_order[2], key=lambda x : x[0], reverse=Minmax.reverse_sort(player))
            if player == 2:
                min_eval = min(min_eval, child_eval)
                beta = min(beta, min_eval)
                if beta < alpha:
                    prunned = True
                    move_search_order[2] = sorted(move_search_order[2], key=lambda x : x[0], reverse=Minmax.reverse_sort(player))
            index += 1

        if prunned == False:
            move_search_order[2] = sorted(move_search_order[2], key=lambda x : x[0], reverse=Minmax.reverse_sort(player))

        # return the highest eval for the player and the move that gets you there
        # and the list of best moves for each board
        if player == 2:
            return_eval = min_eval
        else:
            return_eval = max_eval

        return return_eval, move_search_order

    # update the classes best_move, eval, and highest_depth
    def update_info(self, best_moves, eval_, depth):
        self.best_move = extract_moves(best_moves)[0]
        self.move_list = best_moves
        self.eval = eval_
        self.highest_depth = depth

    # acomplishes the same thing as search but takes time constraints into a count
    def iterative_depth(self, p_time, player, output=dict()):
        # prepare the needed data
        self.s_time = process_time()
        self.p_time = p_time
        piece_loc = get_piece_locations(self.initial_board[0], self.initial_board[1], self.initial_board[2], self.initial_board[3])
        offset_dict = compute_offsets()
        start_time = process_time()
        depth = 2

        # do a initial search to depth 4 as this should always complete in a matter of ms
        eval_, prev_best_moves = self.search(player, -1000, 1000, depth, self.initial_board[0], self.initial_board[1], \
                                            self.initial_board[2], self.initial_board[3], piece_loc, offset_dict, [None, None, []])
        # if there is only one move available dont search furth just return the move
        if len(prev_best_moves[2]) == 1:
            self.update_info(prev_best_moves, eval_, depth)
            output["leafs"] = self.nodes_traversed
            output["hashes"] = 0
            output["depth"] = self.highest_depth
            output["eval"] = self.eval
            output["best_move"] = self.best_move
            return

        # search deaper until the time limit is reached
        while process_time() - start_time < p_time:
            settled_eval = eval_
            settled_best_moves = prev_best_moves
            depth += 1
            self.update_info(settled_best_moves, settled_eval, depth)

            # update the output dict for live gui updates
            output["leafs"] = self.nodes_traversed
            output["hashes"] = 0
            output["depth"] = self.highest_depth
            output["eval"] = self.eval
            output["best_move"] = self.best_move
            
            eval_, prev_best_moves = self.search(player, -1000, 1000, depth, self.initial_board[0], self.initial_board[1], \
                                    self.initial_board[2], self.initial_board[3], piece_loc, offset_dict, settled_best_moves)



        
# generates every board to n ply down
def generate_n_ply(player, p1bits, p1kbits, p2bits, p2kbits, piece_loc, offsets, depth):
    boards = 0
    reverse_state = lambda a : 1 if a == 2 else 2
    if depth == 0:
        return 1
    moves = generate_all_moves(player, piece_loc, p1bits, p1kbits, p2bits, p2kbits, offsets, False)
    for move in moves:

        bits = make_move(move, p1bits, p1kbits, p2bits, p2kbits)

        if  abs(move[0]-move[1]) > 10\
           and generate_moves(move[1], player, bits[0], bits[1], bits[2], bits[3], offsets, only_jump=True):
            player_ = player
        else:
            player_ = reverse_state(player)

        # updte the board and search further

        # update the piece set
        update_piece_locations(move, piece_loc)
            
        boards += generate_n_ply(player_, bits[0], bits[1], bits[2], bits[3], piece_loc, offsets, depth-1)
        # undo the piece set update
        undo_update_piece_location(move, piece_loc)

    return boards


# running this file just does a dynamic search on this board position 
if __name__ == "__main__":
    board = [[0, 2, 0, 2, 0, 2, 0, 2],
             [2, 0, 2, 0, 2, 0, 2, 0],
             [0, 2, 0, 2, 0, 2, 0, 2],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 1, 0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0, 1, 0, 1],
             [1, 0, 1, 0, 1, 0, 1, 0]]
                                                                                                   
    bit = convert_to_bitboard(board)
    dict_ = compute_offsets()
    set_ = get_piece_locations(bit[0], bit[1], bit[2], bit[3])
    m = Minmax(board)

    for i in range(1, 11):
        print(f"Search depth of: {i}")
        start_time = process_time()
        print(f"Number of boards new: {generate_n_ply(1, bit[0], bit[1], bit[2], bit[3], set_, dict_, i)}")
        print(f"took {process_time() - start_time} seconds\n")
        
