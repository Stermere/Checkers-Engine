# board opperations for checkers
# uses array indexes to specify the piece that is being moved
# much easier to work with than a bit board but to slow to be used in the search so,
# use bitboard_converters function to convert to and from bitboard

class Board: # class to store the board data and update it accordingly
    def __init__(self) -> None:
        self.board = [[0, 2, 0, 2, 0, 2, 0, 2],
                      [2, 0, 2, 0, 2, 0, 2, 0],
                      [0, 2, 0, 2, 0, 2, 0, 2],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 1, 0, 1, 0, 1, 0],
                      [0, 1, 0, 1, 0, 1, 0, 1],
                      [1, 0, 1, 0, 1, 0, 1, 0]]

        # easy solve
        #self.board = [[0, 4, 0, 0, 0, 0, 0, 0],
        #             [4, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 3],
        #             [0, 0, 0, 0, 0, 0, 0, 0]]
        
        # medium solve
        #self.board = [[0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 4, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 4, 0, 0, 0],
        #             [0, 0, 0, 4, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 0],
        #             [0, 0, 0, 0, 0, 0, 0, 3],
        #             [0, 0, 0, 0, 0, 0, 3, 0]]

        # hard solve
        #self.board = [[0, 3, 0, 0, 0, 0, 0, 3],
        #              [0, 0, 0, 0, 0, 0, 0, 0],
        #              [0, 0, 0, 4, 0, 4, 0, 0],
        #              [0, 0, 0, 0, 0, 0, 0, 0],
        #              [0, 0, 0, 0, 0, 4, 0, 0],
        #              [0, 0, 0, 0, 0, 0, 0, 0],
        #              [0, 0, 0, 0, 0, 0, 0, 1],
        #              [0, 0, 0, 0, 0, 0, 0, 0]]
                    
    def reset_board(self, p1 : object) -> None:
        self.__init__()
        p1.board = self.board


# functions to interact with simulated boards and the current board
def generate_options(piece_pos : tuple, board : list, only_jump=False) -> list:
    posibilitys = []
    piece = board[piece_pos[1]][piece_pos[0]] 
    if piece != 0:
        # get the possible moves a piece would have
        if piece == 1:
            options = [[-1, -1],[1, -1]]
        elif piece == 2:
            options = [[-1, 1],[1, 1]]
        else:
            options = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
        # loop through the options to see what moves are valid
        for i in options:
            new_loc = piece_pos[0] + i[0], piece_pos[1] + i[1]
            if new_loc[0] < 0 or new_loc[1] < 0\
                or new_loc[0] > 7 or new_loc[1] > 7:
                continue
            if not only_jump and board[new_loc[1]][new_loc[0]] == 0:
                posibilitys.append(new_loc)

            elif is_enemy_piece(piece, board[new_loc[1]][new_loc[0]]) == True:
                new_loc = new_loc[0] + i[0], new_loc[1] + i[1]
                if new_loc[0] < 0 or new_loc[1] < 0\
                    or new_loc[0] > 7 or new_loc[1] > 7:
                    continue
                if board[new_loc[1]][new_loc[0]] == 0:
                    posibilitys.append(new_loc)
    return posibilitys


def generate_all_options(board : list, state : int, only_jump : bool) -> list:
    moves = []
    if state == 2:
        king = 4
    else:
        king = 3
    # generate every more the player could make
    for y, row in enumerate(board):
        for x, col in enumerate(row):
            if col == state or col == king:
                move_options = generate_options((x, y), board, only_jump=only_jump)
                for m in move_options:
                    moves.append(((x, y), m))
    return moves


def update_board(piece_pos : tuple, new_pos : tuple, board : list) -> bool:
    board[new_pos[1]][new_pos[0]] = board[piece_pos[1]][piece_pos[0]]
    board[piece_pos[1]][piece_pos[0]] = 0
    piece = board[new_pos[1]][new_pos[0]]
    # check if the piece should become a king
    if piece == 1 and new_pos[1] == 0:
        board[new_pos[1]][new_pos[0]] = 3
    if piece == 2 and new_pos[1] == 7:
        board[new_pos[1]][new_pos[0]] = 4

    jump_dist = piece_pos[0] - new_pos[0]
    if (jump_dist == 2 or jump_dist == -2):
        x, y = int((piece_pos[0] + new_pos[0]) / 2), int((piece_pos[1] + new_pos[1]) / 2)
        board[y][x] = 0
        return not (piece <= 2 and (new_pos[1] == 0 or new_pos[1] == 7))
    return False


def check_jump_required(board : list, player : int, piece=False) -> list:
    if piece != False:
        o = generate_options(piece, board, only_jump=True)
        return o
    if player == 1:
        king = 3
    else:
        king = 4
    required_moves = []
    for y, row in enumerate(board):
        for x, block in enumerate(row):
            if block == player or block == king:
                o = generate_options((x, y), board, only_jump=True)
                if o:
                    required_moves.append((x, y))
    return required_moves


def check_win(board : list, next_player : int) -> int:
    p1 = False
    p2 = False
    for y, row in enumerate(board):
        for x, block in enumerate(row):
            if block == 1 or block == 3:
                o = generate_options((x, y), board)
                if o:
                    p1 = True

            elif block == 2 or block == 4:
                o = generate_options((x, y), board)
                if o:
                    p2 = True

    if next_player == 1 and not p1:
        return 2
    elif next_player == 2 and not p2:
        return 1
    else:
        return 0

# return True if the game is a tie and False if not
def check_tie(boards : list) -> bool:
    # if the same board comes up three time then it is a tie
    for board in boards:
        count = 0
        for b in boards:
            if board == b:
                count += 1
        if count >= 3:
            return True
    return False
                



def is_enemy_piece(piece, other_piece):
    if (piece == 1 or piece == 3) and (other_piece == 2 or other_piece == 4):
        return True
    if (piece == 2 or piece == 4) and (other_piece == 1 or other_piece == 3):
        return True
    else:
        return False


def undo_update_board(move, jumped_piece, jumped_piece_loc, moving_piece, board):
    # undo the main move
    board[move[0][1]][move[0][0]] = moving_piece
    board[move[1][1]][move[1][0]] = 0

    # if the move was a jump revert the piece that was jumped
    if jumped_piece != None:
        board[jumped_piece_loc[1]][jumped_piece_loc[0]] = jumped_piece

