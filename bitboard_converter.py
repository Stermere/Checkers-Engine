# converts from a 2d matrix to a bit board representation of a board to search moves much faster than a standard 2d matrix

# converts a 2d matrix representation in to a bit board
def convert_to_bitboard(board):
    # initialze the integers
    p1bits = 0
    p2bits = 0
    p1kbits = 0
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
    return (p1bits, p2bits, p1kbits, p2kbits)

# convert a bitboard move to a matrix board move
def convert_to_matrix(p1, p2, p1k, p2k):
    row = [0, 0, 0, 0, 0, 0, 0 ,0]
    board = [row.copy(), row.copy(), row.copy(), row.copy(), row.copy(), row.copy(), row.copy(), row.copy()]
    # loop through the elements of the board
    for num in range(64):
        # if player2 king
        if p2k & (1 << num):
            board[num//8][num%8] = 4
        # if player1 king
        elif p1k & (1 << num):
            board[num//8][num%8] = 3
        # if player 2 piece
        elif p2 & (1 << num):
            board[num//8][num%8] = 2
        # if player 1 piece
        elif p1 & (1 << num):
            board[num//8][num%8] = 1
        # if no piece
        else:
            board[num//8][num%8] = 0
    return board

# convert a matrix board move to a bit board move
def convert_matrix_move(move):
    # move is returned with init_pos final_pos format
    return (move[0][0] + move[0][1]*8, move[1][0] + move[1][1]*8)


# convert a bitboard move to a matrix board move
def convert_bit_move(move):
    # move is returned with x, y format
    return ((move[0] % 8, move[0]//8), (move[1] % 8, move[1]//8))
