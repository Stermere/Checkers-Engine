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
def convert_bit_move(move):
    # move is returned with (x, y format)
    return ((move[0] % 8, move[0]//8), (move[1] % 8, move[1]//8))
