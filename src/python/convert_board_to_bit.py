import Board_opperations as bo
import bitboard_converter as bc

# This file serves as a refrence and helper for the board evaluation bit check evals.
# Positions below are the positions being given a bonus

# first check
"""[[0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]]
"""

# second check
"""[[0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 2, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]]
"""

board = [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]
         

# convert to a bitboard
p1, p2, p1k, p2k = bc.convert_to_bitboard(board)

print("board in bit format:")
print(hex(p1))
print(hex(p2))
print(hex(p1k))
print(hex(p2k))
print()

# reverse the board
for item in board:
    item.reverse()
board.reverse()

# reverse the players
for item in board:
    for i in range(len(item)):
        if item[i] == 1:
            item[i] = 2
        elif item[i] == 2:
            item[i] = 1
        elif item[i] == 3:
            item[i] = 4
        elif item[i] == 4:
            item[i] = 3

# convert to a bitboard
p1, p2, p1k, p2k = bc.convert_to_bitboard(board)
print("check for other player:")
print(hex(p1))
print(hex(p2))
print(hex(p1k))
print(hex(p2k))


