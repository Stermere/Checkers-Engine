# static evaluation algorithm for any given checkers board
import math
from Board_opperations import *

# function that returns a board value
def get_eval(board):
    pieces = piece_count(board)
    score = numerical_eval(board, pieces)
    #score -= avg_dist_king(board, pieces[0], 1) * 0.02
    #score += avg_dist_king(board, pieces[1], 2) * 0.02

    return score


# returns player1 pieces - player 2 pieces with kings worth more
# also factors in the piece position and king distance
def numerical_eval(board, loc):
    score = 0
    for i in loc[0]:
        if board[i[0]][i[1]] == 3:
            score += 5
            score += king_position(i, loc[1], board) * 0.1
        else:
            score += 3
        score += eval_piece_pos(i) * 0.1
    for i in loc[1]:
        if board[i[0]][i[1]] == 4:
            score -= 5
            score -= king_position(i, loc[0], board) * 0.1
        else:
            score -= 3
        score -= eval_piece_pos(i) * 0.1
    return score


def king_position(loc, enemy_pieces, board):
    lowest_dist = 16
    for i in enemy_pieces:
        if board[i[0]][i[1]] == 3 or board[i[0]][i[1]] == 4:
            dist = abs(loc[0] - i[0]) + abs(loc[1] - i[1])
            if dist < lowest_dist:
                lowest_dist = dist

    return -lowest_dist


# returns the average distance from becoming a king for a player
def avg_dist_king(board, loc, player):
    if player == 2:
        king_row = 7
    else:
        king_row = 0
    total_norm_piece = 0
    total_dist = 0
    for i in loc:
        if board[i[0]][i[1]] == player:
            total_norm_piece += 1
            total_dist += abs(i[0] - king_row)
    if total_norm_piece == 0:
        return 0 
    total_dist = total_dist / total_norm_piece
    
    return total_dist


# if a player is ahead it is benificial for less pieces to be in play
# so give a small reward to the player ahead for there being less pieces
def trade_benifit():
    pass


# get pieces on the board and there locations
def piece_count(board):
    p1 = []
    p2 = []
    for row in range(0,8):
        if row % 2 == 0:
            j = 1
        else:
            j = 0
        for col in range(j, 8, 2):
            if board[row][col] != 0:
                if board[row][col] == 1 or board[row][col] == 3:
                    p1.append((row, col))
                else:
                    p2.append((row, col))
    return (p1, p2)


# get the integer from the heat map for a pieces value
def eval_piece_pos(piece_pos : tuple) -> float:
    heat_map = get_heat_map()
    score = 0
    score += heat_map[piece_pos[1]][piece_pos[0]]
    return score


# TODO returns the heat map that corisponds to how far the game has progressed with more granularity
def get_heat_map():
    heat_map = [[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]]
    return heat_map

