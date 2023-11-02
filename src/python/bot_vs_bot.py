# a python file that pitts two bots against each other

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib.win-amd64-cpython-310')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib.win-amd64-cpython-3.9')))


import search_engine as se
import search_engine_old as seo

from copy import deepcopy
from bitboard_converter import convert_bit_move, convert_to_bitboard, convert_to_matrix
from Board_opperations import Board, check_jump_required, update_board, check_win, check_tie
from random import randint

# generates all ll man ballot starting boards (does not make the moves only removes pieces)
def get11man_boards():
    boards = []
    p1, p2, _, _ = convert_to_bitboard(Board().board)
    for i in [55, 53, 51, 49, 46, 44, 42, 40]:
        next_p1 = p1 ^ (1 << i)
        for j in [8, 10, 12, 14, 17, 19, 21, 23]:
            next_p2 = p2 ^ (1 << j)

            # add the board to the list
            boards.append((next_p1, next_p2, 0, 0))

    return boards


# start the main loop of the game
def main(args) -> None:
    start_player = 1
    player = 1
    board = Board()
    game_history = []
    p_time = 1
    ply = 50

    engine_new_wins = 0
    engine_old_wins = 0

    games_played = 0

    start_states = get11man_boards()

    # main loop
    for sBoard in start_states:

        games_at_start = games_played
        board.board = convert_to_matrix(sBoard[0], sBoard[1], sBoard[2], sBoard[3])
        while (games_played - games_at_start) < 2:
            game_history.append(deepcopy(board.board))

            # player 1's turn
            if player == 1:
                # the bot
                print("new engine")
                p1, p2, p1k, p2k = convert_to_bitboard(board.board)
                results = se.search_position(p1, p2, p1k, p2k, player, p_time, ply)

                depth = results[-1][0]
                depth_extended = results[-1][1]
                leafs = results[-1][2]
                eval_ = results[-1][4]
                hashes = results[-1][3]
                best_move = convert_bit_move(results[-2])
                

                # update the board with the bots chosen move
                turn = update_board(best_move[0], best_move[1], board.board)
                if turn == True and check_jump_required(board.board, player, best_move[1]):
                    turn = True
                else:
                    turn = False
                if turn:
                    continue

            # player 2's turn
            else:
                # the bot
                print("old engine")
                p1, p2, p1k, p2k = convert_to_bitboard(board.board)
                results = seo.search_position(p1, p2, p1k, p2k, player, p_time, ply)

                depth = results[-1][0]
                depth_extended = results[-1][1]
                leafs = results[-1][2]
                eval_ = results[-1][4]
                hashes = results[-1][3]
                best_move = convert_bit_move(results[-2])

                # update the board with the bots chosen move
                turn = update_board(best_move[0], best_move[1], board.board)
                if turn == True and check_jump_required(board.board, player, best_move[1]):
                    turn = True
                else:
                    turn = False
                if turn:
                    continue

            player = abs(player - 3)

            # check for a win
            win = check_win(board.board, player)
            tie = check_tie(game_history)
            if win == 1:
                engine_new_wins += 1
                games_played += 1

                board.board = convert_to_matrix(sBoard[0], sBoard[1], sBoard[2], sBoard[3])
                game_history = []

                # filp the player 
                start_player = abs(start_player - 3)
                player = start_player
            elif win == 2:
                engine_old_wins += 1
                games_played += 1

                board.board = convert_to_matrix(sBoard[0], sBoard[1], sBoard[2], sBoard[3])
                game_history = []

                # filp the starting player
                start_player = abs(start_player - 3)
                player = start_player

            elif tie:
                engine_new_wins += 0.5
                engine_old_wins += 0.5
                games_played += 1
                board.board = convert_to_matrix(sBoard[0], sBoard[1], sBoard[2], sBoard[3])
                game_history = []

                # filp starting player
                start_player = abs(start_player - 3)
                player = start_player

            # print the results so far
            if games_played > 0:
                print(f"new engine: {engine_new_wins} old engine: {engine_old_wins}")
        


if __name__ == '__main__':
    main(sys.argv)