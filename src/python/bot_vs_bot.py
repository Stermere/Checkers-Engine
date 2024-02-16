# a python file that pitts two bots against each other

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib.win-amd64-cpython-310')))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib.win-amd64-cpython-3.9')))


import search_engine as se
import search_engine_old as seo

from copy import deepcopy
from bitboard_converter import convert_bit_move, convert_to_bitboard, convert_to_matrix
from Board_opperations import Board, check_jump_required, update_board, check_win, check_tie, generate_all_options

# Generates all ll man ballot starting boards
def get11manBoards():
    boards = []
    p1, p2, _, _ = convert_to_bitboard(Board().board)
    for i in [55, 53, 51, 49, 46, 44, 42, 40]:
        next_p1 = p1 ^ (1 << i)
        for j in [8, 10, 12, 14, 17, 19, 21, 23]:
            next_p2 = p2 ^ (1 << j)

            # add the board to the list
            boards.append((next_p1, next_p2, 0, 0))

    # convert them all to matrix boards
    boards = [convert_to_matrix(board[0], board[1], board[2], board[3]) for board in boards]

    # for each board make one move per side
    boards_with_moves = []
    for i in range(len(boards)):
        board = boards[i]
        moves = generate_all_options(board, 1, False)
        # filter out moves off the backrank
        moves = [move for move in moves if move[0][1] != 0 and move[0][1] != 7]
        for move in moves:
            newBoard = deepcopy(board)
            update_board(move[0], move[1], newBoard)

            moves2 = generate_all_options(newBoard, 2, False)
            moves2 = [move2 for move2 in moves2 if move2[0][1] != 0 and move2[0][1] != 7]
            for move2 in moves2:
                newBoard2 = deepcopy(newBoard)
                update_board(move2[0], move2[1], newBoard2)
                boards_with_moves.append(newBoard2)

    # filter out the boards that are the same
    boards_with_moves = list(set([tuple(map(tuple, board)) for board in boards_with_moves]))

    # convert them back to a list of lists
    boards_with_moves = [[list(row) for row in board] for board in boards_with_moves]

    return boards_with_moves


# start the main loop of the game
def main(args) -> None:
    start_player = 1
    player = 1
    board = Board()
    game_history = []
    p_time = 0.1
    ply = 100

    engine_new_wins = 0
    engine_old_wins = 0
    ties = 0

    games_played = 0

    start_states = get11manBoards()

    # main loop
    for sBoard in start_states:
        games_at_start = games_played
        board.board = deepcopy(sBoard)
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

                board.board = deepcopy(sBoard)
                game_history = []

                # filp the player 
                start_player = abs(start_player - 3)
                player = start_player
            elif win == 2:
                engine_old_wins += 1
                games_played += 1

                board.board = deepcopy(sBoard)
                game_history = []

                # filp the starting player
                start_player = abs(start_player - 3)
                player = start_player

            elif tie:
                engine_new_wins += 0.5
                engine_old_wins += 0.5
                ties += 1
                games_played += 1
                board.board = deepcopy(sBoard)
                game_history = []

                # filp starting player
                start_player = abs(start_player - 3)
                player = start_player

            # print the results so far
            if games_played > 0:
                print(f"new engine: {engine_new_wins} old engine: {engine_old_wins} ties: {ties} wins: {games_played - ties}")
        


if __name__ == '__main__':
    main(sys.argv)