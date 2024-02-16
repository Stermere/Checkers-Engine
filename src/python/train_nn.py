import torch
import torch.nn as nn
import torch.onnx
from copy import deepcopy
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
from random import shuffle

import Board_opperations as bo
import bitboard_converter as bc
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'lib.win-amd64-cpython-310')))
import search_engine as se

P_TIME = 0.5
PLY = 100
INPUT_SIZE = ((32 + 28) * 2) + 1
P1K_INPUT_START = 0
P2K_INPUT_START = 32
P1_INPUT_START = 64
P2_INPUT_START = 92
PLAYER_INPUT_START = 120

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_size = INPUT_SIZE
        self.hidden_size = 32
        self.output_size = 1

        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def trainModel(self, X, y, epochs=5, lr=0.005, batchSize=16, criterion=torch.nn.MSELoss()):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        torch.set_grad_enabled(True)

        # Train the model
        for epoch in range(epochs):
            totalLoss = 0
            numLosses = 0
            # Train the model on each batch
            for i in range(0, len(X) - batchSize, batchSize):
                optimizer.zero_grad()
                X_batch = X[i:i + batchSize]
                y_batch = y[i:i + batchSize]
                X_batch = torch.stack(X_batch)
                y_batch = torch.tensor(y_batch, dtype=torch.float32)
                output = self.forward(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                totalLoss += loss.item()
                numLosses += 1


            totalLoss /= numLosses
            numLosses = 0
            print(f'Epoch {epoch} loss: {totalLoss}')

    def test(self, X, y, graph=False):
        outputs = []
        for i in range(len(X)):
            output = self.forward(torch.stack([X[i]]))
            outputs.append(output.item())

        # plot actual vs predicted
        if graph:
            sns.scatterplot(x=range(len(X)), y=[outputs[i] for i in range(len(X))], label='Predicted')
            sns.scatterplot(x=range(len(X)), y=[y[i][0] for i in range(len(X))], label='Actual')
            plt.show()


    def predict(self, X):
        torch.set_grad_enabled(False)
        retVal = []
        X = torch.stack([X])
        for i in range(len(X)):
            retVal.append(self.forward(X[i]).item())

        return retVal

    def save(self, fileName):
        torch.save(self, fileName)

        # dump the parameters to a file
        with open(f'{fileName}.params', 'w') as file:
            for param in self.parameters():
                file.write(str(param.data))
                file.write('\n')

    def saveToOnnx(self, fileName):
        dummy_input = torch.randn(1, INPUT_SIZE)
        torch.onnx.export(self, dummy_input, fileName, verbose=True, input_names=['input'], output_names=['output'])

    def load(self, fileName):
        return torch.load(fileName)
    
# Takes a list of models and a tensor and returns the prediction
def sentToModel(models, X):
    if not isinstance(X, torch.Tensor):
        X = torch.stack(X)

    count = torch.sum(X[0:INPUT_SIZE - 1])
    match count // 6:
        case 3 | 4:
            return models[0].predict(X)
        case 2:
            return models[1].predict(X)
        case 1:
            return models[2].predict(X)
        case 0:
            return models[3].predict(X)
        case _:
            raise ValueError('Invalid number of pieces')

# Segment the data into 4 groups depending on the number of pieces
def segmentData(X, y):
    if not isinstance(X, torch.Tensor):
        X = torch.stack(X)
    # Sort the data by the number of pieces
    X, y = zip(*sorted(zip(X, y), key=lambda x: torch.sum(x[0][0:INPUT_SIZE - 1])))
    # Segment the data into 4 groups depending on the number of pieces
    Xy1, Xy2, Xy3, Xy4 = [], [], [], []
    for i in range(0, len(X)):
        match torch.sum(X[i][0:INPUT_SIZE - 1]).item() // 6:
            case 4 | 3:
                Xy1.append((X[i], y[i]))
            case 2:
                Xy2.append((X[i], y[i]))
            case 1:
                Xy3.append((X[i], y[i]))
            case 0:
                Xy4.append((X[i], y[i]))
            case _:
                print('Error: Invalid number of pieces')

    return Xy1, Xy2, Xy3, Xy4

# Generates all ll man ballot starting boards
def get11manBoards():
    boards = []
    p1, p2, _, _ = bc.convert_to_bitboard(bo.Board().board)
    for i in [55, 53, 51, 49, 46, 44, 42, 40]:
        next_p1 = p1 ^ (1 << i)
        for j in [8, 10, 12, 14, 17, 19, 21, 23]:
            next_p2 = p2 ^ (1 << j)

            # add the board to the list
            boards.append((next_p1, next_p2, 0, 0))

    # convert them all to matrix boards
    boards = [bc.convert_to_matrix(board[0], board[1], board[2], board[3]) for board in boards]

    # for each board make one move per side
    boards_with_moves = []
    for i in range(len(boards)):
        board = boards[i]
        moves = bo.generate_all_options(board, 1, False)
        # filter out moves off the backrank
        moves = [move for move in moves if move[0][1] != 0 and move[0][1] != 7]
        for move in moves:
            newBoard = deepcopy(board)
            bo.update_board(move[0], move[1], newBoard)

            moves2 = bo.generate_all_options(newBoard, 2, False)
            moves2 = [move2 for move2 in moves2 if move2[0][1] != 0 and move2[0][1] != 7]
            for move2 in moves2:
                newBoard2 = deepcopy(newBoard)
                bo.update_board(move2[0], move2[1], newBoard2)
                boards_with_moves.append(newBoard2)

    # filter out the boards that are the same
    boards_with_moves = list(set([tuple(map(tuple, board)) for board in boards_with_moves]))
    boards_with_moves = [list(map(list, board)) for board in boards_with_moves]

    return boards_with_moves

# Transforms the data from game outcome pairs, to position outcome pairs
# Format of return: ([X's], [y's])
def convertToTensor(X, y, startPositions):
    newX = []
    newy = []
    for gameIndex in range(len(X)):
        # Convert the outcome to a 0 for player 2 win, 1 for player 1 win, and 0.5 for a tie
        outcome = 1 if y[gameIndex] == 1 else 0 if y[gameIndex] == 2 else 0.5
        firstMove = X[gameIndex][0]
        board = deepcopy(startPositions[gameIndex])
        player = board[firstMove[0][1]][firstMove[0][0]]
        for move in X[gameIndex]:
            # convert the board to a tensor
            tensor = boardToTensor(board, player)
            newX.append(tensor)
            newy.append([float(outcome)])


            # update the board
            jump = bo.update_board(move[0], move[1], board)
            if jump == True and bo.check_jump_required(board, player, move[1]):
                continue
            player = player ^ 3
    
    return newX, newy

# Converts a board to a tensor for the neural network
# Format is as follows: 32 inputs for p1k, 32 inputs for p2k, 28 inputs for p1, 28 inputs for p2 and 1 input for the player to move
def boardToTensor(board, player):
    tensor = torch.zeros(INPUT_SIZE)
    for row in range(8):
        for col in range(8):
            if board[row][col] == 3:
                tensor[((row * 8) + col) // 2 + P1K_INPUT_START] = 1
            elif board[row][col] == 4:
                tensor[((row * 8) + col) // 2 + P2K_INPUT_START] = 1
            elif board[row][col] == 1:
                tensor[(((8 - row) * 8) + col) // 2 + P1_INPUT_START] = 1
            elif board[row][col] == 2:
                tensor[((row * 8) + col) // 2 + P2_INPUT_START] = 1

    tensor[PLAYER_INPUT_START] = player - 1

    return tensor

# Take a data set and augment it by flipping the boards, moves, and outcomes
# this will double the size of the data set but remove some bias
def augmentData(X, y, startingPositions):
    startingPositionsNew = []
    newX = []
    newy = []
    for i in range(len(X)):
        # Flip the starting position
        newStartingPos = deepcopy(startingPositions[i])
        for row in range(8):
            newStartingPos[row] = newStartingPos[row][::-1]
        newStartingPos = newStartingPos[::-1]
        newStartingPos = [[2 if block == 1 else 1 if block == 2 else 4 if block == 3 else 3 if block == 4 else 0 for block in row] for row in newStartingPos]
        startingPositionsNew.append(newStartingPos)

        # Flip the moves
        newMoves = []
        for move in X[i]:
            newMoves.append(((7 - move[0][0], 7 - move[0][1]), (7 - move[1][0], 7 - move[1][1])))
        newX.append(newMoves)

        # Flip the outcome
        newy.append(1 if y[i] == 2 else 2 if y[i] == 1 else y[i])

    X.extend(newX)
    y.extend(newy)
    startingPositions.extend(startingPositionsNew)

# Load the pdns found in the file and construct the training data
# Format of return: [([X's]), ([y's], [startingPositions])]
def loadGames(fileName):
    X = []
    y = []
    startPositions = []
    with open(fileName, 'r') as file:
        for line in file:
            moves = line.split(',')
            startPosition, moves, winner = moves[0:1], moves[1:-1], moves[-1]
            startPosition = startPosition[0]

            board = [[0 for i in range(8)] for j in range(8)]
            for i in range(len(startPosition)):
                col = i % 8
                row = int(i / 8)
                board[row][col] = int(startPosition[i])
            startPositions.append(deepcopy(board))

            game = []
            for move in moves:
                move = move.strip().split(' ')
                game.append(((int(move[0]), int(move[1])), (int(move[2]), int(move[3]))))
            
            X.append(game)
            y.append(int(winner))

    return X, y, startPositions

# Save the games to a file
def saveGames(fileName, games):
    with open(fileName, 'w') as file:
        for entry in games:
            game, startPosition, winner = entry
            for row in startPosition:
                for block in row:
                    file.write(f'{block}')
            file.write(', ')
            for move in game:
                file.write(f'{move[0][0]} {move[0][1]} {move[1][0]} {move[1][1]}, ')
            file.write(f'{winner}\n')

# Play numGames games and save the results
# return the results
def playGames(games):
    pool = Pool(8)
    results = pool.map(playGame, [(1, deepcopy(game)) for game in games])
    pool.close()

    return results

# Plays the game to completion and returns the game history and the winner
# Format of return: ([game_history], startingPos, winner)
def playGame(args):
    player, board = args

    gameHistory = []
    boardStack = []
    startPosition = deepcopy(board)

    while True:
        # get the next move
        p1, p2, p1k, p2k = bc.convert_to_bitboard(board)
        results = se.search_position(p1, p2, p1k, p2k, player, P_TIME, PLY)
        best_move = bc.convert_bit_move(results[-2])

        # update the board
        jump = bo.update_board(best_move[0], best_move[1], board)
        gameHistory.append(best_move)
        boardStack.append(deepcopy(board))
        if jump == True and bo.check_jump_required(board, player, best_move[1]):
            continue
        
        # switch players
        player = player ^ 3

        # Check if the game is over and evaluate who could have won if a tie did occure
        winner = bo.check_win(board, player)
        result = adjudicateGame(board)
        if winner != 0:
            return gameHistory, startPosition, winner
        elif result != -1:
            return gameHistory, startPosition, result
        elif bo.check_tie(boardStack):
            # Determine the winner by the evaluation from a deeper search)
            p1, p2, p1k, p2k = bc.convert_to_bitboard(board)
            results = se.search_position(p1, p2, p1k, p2k, player, P_TIME * 3, PLY)
            eval_ = results[-1][4]

            if eval_ > 80:
                return gameHistory, startPosition, player
            elif eval_ < -80:
                return gameHistory, startPosition, player ^ 3
            return gameHistory, startPosition, 0

# Returns -1 unless its an endgame where we can guess at the winner
def adjudicateGame(board):
    p1 = 0
    p2 = 0
    for row in board:
        for block in row:
            if block == 1 or block == 3:
                p1 += 1
            elif block == 2 or block == 4:
                p2 += 1
    
    if p1 < 3 or p2 < 3:
        if p1 - 1 > p2:
            return 1
        elif p2 - 1 > p1:
            return 2

    return -1

def main():
    # Play a game
    # games = get11manBoards()
    # results = playGames(games)
    # saveGames('gamesNew.txt', results)
    # quit()

    # Load the games
    X, y, startPositions = loadGames('gamesNew.txt')

    # Convert the games to a tensor
    X, y = convertToTensor(X, y, startPositions)

    # slice off the last 20% of the data for testing
    X_test = X[int(len(X) * 0.8):]
    y_test = y[int(len(y) * 0.8):]

    X_train = X[:int(len(X) * 0.8)]
    y_train = y[:int(len(y) * 0.8)]


    # Train the models
    models = []
    Xy1, Xy2, Xy3, Xy4 = segmentData(X_train, y_train)
    for i, segment in enumerate([Xy1, Xy2, Xy3, Xy4]):
        print(f'Training model {i}')
        shuffle(segment)
        model = Net()
        X, y = zip(*segment)
        model.trainModel(X, y)
        models.append(model)
    
    # Test the model
    Xy1, Xy2, Xy3, Xy4 = segmentData(X_test, y_test)
    for i, segment  in enumerate([Xy1, Xy2, Xy3, Xy4]):
        model = models[i]
        sortedData = sorted(segment, key=lambda x: x[1][0])
        X, y = zip(*sortedData)
        model.test(X, y, graph=True)

    # Save the models
    for i in range(len(models)):
        models[i].save(f'model{i}')


    # Test the model by evaluating a won 4 vs 3 game
    testBoard = [[0, 4, 0, 4, 0, 4, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [3, 0, 3, 0, 3, 0, 3, 0]]
    tensor = boardToTensor(testBoard, 1)
    result = sentToModel(models, tensor)
    print(f"Eval if player 1 to play: {int((result[0] - 0.5) * 1000)}")
    tensor = boardToTensor(testBoard, 2)
    result = sentToModel(models, tensor)
    print(f"Eval if player 2 to play: {int((result[0] - 0.5) * 1000)}")


    # Test the model by evaluating a lost 4 vs 3 game
    testBoard = [[0, 0, 0, 4, 0, 0, 0, 0],
                 [0, 0, 4, 0, 0, 0, 0, 0],
                 [0, 0, 0, 4, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 4, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 3, 0, 0, 0, 0],
                 [0, 0, 0, 0, 3, 0, 3, 0]]
    tensor = boardToTensor(testBoard, 1)
    result = sentToModel(models, tensor)
    print(f"Eval if player 1 to play: {int((result[0] - 0.5) * 1000)}")
    tensor = boardToTensor(testBoard, 2)
    result = sentToModel(models, tensor)
    print(f"Eval if player 2 to play: {int((result[0] - 0.5) * 1000)}")

    # Save the models to onnx
    for i in range(len(models)):
        models[i].saveToOnnx(f'model{i}.onnx')

    # test the model by loading it and using it
    import onnx
    onnx_model = onnx.load("model2.onnx")
    onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    main()

