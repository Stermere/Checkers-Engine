import torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.output_size = 1
        self.feature_extractor = CheckersFeatureExtractor()
        self.input_size = self.feature_extractor.featureCount

        self.fc = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        x = self.fc(x)
        return x.sigmoid()

    def train_model(self, X, y, epochs=5, lr=0.1, batch_size=1024):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # convert the data to tensors 
        X = [self.feature_extractor.extract_features(x) for x in X]
        y = [[i] for i in y]

        print("Beginning training...")

        y = torch.tensor(y, dtype=torch.float32)

        for epoch in range(epochs):
            total_loss = 0
            num_losses = 0

            for i in range(0, len(X) - batch_size, batch_size):
                optimizer.zero_grad()
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                X_batch = torch.stack(X_batch)
                y_batch = torch.tensor(y_batch, dtype=torch.float32)
                output = self.forward(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_losses += 1

            total_loss /= num_losses
            print(f'Epoch {epoch} loss: {total_loss}')

    def test(self, X, y, graph=False):
        outputs = []
        y = [[i] for i in y]
        X = [self.feature_extractor.extract_features(x) for x in X]
        for i in range(len(X)):
            output = self.forward(X[i])
            outputs.append(output.item())

        sorted_outputs_pred = zip(outputs, y)
        sorted_outputs_pred = sorted(sorted_outputs_pred, key=lambda x: x[1])
        outputs, y = zip(*sorted_outputs_pred)

        if graph:
            sns.scatterplot(x=range(len(X)), y=[outputs[i] for i in range(len(X))], label='Predicted')
            sns.scatterplot(x=range(len(X)), y=[y[i][0] for i in range(len(X))], label='Actual')
            plt.show()

    def predict(self, X):
        with torch.no_grad():
            X = torch.stack([X])
            return [self.forward(X[i]).item() for i in range(len(X))]

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))


class CheckersFeatureExtractor():
    def __init__(self):
        self.featurePairs = self.genFeaturePairs()
        self.featureCount= sum([len(x) for x in self.featurePairs.values()]) * 5
        print(self.featureCount)

    def genFeaturePairs(self):
        features = dict()
        for i in range(64):
            # if not a square that we can place a piece on then skip
            if i % 2 == 0:
                pass
            squares_in_feature = set()
            squares_in_feature.add(i)
            for offset in [7, 9, -7, -9]:
                for secondOffset in [7, 9, -7, -9]:
                    if i + offset + secondOffset <= 0 or i + offset >= 63:
                        continue
                    squares_in_feature.add(i + offset + secondOffset)
            features[i] = squares_in_feature
        return features

    def extract_features(self, board):
        p1, p2, p1k, p2k = board

        # output tensor
        features = torch.zeros(self.featureCount)

        index = 0
        for i in range(64):
            if i % 2 == 0:
                continue

            # total features for this square is 5 * len(self.featurePairs[i]) 
            for j, item in enumerate(self.featurePairs[i]):
                piece = self.getPiece(p1, p2, p1k, p2k, item)
                features[index + j * 5 + piece] = 1
            index += len(self.featurePairs[i]) * 5

        return features

    def pieceListToIndex(self, pieces):
        return 
    
    def getPiece(self, p1, p2, p1k, p2k, loc):
        if p1 & (1 << loc):
            return 1
        elif p2 & (1 << loc):
            return 2
        elif p1k & (1 << loc):
            return 3
        elif p2k & (1 << loc):
            return 4
        else:
            return 0