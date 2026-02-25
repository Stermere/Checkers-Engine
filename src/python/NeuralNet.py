import torch
import seaborn as sns
import matplotlib.pyplot as plt


INPUT_SIZE = ((32 + 28) * 2) + 1

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