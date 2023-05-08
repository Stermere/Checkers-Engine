# import the network
import sys
import os
sys.path.insert(0, os.getcwd() + "/build/lib.win-amd64-3.9/")
import checkers_NN

def main():
    # create a new neural network
    num_inputs = 128
    num_outputs = 1
    num_layers = 3
    hidden_size = 32
    file_name = 'neural_net/neural_net'

    checkers_NN.init_net(num_inputs, num_layers, num_outputs, hidden_size, file_name)



if __name__ == '__main__':
    main()