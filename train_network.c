// provides the python calls to train a network using the Cneural_net.h file
// takes in a neural network file, a training file, epochs, and learning rate
// Models are saved to a file after training is complete

#include "Cneural_net.h"
//#include <Python.h>


// function to train the neural network using using a preloaded network and a training file
// takes in a neural network file, a training file, epochs, and learning rate
void train_network(){

}

// function to test the neural network using a preloaded network and a test file
// takes in a neural network file and a test file
void test_network(){

}


// function to verify the workings of the program structure
void test_basic_functions(){
    // create a fresh neural network
    struct neural_net *net = generate_new_network(32, 1, 3, 2);
    // print the network
    print_network_to_stdio(net);

}



int main(){
    test_basic_functions();

    return 1;
}

