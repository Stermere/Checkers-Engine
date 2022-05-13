// provides the python calls to train a network using the Cneural_net.h file
// takes in a neural network file, a training file, epochs, and learning rate
// Models are saved to a file after training is complete

#include "Cneural_net.h"
//#include <Python.h>


// function to train the neural network using using a preloaded network and a training file
// takes in a neural network file, a training file, epochs, and learning rate
// returns 
double train_network(){

}

// function to test the neural network using a preloaded network and a test file
// takes in a neural network file and a test file
// returns the error rate
double test_network(){

}


// function to verify the workings of the program structure
void test_basic_functions(){
    // create a fresh neural network
    struct neural_net *net = generate_new_network(128, 1, 5, 24);
    // print the network
    //print_network_to_stdio(net);

    // push an input through the network
    double startTime = (double)clock()/CLOCKS_PER_SEC;
    populate_input(net, 96872345, 27654, 2698736, 342);
    for (int i = 1; i < 1000000; i++){
        forward_propagate(net);
    }
    forward_propagate(net);

    // see how long it took
    startTime = (double)clock()/CLOCKS_PER_SEC - startTime;
    printf("Time to run forward prop: %f\n", startTime);

    // print the output
    
    printf("output:");
    for (int i = 0; i < net->layers[net->num_layers - 1].num_neurons; i++){
        printf("\t%f\n", net->layers[net->num_layers - 1].neurons[i].output);
    }
    

}



int main(){
    test_basic_functions();

    return 1;
}

