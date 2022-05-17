// provides the python calls to train a network using the Cneural_net.h file
// takes in a neural network file, a training file, epochs, and learning rate
// Models are saved to a file after training is complete

#include "Cneural_net.h"
//#include <Python.h>


// function to train the neural network using using a preloaded network and a training file
// takes in a neural network file, a loaded data set, epochs, and learning rate
// returns 
double train_network(struct neural_net *net, struct data_set *data, int epochs, double learning_rate){
    // load the data set
    // load the neural network
    // run the training
    // save the network
    // return the error
    return 0.0;
}

// function to test the neural network using a preloaded network and a test file
// takes in a neural and a data set
// returns the error rate
double test_network(struct neural_net *net, struct data_set *data){
    // load the data set
    // load the neural network
    // run the test
    // return the error rate
    return 0.0;

}


// function to verify the workings of the program structure
void test_basic_functions(){
    // create a fresh neural network
    struct neural_net *net = generate_new_network(128, 1, 5, 16);
    // print the network
    //print_network_to_stdio(net);

    // push an input through the network
    double startTime = (double)clock()/CLOCKS_PER_SEC;
    populate_input(net, 96872345, 27654, 2698736, 342);
    for (int i = 1; i < 1000000; i++){
        populate_input(net, 96872345, 27654, 2698736, 342);
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

void test_backpropagation(){
    // create a fresh neural network
    struct neural_net *net = generate_new_network(128, 4, 5, 24);
    double startTime = (double)clock()/CLOCKS_PER_SEC;
    populate_input(net, 24535243356 , 467873452341, 78964534231, 23454567789);

    // print the network
    //print_network_to_stdio(net);

    // run the network forward
    forward_propagate(net);

    // print the output
    printf("output:");
    for (int i = 0; i < net->layers[net->num_layers - 1].num_neurons; i++){
        printf("\t%f\n", net->layers[net->num_layers - 1].neurons[i].output);
    }

    // run the network backwards
    double target[] = {0.4, -0.5, 0.6, -0.7};
    for (int i = 0; i < 300; i++){
        forward_propagate(net);
        back_propagate(net, 0.01, &target[0]);
    }
    // get time to run
    startTime = (double)clock()/CLOCKS_PER_SEC - startTime;
    printf("Time to run back prop: %f\n", startTime);

    // run the network forward again
    forward_propagate(net);

    // print the output
    printf("output:");
    for (int i = 0; i < net->layers[net->num_layers - 1].num_neurons; i++){
        printf("\t%f\n", net->layers[net->num_layers - 1].neurons[i].output);
    }

    // print the network
    //print_network_to_stdio(net);

    // save the network
    save_network_to_file(net, "neural_net/test_network");

    // load the network
    struct neural_net *net2 = load_network_from_file("neural_net/test_network");

    // test the network
    populate_input(net2, 24535243356 , 467873452341, 78964534231, 23454567789);
    forward_propagate(net2);

    // print the output
    printf("output loaded from File:");
    for (int i = 0; i < net2->layers[net2->num_layers - 1].num_neurons; i++){
        printf("\t%f\n", net2->layers[net2->num_layers - 1].neurons[i].output);
    }


}



int main(){
    //test_basic_functions();
    test_backpropagation();

    return 1;
}

