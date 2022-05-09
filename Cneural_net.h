// neural network for evaluating checkers moves
// file contains the structs and functions for the neural network

#include "Cdata_loader_net.h"

// structure for the neural network
struct neural_net {
    int num_inputs;
    int num_outputs;
    int num_layers;
    struct layer *layers;
};

// structure for a layer of the neural network
struct layer {
    int num_neurons;
    int num_inputs;
    struct neuron *neurons;
};

// structure for a neuron of the neural network
struct neuron {
    double *weights;
    double bias;
    double output;
};

// get the output of the neural network for a given input
double get_output(struct neural_net *net, long long int p1, long long int p2, long long int p1k, long long int p2k){
    return 0;
}

// train the neural network
void train(struct neural_net *net, struct *data_set){

}