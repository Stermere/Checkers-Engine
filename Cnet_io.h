// provides the functions to load save and create data for the neural network
// also holds the structs for the neural network and data set 
// provides many of the standalone helper functions for the neural network

#include <stdio.h>
#include <stdlib.h>

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
    struct neuron *neurons;
};

// structure for a neuron of the neural network
struct neuron {
    int prev_layer_neurons_num;
    double *weights;
    double bias;
    double output;
};

// struct to hold the data set
struct data_set{
    int num_games;
};

// function to load the data set from the python list input
void load_data_set_from_py(){

}

// function to calculate the actual values of the data set (only for when converting from python)
void calculate_actual_values(){

}

// function to save the data set to a file
void save_data_set_to_file(){

}

// function to load the data set from a file
void load_data_set_from_file(){

}

// function to load the neural network from a file
void load_neural_network_from_file(){

}

// funciton to save the neural network to a file
void save_neural_network_to_file(){

}


// initialize a fresh network with random weights and biases
struct neural_net* generate_new_network(int num_inputs, int num_outputs, int num_layers, int hidden_layer_size){
    struct neural_net *net = (struct neural_net*)malloc(sizeof(struct neural_net));
    net->num_inputs = num_inputs;
    net->num_outputs = num_outputs;
    net->num_layers = num_layers;
    net->layers = (struct layer*)malloc(sizeof(struct layer) * num_layers);
    // populate the layers
    int temp_layer_size;
    int temp_last_layer_size = 0;
    for (int i = 0; i < num_layers; i++){
        // if this is the first layer then the input size is the number of inputs
        if (i == 0){
            net->layers[i].num_neurons = num_inputs;
            net->layers[i].neurons = (struct neuron*)malloc(sizeof(struct neuron) * num_inputs);
            temp_layer_size = num_inputs;
        }
        // if this is a hidden layer then the input size is the number of neurons in the previous layer
        else {
            net->layers[i].num_neurons = hidden_layer_size;
            net->layers[i].neurons = (struct neuron*)malloc(sizeof(struct neuron) * hidden_layer_size);
            temp_layer_size = hidden_layer_size;
        }
        for (int j = 0; j < temp_layer_size; j++){
            net->layers[i].neurons[j].prev_layer_neurons_num = temp_last_layer_size;
            net->layers[i].neurons[j].weights = (double*)malloc(sizeof(double) * temp_last_layer_size);
            net->layers[i].neurons[j].bias = 0;
            net->layers[i].neurons[j].output = 0;
            for (int k = 0; k < temp_last_layer_size; k++){
                net->layers[i].neurons[j].weights[k] = rand() / (double)RAND_MAX;
           }
        }
        temp_last_layer_size = temp_layer_size;
    }
    return net;
}

// function to print the network to stdio
void print_network_to_stdio(struct neural_net *net){
    printf("Network:\n");
    for (int i = 0; i < net->num_layers; i++){
        printf("\tLayer %d:\n", i);
        for (int j = 0; j < net->layers[i].num_neurons; j++){
            printf("\t\tNeuron %d:\n", j);
            printf("\t\t\tweights:\n");
            for (int k = 0; k < net->layers[i].neurons[j].prev_layer_neurons_num; k++){
                printf("\t\t\t\t%f\n", net->layers[i].neurons[j].weights[k]);
            }
            printf("\n");
        }
    }
}

