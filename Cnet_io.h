// provides the functions to load save and create data for the neural network
// also holds the structs for the neural network and data set 
// provides many of the standalone helper functions for the neural network

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


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
    double error;
};

// struct to hold the data set
struct data_set{
    int endgame_state;
    int move_num;
    struct board_data *game_data;
};

// hold a boad state and eval
struct board_data{
    double true_eval;
    long long int p1;
    long long int p2;
    long long int p1k;
    long long int p2k;
};

// function to load the data set from the python list input
void load_data_set_from_py(){

}

// function to calculate the actual values of the data set (only for when converting from python)
// takes in the data set
void calculate_actual_values(struct data_set *data){

}


// function to save the data set to a file
void save_data_set_to_file(struct data_set *data){

}

// function to load the data set from a file
struct data_set* load_data_set_from_file(){

}

// function to load the neural network from a file
struct neural_net* load_neural_network_from_file(){

}

// function to save the neural network to a file
void save_neural_network_to_file(struct neural_net *net){

}


// initialize a fresh network with random weights and biases
// note: while the network will work with any number of inputs and outputs it 
// is recommended to have the layer size as a multiple of 4
// parameters: num_inputs, num_outputs, num_layers, hidden layer size
struct neural_net* generate_new_network(int num_inputs, int num_outputs, int num_layers, int hidden_layer_size){
    srand(time(NULL));
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
        // if this is the last layer then the output size is the number of outputs
        else if (i == (num_layers - 1)){
            net->layers[i].num_neurons = num_outputs;
            net->layers[i].neurons = (struct neuron*)malloc(sizeof(struct neuron) * num_outputs);
            temp_layer_size = num_outputs;
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
                // generate random weights between -1 and 1 that will not be too close to 0
                net->layers[i].neurons[j].weights[k] = ((rand() / (double)RAND_MAX)) - .5;
                // if the weight is 0 then offset it by 0.1
                if (net->layers[i].neurons[j].weights[k] == 0){
                    net->layers[i].neurons[j].weights[k] += 0.1;
                }
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
            printf("\t\tNeuron %d Bias: %f\n", j, net->layers[i].neurons[j].bias);
            printf("\t\t\tweights:\n");
            for (int k = 0; k < net->layers[i].neurons[j].prev_layer_neurons_num; k++){
                printf("\t\t\t\t%f\n", net->layers[i].neurons[j].weights[k]);
            }
            printf("\n");
        }
    }
}

