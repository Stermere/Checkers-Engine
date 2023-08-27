// provides the functions to load save and create data for the neural network
// also holds the structs for the neural network and data set 
// provides many of the standalone helper functions for the neural network

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


// declerations
struct neural_net* generate_new_network(int num_inputs, int num_outputs, int num_layers, int hidden_layer_size);


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
    struct data_entry *game_data;
};

// hold a boad state and eval
struct data_entry{
    double true_eval;
    long long int p1;
    long long int p2;
    long long int p1k;
    long long int p2k;
};

// randomly shuffle the data set
void shuffle_data_set(struct data_set *data){
    srand(time(NULL));
    int i, j;
    struct data_entry temp;
    for (i = data->move_num - 1; i > 0; i--){
        j = rand() % (i + 1);
        temp = data->game_data[i];
        data->game_data[i] = data->game_data[j];
        data->game_data[j] = temp;
    }
}

// function to save the data set to a file
void save_data_set_to_file(struct data_set *data, char *filename){
    // prepare the file to write to
    char *new_filename = malloc(strlen(filename) + 5);
    strcpy(new_filename, filename);
    strcat(new_filename, ".ds");

    // open the file
    FILE *file = fopen(new_filename, "w");

    // write the data set to the file
    fprintf(file, "%d ", data->endgame_state);
    fprintf(file, "%d\n", data->move_num);
    for (int i = 0; i < data->move_num; i++){
        fprintf(file, "%lld ", data->game_data[i].p1);
        fprintf(file, "%lld ", data->game_data[i].p2);
        fprintf(file, "%lld ", data->game_data[i].p1k);
        fprintf(file, "%lld ", data->game_data[i].p2k);
        fprintf(file, "%lf\n", data->game_data[i].true_eval);
    }

    // close the file
    fclose(file);
}   

// function to load the data set from a file
struct data_set* load_data_set_from_file(char *filename){
    // prepare the file to read from
    char *new_filename = malloc(strlen(filename) + 5);
    strcpy(new_filename, filename);
    strcat(new_filename, ".ds");

    // open the file
    FILE *file = fopen(new_filename, "r");

    // read the data set from the file
    struct data_set *data = malloc(sizeof(struct data_set));
    fscanf(file, "%d", &data->endgame_state);
    fscanf(file, "%d", &data->move_num);
    data->game_data = malloc(sizeof(struct data_entry) * data->move_num);
    for (int i = 0; i < data->move_num; i++){
        fscanf(file, "%lld", &data->game_data[i].p1);
        fscanf(file, "%lld", &data->game_data[i].p2);
        fscanf(file, "%lld", &data->game_data[i].p1k);
        fscanf(file, "%lld", &data->game_data[i].p2k);
        fscanf(file, "%lf", &data->game_data[i].true_eval);
    }

    // close the file
    fclose(file);

    // return the data set
    return data;
}

// function to load the neural network from a file
struct neural_net* load_network_from_file(char *filename){

    // append the .nn to the filename
    char *new_filename = malloc(strlen(filename) + 5);
    strcpy(new_filename, filename);
    strcat(new_filename, ".nn");

    // open the file
    FILE *file = fopen(new_filename, "r");
    if (file == NULL){
        printf("Error opening file\n");
        exit(2);
    }
    // read the number of inputs
    int num_inputs;
    fscanf(file, "%d", &num_inputs);
    // read the number of outputs
    int num_outputs;
    fscanf(file, "%d", &num_outputs);
    // read the number of layers
    int num_layers;
    fscanf(file, "%d", &num_layers);
    // read the number of neurons in each hidder layer
    int num_neurons_hidden;
    fscanf(file, "%d", &num_neurons_hidden);

    // create the neural network
    // this isnt the most efficient way to do this but it works and its not too bad
    struct neural_net *net = generate_new_network(num_inputs, num_outputs, num_layers, num_neurons_hidden);


    // since the first layer is just the inputs, we dont need to read it (it also doesnt exist in the file)
    for (int i = 1; i < net->num_layers - 1; i++){
        for (int j = 0; j < net->layers[i].num_neurons; j++){
            for (int k = 0; k < net->layers[i-1].num_neurons; k++){
                fscanf(file, "%lf", &net->layers[i].neurons[j].weights[k]);
            }
            fscanf(file, "%lf", &net->layers[i].neurons[j].bias);
        }
    }
    // finally read in the output layer
    for (int i = 0; i < net->layers[net->num_layers - 1].num_neurons; i++){
        for (int j = 0; j < net->layers[net->num_layers - 2].num_neurons; j++){
            fscanf(file, "%lf", &net->layers[net->num_layers - 1].neurons[i].weights[j]);
        }
        fscanf(file, "%lf", &net->layers[net->num_layers - 1].neurons[i].bias);
    }
    

    // close the file
    fclose(file);

    // return the network
    return net;
}

// function to save the neural network to a file
void save_network_to_file(struct neural_net *net, char *filename){
    // save the number of inputs, outputs, layer count, and the hidden layer sizes to the file
    // then save the weights and biases for each layer to the file

    // add a extension to the file name
    char *new_filename = malloc(strlen(filename) + 5);
    strcpy(new_filename, filename);
    strcat(new_filename, ".nn");


    FILE *file = fopen(new_filename, "w");
    if (file == NULL){
        perror("File NOT Opened");
        printf("Error opening file!\n");
        exit(2);
    }

    fprintf(file, "%d %d %d %d\n", net->num_inputs, net->num_outputs, net->num_layers, net->layers[1].num_neurons);
    // now in order of layers from input to output save the weights and biases for each layer (except the first its weights are usless)
    for (int i = 1; i < net->num_layers; i++){
        for (int j = 0; j < net->layers[i].num_neurons; j++){
            for (int k = 0; k < net->layers[i].neurons[j].prev_layer_neurons_num; k++){
                fprintf(file, "%f ", net->layers[i].neurons[j].weights[k]);
            }
            fprintf(file, "%f\n", net->layers[i].neurons[j].bias);
        }
    }
    fclose(file);

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
                net->layers[i].neurons[j].weights[k] = ((rand() / (double)RAND_MAX));
                // if the weight is 0 then offset it by 0.1
                if (net->layers[i].neurons[j].weights[k] == 0){
                    net->layers[i].neurons[j].weights[k] += 0.001;
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

void print_data_to_stdio(struct data_set *data){
    printf("Data:\n");
    for (int i = 0; i < data->move_num; i++){
        printf("\tInput %d: %f\n", i, data->game_data[i].true_eval);
    }
}
