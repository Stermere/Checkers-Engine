// Author: Collin Kees

// neural network for evaluating checkers moves
// file contains the functions for training and using the neural network

#include "net_io.c"
#include <math.h>


// declerations
void activation(struct neuron *n, struct layer *last_layer, double squish_val);
void populate_input(struct neural_net *net, long long int p1, long long int p2, long long int p1k, long long int p2k);
void forward_propagate(struct neural_net *net);
double transfer_tanh(double activation);
double transfer_relu(double activation);
double error_relu_hidden(double weight, double error, double activation);
double error_tanh_out(double activation, double target);
void update_weights(struct neural_net *net, double learning_rate, int batch_size);
double transfer_tanh_deriv(double activation);
double error_relu_out(double activation, double target);

/////////////////////////////////////////////////////////////////
// checkers oriented functions

// run the neural network for the given board_position
double get_output(struct neural_net *net, long long int p1, long long int p2, long long int p1k, long long int p2k){
    // put the input in to the network
    populate_input(net, p1, p2, p1k, p2k);
    // run the network
    forward_propagate(net);
    // get the output
    return net->layers[net->num_layers - 1].neurons[0].output;
}

    
// put the input in to the first layers output
void populate_input(struct neural_net *net, long long int p1, long long int p2, long long int p1k, long long int p2k){
    struct layer *input_layer = net->layers;
    // map of the shift amounts to get a certain piece
    int shift_map[] = {
        1, 3, 5, 7,
        8, 10, 12, 14,
        17, 19, 21, 23,
        24, 26, 28, 30,
        33, 35, 37, 39,
        40, 42, 44, 46,
        49, 51, 53, 55,
        56, 58, 60, 62,
    };
    int shift_amount;
    int num_loops = 32;
    double input_value;
    for (int i = 0; i < num_loops; i++){
        shift_amount = shift_map[i];
        if (p1 >> shift_amount & 1 == 1){
            input_layer->neurons[i].output = 1.0;
            input_layer->neurons[i + 32].output = 0.0;
            input_layer->neurons[i + 64].output = 0.0;
            input_layer->neurons[i + 96].output = 0.0;
            input_value = 7.5;
        }
        else if (p2 >> shift_amount & 1 == 1){
            input_layer->neurons[i].output = 0.0;
            input_layer->neurons[i + 32].output = 1.0;
            input_layer->neurons[i + 64].output = 0.0;
            input_layer->neurons[i + 96].output = 0.0;
            input_value = 2.5;
        }
        else if (p1k >> shift_amount & 1 == 1){
            input_layer->neurons[i].output = 0.0;
            input_layer->neurons[i + 32].output = 0.0;
            input_layer->neurons[i + 64].output = 1.0;
            input_layer->neurons[i + 96].output = 0.0;
        }
        else if (p2k >> shift_amount & 1 == 1){
            input_layer->neurons[i].output = 0.0;
            input_layer->neurons[i + 32].output = 0.0;
            input_layer->neurons[i + 64].output = 0.0;
            input_layer->neurons[i + 96].output = 1.0;
        }
        else{
            input_layer->neurons[i].output = 0.0;
            input_layer->neurons[i + 32].output = 0.0;
            input_layer->neurons[i + 64].output = 0.0;
            input_layer->neurons[i + 96].output = 0.0;
        }
    }
}


// back propogate the error
void back_propagate(struct neural_net *net, double learning_rate, double *target){
    // first calculate the error for the output layer
    struct layer *output_layer = net->layers + net->num_layers - 1;
    struct layer *current_layer = output_layer - 1;
    for (int i = 0; i < output_layer->num_neurons; i++){
        output_layer->neurons[i].error = error_relu_out(output_layer->neurons[i].output, target[i]);
    }
    // then calculate the error for the hidden layers
    for (int i = net->num_layers - 2; i >= 0; i--){
        for (int j = 0; j < current_layer->num_neurons; j++){
            double error = 0.0;
            for (int k = 0; k < output_layer->num_neurons; k++){
                // TODO this can be optimized
                error += error_relu_hidden(output_layer->neurons[k].weights[j], output_layer->neurons[k].error, current_layer->neurons[j].output);
            }
            error = error / current_layer->num_neurons;
            current_layer->neurons[j].error += error;
        }
        current_layer--;
        output_layer--;
    }
}

// update the weights after running backpropagation
void update_weights(struct neural_net *net, double learning_rate, int batch_size){
    struct layer *current_layer = net->layers + net->num_layers - 1;
    struct layer *prior_layer = current_layer - 1;
    float lr;
    for (int i = 1; i < net->num_layers; i++){
        if (i == 1){
            lr = learning_rate * 0.1;
        }
        else{
            lr = learning_rate;
        }
        for (int j = 0; j < current_layer->num_neurons; j++){
            for (int k = 0; k < current_layer->neurons[j].prev_layer_neurons_num; k++){
                current_layer->neurons[j].weights[k] -= lr * (current_layer->neurons[j].error / batch_size) * prior_layer->neurons[k].output;
            }
            // update the bias (if it is the last layer dont update the bias)
            current_layer->neurons[j].bias -= learning_rate * (current_layer->neurons[j].error / batch_size) * 0.1;
            current_layer->neurons[j].error = 0.0;
        }

        current_layer--;
        prior_layer--;
    }
}

// run the neural network for the given input (assumes input layer is already set)
void forward_propagate(struct neural_net *net){
    struct layer* last_layer = net->layers;
    struct layer* current_layer = net->layers + 1;
    int current_layer_index = 1;
    int num_layers = net->num_layers - 1;

     // loop through the layers
    for (int i = 1; i < num_layers; i++){
        // loop throught the neurons in the layer
        // also compute 1/total_weights
        int num_neurons = current_layer->num_neurons;
        double squish_val = 1.0 / (double)num_neurons;
        for (int j = 0; j < num_neurons; j++){
            activation(current_layer->neurons + j, last_layer, squish_val);
            current_layer->neurons[j].output = transfer_relu(current_layer->neurons[j].output);
            
        }
        current_layer_index++;
        last_layer = current_layer;
        current_layer = net->layers + current_layer_index;
    }
    // now activate the last layer
    int num_neurons = net->layers[net->num_layers - 1].num_neurons;
    double squish_val = 1.0 / (double)num_neurons;
    for (int i = 0; i < num_neurons; i++){
        activation(current_layer->neurons + i, last_layer, squish_val);
        current_layer->neurons[i].output = transfer_relu(current_layer->neurons[i].output);
    }
}

// transforms the value using the tanh function
double transfer_tanh(double activation){
    return (exp(activation)-exp(-activation))/(exp(activation)+exp(-activation));
}

// returns the derivative of the tanh function
double transfer_tanh_deriv(double activation){
    double temp = transfer_tanh(activation);
    return 1.0 - (temp * temp); 
}

// return the error of the output
double error_tanh_out(double activation, double target){
    return (activation - target) * transfer_tanh_deriv(activation);
}

// calculates the error of the hidden layer
double error_tanh_hidden(double weight, double error, double activation){
    return (weight * error) * transfer_tanh_deriv(activation);
}

// transform the value using the ReLU function
double transfer_relu(double activation){
    if (activation >= 0.0){
        return activation;
    }
    return activation * 0.01;
}

// returns the derivative of the ReLU function
double transfer_relu_deriv(double activation){
    if (activation >= 0.0){
        return 1.0;
    }
    return 0.01;
}

// return the error of the output
double error_relu_out(double activation, double target){
    return (activation - target) * transfer_relu_deriv(activation);
}

// calculates the error of the hidden layer
double error_relu_hidden(double weight, double error, double activation){
    return (weight * error) * transfer_relu_deriv(activation);
}

// activate a neuron
void activation(struct neuron *n, struct layer *last_layer, double squish_val){
    double sum1 = 0;
    double sum2 = 0;
    double sum3 = 0;
    double sum4 = 0;
    int num_weights = n->prev_layer_neurons_num;
    int final_loop = num_weights % 4;
    //sum the weights and the inputs
    for (int i = 0; i < num_weights; i += 4){
        sum1 += n->weights[i] * last_layer->neurons[i].output;
        sum2 += n->weights[i + 1] * last_layer->neurons[i + 1].output;
        sum3 += n->weights[i + 2] * last_layer->neurons[i + 2].output;
        sum4 += n->weights[i + 3] * last_layer->neurons[i + 3].output;
    }
    // sum the rest of the weights
    for (int i = num_weights - final_loop; i < num_weights; i++){
        sum1 += n->weights[i] * last_layer->neurons[i].output;
    }
    //add the bias and sum the sums
    sum1 = sum1 + n->bias;
    sum1 = sum1 + sum2;
    sum3 = sum3 + sum4;
    sum1 = sum1 + sum3;
 
    //set the output
    n->output = sum1 * squish_val;
}