// provides the python calls to train a network using the Cneural_net.h file
// takes in a neural network file, a training file, epochs, and learning rate
// Models are saved to a file after training is complete

#include "Cneural_net.h"
#define PY_SSIZE_T_CLEAN
#include <Python.h>


// python wrapper for train_network and test_network
////////////////////////////////////////////

// train the neural network
static PyObject* train_net(PyObject *self, PyObject *args){

    char *net_file[50];
    char *train_file[50];
    int epochs;
    double learning_rate;

    // get the arguments from python
    if (!Pyarg_Parsetuple(args, "ss", net_file, train_file, &epochs, &learning_rate))
        return NULL;

    
}

// test the neural network
static PyObject* test_net(PyObject *self, PyObject *args){

    char net_file[50];
    char train_file[50];

    // get the arguments from python
    if (!Pyarg_Parsetuple(args, "ss", &net_file[0], &train_file[0]))
        return NULL;

    printf("%s\n", &net_file[0]);
    printf("%s\n", &train_file[0]);

    
}

// tell the pyhton interpreter about the functions we want to use
static PyMethodDef c_neural_net[] = {
    {"train_net", train_net, METH_VARARGS, 
    "train the neural network"},
    {"test_net", test_net, METH_VARARGS,
    "test the neural network"},
    {NULL, NULL, 0, NULL}
};

// define the module
static struct PyModuleDef checkers_NN = {
    PyModuleDef_HEAD_INIT,
    "neural_network module",
    "train and test neural networks",
    -1,
    c_neural_net

};

PyMODINIT_FUNC
PyInit_search_engine(void){
    return PyModule_Create(&checkers_NN);
}

////////////////////////////////////////////


// function to train the neural network using using a preloaded network and a training file
// takes in a neural network, a loaded data set, epochs, learning rate and a file to save the model to
// Models are saved to a file after training is complete
// assumes one output neuron
double train_network(struct neural_net *net, struct data_set *data, int epochs, double learning_rate, char *filename){
    int i;
    double error;
    for(i = 0; i < epochs; i++){
        error = 0;
        for (int j = 0; j < data->move_num; j++){
            populate_input(net, data->game_data[j].p1, data->game_data[j].p2, data->game_data[j].p1k, data->game_data[j].p2k);
            forward_propagate(net);
            back_propagate(net, learning_rate, &data->game_data[j].true_eval);
            error += net->layers[net->num_layers - 1].neurons[0].error;

        }
        error = error / data->move_num;
        printf("Epoch %d: %f\n", i, error);
    }
    save_network_to_file(net, filename);
    return error;
}

// function to test the neural network using a preloaded network and a test file
// takes in a neural and a data set
// returns the error rate
double test_network(struct neural_net *net, struct data_set *data){
    double error;
    error = 0;
    for (int j = 0; j < data->move_num; j++){
        populate_input(net, data->game_data[j].p1, data->game_data[j].p2, data->game_data[j].p1k, data->game_data[j].p2k);
        forward_propagate(net);
        // calculate the error
        error += error_tanh_out(net->layers[net->num_layers - 1].neurons[0].output, data->game_data[j].true_eval);
    }

    return error / data->move_num;

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
    struct neural_net *net = generate_new_network(128, 4, 100, 16);
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
    return 1;
    //test_basic_functions();
    test_backpropagation();

    return 1;
}

