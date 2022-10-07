// provides the python calls to train a network using the Cneural_net.h file
// takes in a neural network file, a training file, epochs, and learning rate
// Models are saved to a file after training is complete

#include "Cneural_net.h"
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// declerations
double train_network(struct neural_net *net, struct data_set *data, int epochs, double learning_rate, char *filename);
double test_network(struct neural_net *net, struct data_set *data);

// python wrapper for train_network and test_network

// train the neural network
static PyObject* train_net(PyObject *self, PyObject *args){

    char net_file_arr[50];
    char train_file_arr[50];

    char* net_file = &net_file_arr[0];
    char* train_file = &train_file_arr[0];

    int epochs;
    int batch_size;
    double learning_rate;

    Py_buffer* net_buf = (Py_buffer*)malloc(sizeof(Py_buffer));
    Py_buffer* train_buf = (Py_buffer*)malloc(sizeof(Py_buffer));

    // get the arguments from python
    if (!PyArg_ParseTuple(args, "s*s*iid", net_buf, train_buf, &batch_size, &epochs, &learning_rate)){
        return NULL;
    }

    // copy the buffer to a string
    // print net_buf->buf
    strcpy(net_file, net_buf->buf);
    strcpy(train_file, train_buf->buf);

    // train the network
    struct neural_net* net = load_network_from_file(net_file);
    struct data_set* data = load_data_set_from_file(train_file);

    double error;
    error = train_network(net, data, epochs, learning_rate, batch_size, net_file);

    save_network_to_file(net, net_file);


    free(net);
    free(data);
    free(net_buf);
    free(train_buf);

    // return the error
    return Py_BuildValue("d", error);
}

// test the neural network
static PyObject* test_net(PyObject *self, PyObject *args){

    char net_file_arr[50];
    char train_file_arr[50];

    char* net_file = &net_file_arr[0];
    char* train_file = &train_file_arr[0];

    Py_buffer* net_buf = (Py_buffer*)malloc(sizeof(Py_buffer));
    Py_buffer* train_buf = (Py_buffer*)malloc(sizeof(Py_buffer));

    // get the arguments from python
    if (!PyArg_ParseTuple(args, "s*s*", net_buf, train_buf)){
        return NULL;
    }

    // copy the buffer to a string
    strcpy(net_file, net_buf->buf);
    strcpy(train_file, train_buf->buf);

    // test the network
    struct neural_net* net = load_network_from_file(net_file);
    struct data_set* data = load_data_set_from_file(train_file);
    double error;

    error = test_network(net, data);

    free(net);
    free(data);

    // return the error
    return Py_BuildValue("d", error);
    
}

// function to initialize a fresh neural network
static PyObject* init_net(PyObject *self, PyObject *args){

    int num_inputs;
    int num_layers;
    int num_outputs;
    int hidden_size;

    char file_arr[50];

    char* save_file = &file_arr[0];

    Py_buffer* file_buf = (Py_buffer*)malloc(sizeof(Py_buffer));

    // get the arguments from python
    if (!PyArg_ParseTuple(args, "iiiis*", &num_inputs, &num_layers, &num_outputs, &hidden_size, file_buf)){
        return NULL;
    }

    // copy the buffer to a string
    strcpy(save_file, file_buf->buf);

    // initialize the network
    struct neural_net* net = generate_new_network(num_inputs, num_outputs, num_layers, hidden_size);

    save_network_to_file(net, save_file);

    free(net);
    free(file_buf);

    // return the network
    return Py_BuildValue("i", 1);
}


// tell the pyhton interpreter about the functions we want to use
static PyMethodDef c_neural_net[] = {
    {"train_net", train_net, METH_VARARGS, 
    "train the neural network"},
    {"test_net", test_net, METH_VARARGS,
    "test the neural network"},
    {"init_net", init_net, METH_VARARGS,
    "initialize a fresh neural network"},
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
PyInit_checkers_NN(void){
    return PyModule_Create(&checkers_NN);
}

// function to train the neural network using using a preloaded network and a training file
// takes in a neural network, a loaded data set, epochs, learning rate and a file to save the model to
// Models are saved to a file after training is complete
// assumes one output neuron
double train_network(struct neural_net *net, struct data_set *data, int epochs, double learning_rate, int batch_size, char *filename){
    double error;
    double error_final = 0;

    for (int i = 0; i < epochs; i++){
        error = 0;
        
        // shuffle the data set
        shuffle_data_set(data);
        
        // only train on a random subset of the data that is batch size
        int rand_start = rand() % (data->move_num - batch_size);
        int rand_end = rand_start + batch_size;

        for (int j = rand_start; j < rand_end; j++){
            populate_input(net, data->game_data[j].p1, data->game_data[j].p2, data->game_data[j].p1k, data->game_data[j].p2k);
            forward_propagate(net);
            back_propagate(net, learning_rate, &data->game_data[j].true_eval);
            error += abs(error_relu_out(net->layers[net->num_layers - 1].neurons[0].output, data->game_data[j].true_eval));
            

        }
        // update the weights
        update_weights(net, learning_rate, batch_size);
        error_final += error / batch_size;
        error = 0;
    }
    save_network_to_file(net, filename);
    return error_final / epochs;
}

// function to test the neural network using a preloaded network and a test file
// takes in a neural and a data set
// returns the error rate
double test_network(struct neural_net *net, struct data_set *data){
    double error;

    // shuffle the data set
    shuffle_data_set(data);
    error = 0;
    
    for (int j = 0; j < data->move_num; j++){
        populate_input(net, data->game_data[j].p1, data->game_data[j].p2, data->game_data[j].p1k, data->game_data[j].p2k);
        forward_propagate(net);
        // calculate the error
        error += abs(error_relu_out(net->layers[net->num_layers - 1].neurons[0].output, data->game_data[j].true_eval));


    }
    return error / data->move_num;
}

// start of temperary test code
////////////////////////////////////////////////////////////////////


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

    return 0;

    //test_basic_functions();
    //test_backpropagation();

    // create a fresh neural network
    struct neural_net *net = generate_new_network(32, 1, 5, 32);
    // save the network
    save_network_to_file(net, "neural_net/test_network");

    free(net);

    return 1;
}

