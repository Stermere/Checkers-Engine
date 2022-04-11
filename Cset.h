// Author: Collin Kees

// a basic implimentation of a set to hold the location of pieces on a checkers board
// due to the nature of this task this is a specialized set that only holds ints and is 
// not meant to be used for anything else, as it does not check for a element before inserting it

// the set is implemented as an avl tree with a sentinel node at the root
#include <stdlib.h>

struct set {
    int value;
    struct set *right;
    struct set *left;
    struct set *parent;
};

// remove a value from the set
void set_remove(struct set *s, int x){
    // if the node is the entrypoint node then we need to get the next node (pointer is in the left child parameter)
    if (s->value == -1){
        // note: it is very important that the right child is used as the set add and remove can break if it is not (yes I could fix this but this is a faster fix and it works)
        set_remove(s->right, x);
    }
    // if the value at the node is the search value go through and see what condition it meets
    else if (s->value == x){
        // if the node has no children then just remove it
        if (s->right == NULL && s->left == NULL){
            if (s->parent->value > x){
                s->parent->left = NULL;
                free(s);
            } 
            else if (s->parent->value < x){
                s->parent->right = NULL;
                free(s);
            }
            else {
                printf("error: multiple entries with the same value exiting...");
                exit(1);
            }
        }

        // if one node is NULL just move its one child in to its place
        else if (s->right == NULL){
            if (s->parent->value > x){
                s->parent->left = s->left;
                s->left->parent = s->parent;
                free(s);
            } 
            else if (s->parent->value < x){
                s->parent->right = s->left;
                s->left->parent = s->parent;
                free(s);
            }
            else {
                printf("error: multiple entries with the same value exiting...");
                exit(1);
            }
        }
        // same thing just for the other child node
        else if (s->left == NULL){
            if (s->parent->value > x){
                s->parent->left = s->right;
                s->right->parent = s->parent;
                free(s);
            } 
            else if (s->parent->value < x){
                s->parent->right = s->right;
                s->right->parent = s->parent;
                free(s);
            }
            else {
                printf("error: multiple entries with the same value exiting...");
                exit(1);
            }
        }
        // if both children are not NULL then we need to find the proper way to update the pointers
        else {
            // geta temp struct for the node as removing a node and moving its children around
            struct set *insertion_point;
            // update the pointers
            if (s->parent->value > x){
                s->parent->left = s->left;
                s->left->parent = s->parent;
                insertion_point = s->left;
                while (insertion_point->right != NULL){
                    insertion_point = insertion_point->right;
                }
                insertion_point->right = s->right;
                s->right->parent = insertion_point;
                free(s);
            } 
            else if (s->parent->value < x){
                s->parent->right = s->right;
                s->right->parent = s->parent;
                insertion_point = s->right;
                while (insertion_point->left != NULL){
                    insertion_point = insertion_point->left;
                }
                insertion_point->left = s->left;
                s->left->parent = insertion_point;
                free(s);
            }
            else{
                printf("error: multiple entries with the same value exiting...");
                exit(1);
            }
        } 
    }
    else if (s->value > x){
        set_remove(s->left, x);
    }
    else if (s->value < x){
        set_remove(s->right, x);
    }
}

// add a value to the set
// takes the sentinel node and the value to add as arguments
void set_add(struct set *s, int x){
    // if the node is the entrypoint node then we need to get the next node (pointer is in the left child parameter)
    if (s->value == -1){
        if (s->right == NULL){
            s->right = malloc(sizeof(struct set));
            s->right->value = x;
            s->right->right = NULL;
            s->right->left = NULL;
            s->right->parent = s;
            return;
        }
        set_add(s->right, x);
    }

    if (s->value > x){
        if (s->left == NULL){
            s->left = malloc(sizeof(struct set));
            s->left->value = x;
            s->left->left = NULL;
            s->left->right = NULL;
            s->left->parent = s;
            return;
        }
        set_add(s->left, x);
    }

    if (s->value < x){
        if (s->right == NULL){
            s->right = malloc(sizeof(struct set));
            s->right->value = x;
            s->right->left = NULL;
            s->right->right = NULL;
            s->right->parent = s;
            return;
        }
        set_add(s->right, x);
    }
}

// rebalances the tree after a value is added or removed
// takes the sentinel node as an argument
// this function is not meant to be called directly, it is called by the set_add and set_remove functions
void rebalance_tree(){

}

// create a set
struct set* create_set(){
    // create a sentinel node
    struct set *s = malloc(sizeof(struct set));
    s->value = -1;
    s->right = NULL;
    s->left = NULL;
    return s;
}

// print the set
void print_set(struct set *s){
    if(s->value != -1){
        printf("%d ", s->value);
    }
    if(s->right != NULL){
        print_set(s->right);
    }
    if(s->left != NULL){
        print_set(s->left);
    }
}

int populate_array_recursive(struct set *s, int *array, int i, int depth){
    // add the value to the array and increment the index
    array[i] = s->value;
    i++;
    // check if the node has a right child
    if (s->right != NULL){
        // call the function recursively
        i = populate_array_recursive(s->right, array, i, depth + 1);
    }
    // check if the node has a left child
    if (s->left != NULL){
        // call the function recursively
        i = populate_array_recursive(s->left, array, i, depth + 1);
    }
    return i;
}

// populates the array with the values in the set and returns the number of values in the set
// takes a pointer to memory to store the array in as an argument
// and a set pointer that is the tree node we are at
int populate_array(struct set *s, int *array){
    // check if the array element given was the head
    if (s->value != -1){
        printf("error: array element was not of the proper type exiting...");
        exit(1);
    }
    else{
        // return the elements in the set
        return populate_array_recursive(s->right, array, 0, 0);
    }
}




