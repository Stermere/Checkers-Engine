// Author: Collin Kees

// a basic implimentation of a set to hold the location of pieces on a checkers board
// due to the nature of this task this is a specialized set that only holds ints and is 
// not meant to be used for anything else, as it does not check for a element before inserting it

// the set is implemented as a linked list, with each node holding a int and a pointer to the next node
# include <stdlib.h>

struct set {
    int value;
    struct set *next;
};

// remove a value from the set
void remove(struct set *s, int x){
    int flag = 1;
    struct set *temp = s;
    s = s->next;
    while(flag){
        if (s->value == x){
            temp->next = s->next;
            free(s);
            flag = 0;
        }
        else{
            temp = s;
            s = s->next;
        }
    }
}

// add a value to the set
// this function assumes your value is not already in the set
// so it is an unsafe add
void add(struct set *s, int x){
    struct set *new_ = (struct set*)malloc(sizeof(struct set));
    new_->value = x;
    new_->next = s;
}

struct set* create_set(){
    struct set *s = (struct set*)malloc(sizeof(struct set));
    s->next = NULL;
    s-> value = -1;
    return s;
}





