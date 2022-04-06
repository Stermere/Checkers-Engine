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
void set_remove(struct set *s, int x){
    // if the set head value is the value to remove do nothing
    if (x == -1) return;

     // remove the value from the set
     struct set *temp = s;
     while(temp->next != NULL){
         if (temp->next->value == x){
             struct set *temp2 = temp->next;
             temp->next = temp->next->next;
             free(temp2);
         }
         else{
             temp = temp->next;
         }
     }

}

// add a value to the set
// this function assumes your value is not already in the set
// so it is an unsafe add
void set_add(struct set *s, int x){
    // make a new node and give it the value x and put it in the begining of the set
    struct set *new_ = malloc(sizeof(struct set));
    new_->value = x;
    struct set* temp = s->next;
    s->next = new_;
    new_->next = temp;
}

// create a set
struct set* create_set(){
    struct set *s = malloc(sizeof(struct set));
    s->next = NULL;
    s-> value = -1;
    return s;
}

// print the set
void print_set(struct set *s){
    struct set *temp = s->next;
    while(temp != NULL){
        printf("%d ", temp->value);
        temp = temp->next;
    }
    printf("\n");
}





