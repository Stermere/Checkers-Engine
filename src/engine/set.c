// Author: Collin Kees

// a basic implimentation of a set to hold the location of pieces on a checkers board
// due to the nature of this task this it only holds numbers between 0 and 63 inclusive
// this is because the board is 8x8 and the numbers are stored in a 64 bit integer
// uses a bit set to store the values

#include <stdlib.h>
#include <stdint.h>

static inline int countTrailingZeros(unsigned long long x) {
#ifdef _WIN64
    return (int)_tzcnt_u64(x);
#else
    return __builtin_ctzll(x);
#endif
}


struct set {
    long long values;
    int array[64];
    int size;
};

void set_remove(struct set *s, int x){
    s->values &= ~(1ll << x);
}

void set_add(struct set *s, int x){
    s->values |= (1ll << x); 
}

// create a set
struct set* create_set(){
    // create a sentinel node
    struct set *s = malloc(sizeof(struct set));
    s->values = 0;
    s->size = 0;
    return s;
}

// returns the number of bits in the given long
// uses a bit hack to do this in constant time
int get_bits_set(long long i) {
    i = i - ((i >> 1) & 0x5555555555555555);
    i = (i & 0x3333333333333333) + ((i >> 2) & 0x3333333333333333);
    return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0F) * 0x101010101010101) >> 56;
}

// print the set
void print_set(struct set *s){
    printf("set: {");
    for (int i = 0; i < s->size; i++){
        printf("%d", s->array[i]);
        if (i != s->size - 1){
            printf(", ");
        }
    }
    printf("}\n");
}

// populates/updates the array with the values in the set and returns the number of values in the set
int populate_set_array(struct set *s){
    unsigned long long number = s->values;
    s->size = 0;
    while (number) {
        unsigned long long set_bit = number & -number; // Extracts the rightmost set bit
        int position = countTrailingZeros(set_bit); // Finds the position of the set bit
        s->array[s->size++] = position;
        number &= (number - 1); // Clear the rightmost set bit
    }

    return s->size;
}