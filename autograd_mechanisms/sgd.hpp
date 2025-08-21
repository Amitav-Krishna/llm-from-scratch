#ifndef SGD_HPP
#define SGD_HPP

#include "autograd.hpp"  // Include Node, add, mul, etc.
#include <vector>

class SGD {
public:
    float lr; // learning rate

    SGD(float learning_rate) : lr(learning_rate) {}

    // Update parameters in-place
    void step(vectorClass<Node*> params);
};

#endif
