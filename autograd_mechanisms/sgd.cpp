#include "sgd.hpp"
#include "../phase_1/vector.hpp"

void SGD::step(vectorClass<Node*> params) {
    for (int i = 0; i < params.size(); i++) {
        // Gradient descent update: params[i] = params[i] - lr * grad
        params[i]->value = params[i]->value + params[i]->grad.scalarMultiply(-lr);

        // Optional: reset gradient to zero after update
        params[i]->grad.fill_zeroes();
    }
}
