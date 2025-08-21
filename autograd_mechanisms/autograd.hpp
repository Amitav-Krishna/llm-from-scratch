#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include <set>
#include <functional>
#include <string>
#include "../phase_1/vector.hpp"

// ---------------- NODE ----------------
struct Node {
    matrix value;
    matrix grad;
    std::function<void()> backward;
    vectorClass<Node*> children;

    Node() = default;
    explicit Node(const matrix& m);
    explicit Node(float val);
};

// ---------------- OPERATIONS ----------------
Node* add(Node* x, Node* y);
Node* mul(Node* x, Node* y);
Node* matmul(Node* x, Node* y);
Node* relu(Node* x);
Node* square(Node* x);
Node* softmax(Node* x);

// ---------------- BACKWARD ----------------
void backward(Node* node);

// ---------------- MSE LOSS ----------------
Node* mse(Node* predictions, Node* targets);
Node* cross_entropy(Node* predictions, Node* targets);
// ---------------- GRAPH PRINTER ----------------
void print_graph(Node* node, std::string prefix="", std::set<Node*>* visited = nullptr);

#endif
