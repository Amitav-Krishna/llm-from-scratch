#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include "../phase_1/vector.hpp"
#include <functional>
#include <iostream>
#include <cmath>
#include <set>

// ---------------- NODE ----------------
struct Node {
    matrix value;
    matrix grad;
    std::function<void()> backward;
    vectorClass<Node*> children;

    Node() = default;

    explicit Node(const matrix& m) {
        value = m;
        grad = matrix(m.size(), m[0].size());
        grad.fill_zeroes();
    }

    explicit Node(float scalar) {
        value = matrix(1,1);
        value[0][0] = scalar;
        grad = matrix(1,1);
        grad.fill_zeroes();
    }
};

// ---------------- GRAPH PRINT ----------------
void print_graph(Node* node, std::string prefix = "", std::set<Node*>* visited = nullptr) {
    if (!node) return;
    if (!visited) visited = new std::set<Node*>();
    if (visited->count(node)) return;
    visited->insert(node);

    std::cout << prefix << "Node@" << node
              << " | value shape = [" 
              << node->value.size() 
              << ", " << (node->value.size() > 0 ? node->value[0].size() : 0)
              << "] | grad shape = ["
              << node->grad.size() 
              << ", " << (node->grad.size() > 0 ? node->grad[0].size() : 0)
              << "] | value = ";
    node->value.print();
    
    for (int i = 0; i < node->children.size(); i++)
        print_graph(node->children[i], prefix + "  ", visited);
}

// ---------------- OPERATIONS ----------------
Node* add(Node* x, Node* y) {
    Node* z = new Node(x->value + y->value);
    z->children.push(x);
    z->children.push(y);
    z->backward = [=]() {
        if (x->grad.size() != x->value.size() || x->grad[0].size() != x->value[0].size() ||
            y->grad.size() != y->value.size() || y->grad[0].size() != y->value[0].size() ||
            z->grad.size() != z->value.size() || z->grad[0].size() != z->value[0].size()) {
            throw std::runtime_error("Grad/value shape mismatch in add backward");
        }
        x->grad = x->grad + z->grad;
        y->grad = y->grad + z->grad;
    };
    return z;
}

Node* mul(Node* x, Node* y) {
    Node* z = new Node(x->value.hadamard(y->value));
    z->children.push(x);
    z->children.push(y);
    z->backward = [=]() {
        x->grad = x->grad + y->value.hadamard(z->grad);
        y->grad = y->grad + x->value.hadamard(z->grad);
    };
    return z;
}

Node* matmul(Node* x, Node* y) {
    Node* z = new Node(x->value * y->value);
    
    z->children.push(x);
    z->children.push(y);
    z->backward = [=]() {
        x->grad = x->grad + z->grad * y->value.transpose();
        y->grad = y->grad + x->value.transpose() * z->grad;
    };
    return z;
}

Node* relu(Node* x) {
    Node* z = new Node(x->value);
    z->children.push(x);
    z->backward = [=]() {
        for (int i = 0; i < x->value.size(); i++)
            for (int j = 0; j < x->value[i].size(); j++)
                x->grad[i][j] += (x->value[i][j] > 0 ? 1.0f : 0.0f) * z->grad[i][j];
    };
    return z;
}

Node* square(Node* x) {
    Node* z = new Node(x->value.hadamard(x->value));
    z->children.push(x);
    z->backward = [=]() {
        x->grad = x->grad + x->value.scalarMultiply(2.0f).hadamard(z->grad);
    };
    return z;
}

// ---------------- SOFTMAX ----------------
Node* softmax(Node* x) {
    Node* z = new Node(x->value);
    z->children.push(x);

    for (int i = 0; i < z->value.size(); i++) {
        float max_val = -1e9;
        for (int j = 0; j < z->value[i].size(); j++)
            if (z->value[i][j] > max_val) max_val = z->value[i][j];

        float sum = 0.0f;
        for (int j = 0; j < z->value[i].size(); j++) {
            z->value[i][j] = std::exp(z->value[i][j] - max_val);
            sum += z->value[i][j];
        }
        for (int j = 0; j < z->value[i].size(); j++)
            z->value[i][j] /= sum;
    }

    z->backward = [=]() {
        for (int i = 0; i < x->value.size(); i++) {
            for (int j = 0; j < x->value[i].size(); j++) {
                float sum = 0.0f;
                for (int k = 0; k < x->value[i].size(); k++) {
                    float jacobian_term = z->value[i][j] * ((j == k) ? 1.0f : 0.0f - z->value[i][k]);
                    sum += z->grad[i][k] * jacobian_term;
                }
                x->grad[i][j] += sum;
            }
        }
    };
    
    return z;
}

// ---------------- BACKWARD ----------------
void backward(Node* node) {
    if (node->backward) {
        node->backward();
        for (int i = 0; i < node->children.size(); i++)
            backward(node->children[i]);
    }
}

// ---------------- MSE ----------------
Node* mse(Node* predictions, Node* targets) {
    // Ensure gradients initialized
    predictions->grad = matrix(predictions->value.size(), predictions->value[0].size());
    targets->grad = matrix(targets->value.size(), targets->value[0].size());
    predictions->grad.fill_zeroes();
    targets->grad.fill_zeroes();

    // diff = predictions - targets
    Node* neg_targets = new Node(targets->value.scalarMultiply(-1.0f));
    neg_targets->grad = matrix(neg_targets->value.size(), neg_targets->value[0].size());
    neg_targets->grad.fill_zeroes();
    
    Node* diff = add(predictions, neg_targets);
    diff->grad = matrix(diff->value.size(), diff->value[0].size());
    diff->grad.fill_zeroes();

    Node* diff_sq = square(diff);
    diff_sq->grad = matrix(diff_sq->value.size(), diff_sq->value[0].size());
    diff_sq->grad.fill_zeroes();

    int rows = predictions->value.size();
    int cols = predictions->value[0].size();
    float scale = 1.0f / (rows * cols);
    
    // Replace the backward function for MSE
    diff_sq->backward = [=]() {
        float factor = 2.0f * scale;
        
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) {
                float d = predictions->value[i][j] - targets->value[i][j];
                predictions->grad[i][j] += factor * d;
                targets->grad[i][j] -= factor * d;
            }
    };
    
    return diff_sq;
}
// ---------------- CROSS ENTROPY LOSS ----------------
Node* cross_entropy(Node* predictions, Node* targets) {
    // Ensure predictions and targets have the same shape
    if (predictions->value.size() != targets->value.size() || 
        predictions->value[0].size() != targets->value[0].size()) {
        throw std::runtime_error("Predictions and targets must have the same shape in cross_entropy");
    }

    // Ensure gradients are initialized
    predictions->grad = matrix(predictions->value.size(), predictions->value[0].size());
    targets->grad = matrix(targets->value.size(), targets->value[0].size());
    predictions->grad.fill_zeroes();
    targets->grad.fill_zeroes();

    int batch_size = predictions->value.size();
    int num_classes = predictions->value[0].size();

    // Apply softmax to predictions
    Node* softmax_probs = softmax(predictions);
    softmax_probs->grad = matrix(batch_size, num_classes);
    softmax_probs->grad.fill_zeroes();

    // Create loss node
    Node* loss_node = new Node(0.0f);
    loss_node->grad = matrix(1, 1);
    loss_node->grad.fill_zeroes();
    loss_node->children.push(softmax_probs);

    // Compute cross entropy loss: -sum(targets * log(softmax_probs)) / batch_size
    float loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            if (targets->value[i][j] == 1.0f) {  // Only consider the true class
                loss += -std::log(softmax_probs->value[i][j] + 1e-8f);  // Add epsilon for numerical stability
            }
        }
    }
    loss /= batch_size;
    loss_node->value[0][0] = loss;

    // Set up backward pass
    loss_node->backward = [=]() {
        float scale = 1.0f / batch_size;
        
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < num_classes; j++) {
                // Gradient of cross entropy with respect to softmax outputs:
                // ∂L/∂softmax_j = (softmax_j - target_j) / batch_size
                softmax_probs->grad[i][j] += scale * (softmax_probs->value[i][j] - targets->value[i][j]);
            }
        }
        
        // Trigger backward pass through softmax
        if (softmax_probs->backward) {
            softmax_probs->backward();
        }
    };

    return loss_node;
}

// Alternative version: Cross entropy with logits (more numerically stable)
Node* cross_entropy_with_logits(Node* logits, Node* targets) {
    // Ensure logits and targets have the same shape
    if (logits->value.size() != targets->value.size() || 
        logits->value[0].size() != targets->value[0].size()) {
        throw std::runtime_error("Logits and targets must have the same shape in cross_entropy_with_logits");
    }

    // Ensure gradients are initialized
    logits->grad = matrix(logits->value.size(), logits->value[0].size());
    targets->grad = matrix(targets->value.size(), targets->value[0].size());
    logits->grad.fill_zeroes();
    targets->grad.fill_zeroes();

    int batch_size = logits->value.size();
    int num_classes = logits->value[0].size();

    // Compute numerically stable softmax cross entropy
    Node* loss_node = new Node(0.0f);
    loss_node->grad = matrix(1, 1);
    loss_node->grad.fill_zeroes();
    loss_node->children.push(logits);

    // Compute loss: -sum(targets * (logits - logsumexp(logits))) / batch_size
    float loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        // Find max for numerical stability
        float max_logit = -1e9f;
        for (int j = 0; j < num_classes; j++) {
            if (logits->value[i][j] > max_logit) {
                max_logit = logits->value[i][j];
            }
        }
        
        // Compute logsumexp
        float logsumexp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            logsumexp += std::exp(logits->value[i][j] - max_logit);
        }
        logsumexp = max_logit + std::log(logsumexp);
        
        // Compute loss for this sample
        for (int j = 0; j < num_classes; j++) {
            if (targets->value[i][j] == 1.0f) {  // Only consider the true class
                loss += -(logits->value[i][j] - logsumexp);
            }
        }
    }
    loss /= batch_size;
    loss_node->value[0][0] = loss;

    // Set up backward pass
    loss_node->backward = [=]() {
        float scale = 1.0f / batch_size;
        
        // Compute softmax probabilities for gradient calculation
        matrix softmax_probs(batch_size, num_classes);
        for (int i = 0; i < batch_size; i++) {
            float max_val = -1e9f;
            for (int j = 0; j < num_classes; j++) {
                if (logits->value[i][j] > max_val) max_val = logits->value[i][j];
            }
            
            float sum = 0.0f;
            for (int j = 0; j < num_classes; j++) {
                softmax_probs[i][j] = std::exp(logits->value[i][j] - max_val);
                sum += softmax_probs[i][j];
            }
            for (int j = 0; j < num_classes; j++) {
                softmax_probs[i][j] /= sum;
            }
        }
        
        // Gradient: ∂L/∂logits = (softmax - targets) / batch_size
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < num_classes; j++) {
                logits->grad[i][j] += scale * (softmax_probs[i][j] - targets->value[i][j]);
            }
        }
    };

    return loss_node;
}
#endif
