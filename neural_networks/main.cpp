#include "../autograd_mechanisms/autograd.hpp"
#include "../autograd_mechanisms/sgd.hpp"
#include <iostream>
#include <vector>

int main() {
    try {
        // Simple dataset: y = 2x + 1
        std::vector<float> X_data = {1.0f, 2.0f, 3.0f};
        std::vector<float> Y_data = {3.0f, 5.0f, 7.0f};

        // Initialize parameters
        Node* w = new Node(matrix{{0.0f}});
        Node* b = new Node(matrix{{0.0f}});

        std::vector<Node*> params = {w, b};
        SGD optimizer(0.01f);  // learning rate

        const int epochs = 5000;

        for (int epoch = 0; epoch < epochs; epoch++) {
            float loss_val = 0.0f;

						for (Node* param : params)
								param->grad.fill_zeroes();


						
            for (size_t i = 0; i < X_data.size(); i++) {
                Node* x = new Node(matrix{{X_data[i]}});
                Node* y_true = new Node(matrix{{Y_data[i]}});

                // Forward pass: y_pred = w * x + b
								
                Node* y_pred = add(matmul(w, x), b);
                // Compute loss using MSE (automatically sets up backward)
								
								
								
								
								
                Node* loss = mse(y_pred, y_true);
								

                // Initialize gradient of final node
                loss->grad = matrix{1, 1};
								loss->grad.fill_zeroes();
                loss->grad[0][0] = 1.0f;
								
								
                // Backward pass
								
                backward(loss);
								
								
                // Accumulate loss for printing
                loss_val += loss->value[0][0];

								

                // Clean up intermediate nodes
                delete x;
                delete y_true;
                delete y_pred;
                delete loss;
            }

            // Update parameters
            optimizer.step(params);

            std::cout << "Epoch " << epoch << ", Loss: " << loss_val / X_data.size() << std::endl;
						std::cout << "Weights " << w->value[0][0] << ", Bias" << b->value[0][0] << std::endl;
        }

        std::cout << "Trained w: " << w->value[0][0] << ", b: " << b->value[0][0] << std::endl;

        // Clean up parameters
        delete w;
        delete b;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception caught!" << std::endl;
        return 1;
    }
}
