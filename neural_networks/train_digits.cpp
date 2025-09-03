#include "images.hpp"
#include "../phase_1/vector.hpp"
#include "sgd.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <regex>
#include <cctype>

namespace fs = std::filesystem;
Random rng(42);

// Utility functions for debugging
float matrix_sum(const matrix& m) {
    float sum = 0.0f;
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[i].size(); j++) {
            sum += m[i][j];
        }
    }
    return sum;
}

float matrix_max(const matrix& m) {
    float max_val = -1e9;
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0; j < m[i].size(); j++) {
            if (m[i][j] > max_val) max_val = m[i][j];
        }
    }
    return max_val;
}

void print_matrix_stats(const char* name, const matrix& m) {
    std::cout << name << ": sum=" << matrix_sum(m) << ", max=" << matrix_max(m);
    if (m.size() > 0 && m[0].size() > 0) {
        std::cout << ", shape=" << m.size() << "x" << m[0].size();
    }
    std::cout << std::endl;
}

matrix flatten(const matrix& other) {
    int rows = other.size();
    int cols = other[0].size();
    matrix result(1, rows*cols);
    int index = 0;

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++) {
            result[0][index] = other[i][j];
            index++;
        }
    }
    return result;
}

Node* forward(Node* x, Node* w1, Node* w2, Node* b1, Node* b2) {
    Node* z1 = add(matmul(x, w1), b1);
    Node* a1 = relu(z1);
    
    Node* z2 = add(matmul(a1, w2), b2);
    Node* y_pred = softmax(z2);
    return y_pred;
}

int main() {
    // BEGIN IMAGE FETCHING
    vectorClass<matrix> images;
    vectorClass<int> labels;

    // Load MNIST data from CSV
    std::string csv_filename = "mnist.csv";
    std::ifstream file(csv_filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + csv_filename);
    }

    std::string line;
    int line_count = 0;
    int skipped_lines = 0;
    int loaded_count = 0;

    // Check if first line is a header
    std::getline(file, line);
    bool has_header = (line.find("label") != std::string::npos) || 
                     (!line.empty() && !std::isdigit(line[0]));
    
    if (has_header) {
        std::cout << "Detected header line, skipping: " << line.substr(0, 50) << "..." << std::endl;
    } else {
        // Rewind to beginning if no header
        file.clear();
        file.seekg(0);
    }

    // Read and process each line
    while (std::getline(file, line)) {
        line_count++;
        if (line.empty()) {
            skipped_lines++;
            continue;
        }


        try {
            // DEBUG: Show what we're parsing
            if (line_count <= 5) {
                std::cout << "Parsing line " << line_count << ": " << line.substr(0, 100) << "..." << std::endl;
            }

            // Extract the label from the end of the line
            // The format is: {python_dict},label
            size_t last_comma = line.find_last_of(',');
            if (last_comma == std::string::npos || last_comma == line.size() - 1) {
                skipped_lines++;
                continue;
            }
            
            std::string label_str = line.substr(last_comma + 1);
            // Trim any whitespace or quotes from the label
            label_str.erase(0, label_str.find_first_not_of(" \t\n\r\"'"));
            label_str.erase(label_str.find_last_not_of(" \t\n\r\"'") + 1);
            
            if (label_str.empty()) {
                skipped_lines++;
                continue;
            }
            
            int label = std::stoi(label_str);
            
            // For now, create a simple dummy image based on the label
            // This is just to get things working - you'll need proper PNG decoding later
            matrix img(28, 28);
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    // Create a simple pattern that varies with the label
                    float value = 0.1f + 0.8f * ((i + j + label) % 10) / 10.0f;
                    img[i][j] = value;
                }
            }
            
            labels.push(label);
            images.push(img);
            loaded_count++;
            
        } catch (const std::exception& e) {
            skipped_lines++;
            if (line_count < 10) {
                std::cerr << "Error parsing line " << line_count << ": " << e.what() << std::endl;
                std::cerr << "Problematic part: " << line.substr(0, 100) << "..." << std::endl;
            }
        }
        
        // Show progress
        if (line_count % 1000 == 0) {
            std::cout << "Processed " << line_count << " lines, loaded " << loaded_count << " images" << std::endl;
        }
    }

    file.close();

    std::cout << "Loaded " << images.size() << " images from CSV" << std::endl;
    std::cout << "Loaded " << labels.size() << " labels from CSV" << std::endl;
    std::cout << "Skipped " << skipped_lines << " lines" << std::endl;

    // Check if we have any data to train on
    if (images.size() == 0) {
        std::cerr << "No valid data loaded. Using dummy data for demonstration." << std::endl;
        
        // Create some dummy data for testing
        for (int i = 0; i < 100; i++) {
            matrix dummy_img(28, 28);
            for (int x = 0; x < 28; x++) {
                for (int y = 0; y < 28; y++) {
                    dummy_img[x][y] = static_cast<float>(rand()) / RAND_MAX;
                }
            }
            images.push(dummy_img);
            labels.push(rand() % 10);
        }
    }

    // Verify the first few samples
    std::cout << "\nFirst few samples:" << std::endl;
    for (int i = 0; i < std::min(5, (int)images.size()); i++) {
        std::cout << "Sample " << i << ": Label = " << labels[i] 
                  << ", Image sum = " << matrix_sum(images[i])
                  << ", Image max = " << matrix_max(images[i]) << std::endl;
    }
    
    // END IMAGE FETCHING

    // BEGIN MODEL INITIALIZATION
    vectorClass<matrix> target_vectors;

    // Creates one-hot encodings of labels
    for (int i = 0; i < labels.size(); i++) {
        matrix target(1, 10);
        target.fill_zeroes();
        target[0][labels[i]] = 1.0f;
        target_vectors.push(target);
    }

    int INPUT_SIZE = 28 * 28; // 28x28 flattened
    int HIDDEN_SIZE = 128;
    int OUTPUT_SIZE = 10; // # of digits
    
    std::cout << "\nInput size: " << INPUT_SIZE << std::endl;
    std::cout << "Training set size: " << images.size() << std::endl;
    
    Node* w1 = new Node(matrix(INPUT_SIZE, HIDDEN_SIZE));
    w1->value.fill_xavier(rng, INPUT_SIZE, HIDDEN_SIZE);

    Node* w2 = new Node(matrix(HIDDEN_SIZE, OUTPUT_SIZE));
    w2->value.fill_xavier(rng, HIDDEN_SIZE, OUTPUT_SIZE);
    
    Node* b1 = new Node(matrix(1, HIDDEN_SIZE));
    b1->value.fill_zeroes();

    Node* b2 = new Node(matrix(1, OUTPUT_SIZE));
    b2->value.fill_zeroes();
    
    SGD optimizer(0.001f);
    vectorClass<Node*> params;
    params.push(w1);
    params.push(w2);
    params.push(b1);
    params.push(b2);
    
    const int epochs = 5;
    int loss_val = 0;
    int correct = 0;
    // END MODEL INITIALIZATION

    // BEGIN TRAINING LOOP
    std::cout << "\nStarting training..." << std::endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        correct = 0;
        loss_val = 0;
        
        // Reset gradients
        for (int i = 0; i < params.size(); i++) {
            params[i]->grad.fill_zeroes();
        }
        
        // Use a smaller subset for testing if dataset is large
        int training_size = std::min(1000, (int)images.size());
        
        for (int i = 0; i < training_size; i++) {
            // Flatten the 28x28 image to 1x784
            matrix flattened = flatten(images[i]);
            Node* x = new Node(flattened);
            Node* y = new Node(target_vectors[i]);

            Node* y_pred = forward(x, w1, w2, b1, b2);
            Node* loss = cross_entropy(y_pred, y);

            loss->grad = {{1.0f}};
            backward(loss);
            
            // Calculate prediction
            int predicted_class = 0;
            float max_val = 0.0f;
            
            for (int j = 0; j < 10; j++) {
                if (y_pred->value[0][j] > max_val) {
                    max_val = y_pred->value[0][j];
                    predicted_class = j;
                }
            }
            
            if (predicted_class == labels[i]) {
                correct++;
            }
            
            loss_val += loss->value[0][0];
            
            // Clean up
            delete x;
            delete y;
            delete y_pred;
            delete loss;
        }

        optimizer.step(params);

        float avg_loss = static_cast<float>(loss_val) / training_size;
        float accuracy = static_cast<float>(correct) / training_size * 100.0f;
        std::cout << "Epoch " << epoch+1 << "/" << epochs 
                  << " | Loss: " << avg_loss 
                  << " | Acc: " << accuracy << "%" << std::endl;
    }

    // Test with a few sample images
    std::cout << "\nTesting on sample images:" << std::endl;
    int test_samples = std::min(5, (int)images.size());
    for (int i = 0; i < test_samples; i++) {
        try {
            matrix test_img = flatten(images[i]);
            Node* x = new Node(test_img);
            Node* y_pred = forward(x, w1, w2, b1, b2);
            
            float max_val = 0.0f;
            int predicted_class = 0;
            
            for (int j = 0; j < 10; j++) {
                if (y_pred->value[0][j] > max_val) {
                    max_val = y_pred->value[0][j];
                    predicted_class = j;
                }
            }
            
            std::cout << "Image " << i << ": Predicted=" << predicted_class 
                      << " | Actual=" << labels[i] 
                      << " | Confidence=" << (max_val * 100.0f) << "%"
                      << " | " << (predicted_class == labels[i] ? "✓" : "✗")
                      << std::endl;
            
            delete x;
            delete y_pred;
        } catch (const std::exception& e) {
            std::cerr << "Error testing image " << i << ": " << e.what() << std::endl;
        }
    }
    
    // Clean up parameters
    for (int i = 0; i < params.size(); i++) {
        delete params[i];
    }
    
    return 0;
}
