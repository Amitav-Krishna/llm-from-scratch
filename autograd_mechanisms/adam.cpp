#include "autograd.hpp"
#include <vector>
#include <cmath>

struct Adam {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    std::vector<matrix> m; // first moment
    std::vector<matrix> v; // second moment
    int t; // timestep

    Adam(float learning_rate = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : lr(learning_rate), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void step(std::vector<Node*>& params) {
        if (m.size() != params.size()) {
            // initialize m and v with zeros
            m.clear();
            v.clear();
            for (Node* p : params) {
                m.push_back(matrix(p->grad.size(), p->grad[0].size()));
                m.back().fill_zeroes();
                v.push_back(matrix(p->grad.size(), p->grad[0].size()));
                v.back().fill_zeroes();
            }
        }

        t += 1;

        for (size_t i = 0; i < params.size(); i++) {
            Node* p = params[i];
            matrix& mi = m[i];
            matrix& vi = v[i];

            int rows = p->grad.size();
            int cols = p->grad[0].size();

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    float g = p->grad[r][c];

                    // Update biased first moment estimate
                    mi[r][c] = beta1 * mi[r][c] + (1 - beta1) * g;

                    // Update biased second raw moment estimate
                    vi[r][c] = beta2 * vi[r][c] + (1 - beta2) * g * g;

                    // Compute bias-corrected estimates
                    float m_hat = mi[r][c] / (1 - std::pow(beta1, t));
                    float v_hat = vi[r][c] / (1 - std::pow(beta2, t));

                    // Update parameter
                    p->value[r][c] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }

            // Optional: reset gradients after update
            p->grad.fill_zeroes();
        }
    }
};
