#include "random.hpp"
#include <cmath>
#include <stdexcept>

Random::Random(unsigned int seed) : gen(seed) {}

float Random::uniform(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

float Random::xavier_uniform(int fan_in, int fan_out) {
		if (fan_in + fan_out == 0) {
			  throw std::invalid_argument("fan_in + fan_out cannot be zero.");
		}
    float a = std::sqrt(6.0f / (fan_in + fan_out));
    return uniform(-a, a);
}
