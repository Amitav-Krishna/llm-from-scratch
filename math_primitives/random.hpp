#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>

class Random {
private:
    std::mt19937 gen;  // Mersenne Twister RNG engine

public:
    explicit Random(unsigned int seed);
    
    // Generate uniform random float in [min, max)
    float uniform(float min, float max);
    
    // Xavier initialization: uniform random in [-a, a] with a = sqrt(6 / (fan_in + fan_out))
    float xavier_uniform(int fan_in, int fan_out);
};

#endif
