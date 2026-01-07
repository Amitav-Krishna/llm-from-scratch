#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <fstream>
#include <sstream>
#include <iterator>
#include <iostream>
#include <iomanip>
#include "../math_primitives/vector.hpp"  // Or better: a vector.hpp

// A function that reads a file and returns tokens
MyList<MyList<char>> tokenizeFile(const std::string& filename);

#endif // TOKENIZER_HPP
