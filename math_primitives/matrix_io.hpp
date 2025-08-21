#ifndef MATRIX_IO_HPP
#define MATRIX_IO_HPP

#include <string>
#include "vector.hpp"

class MatrixIO {
public:
    static void saveBinary(const matrix& m, const std::string& filename);
    static matrix loadBinary(const std::string& filename);
};

#endif
