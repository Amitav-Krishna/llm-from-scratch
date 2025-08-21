#include "matrix_io.hpp"
#include <fstream>
#include <stdexcept>

void MatrixIO::saveBinary(const matrix& m, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for writing");

    int rows = m.size();
    int cols = (rows > 0) ? m[0].size() : 0;

    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out.write(reinterpret_cast<const char*>(&m[i][j]), sizeof(float));

    out.close();
}

matrix MatrixIO::loadBinary(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for reading");

    int rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    matrix m;
    for (int i = 0; i < rows; i++) {
        mathVector row(cols);
        for (int j = 0; j < cols; j++) {
            float val;
            in.read(reinterpret_cast<char*>(&val), sizeof(float));
            row[j] = val;
        }
        m.push(row);
    }

    in.close();
    return m;
}
