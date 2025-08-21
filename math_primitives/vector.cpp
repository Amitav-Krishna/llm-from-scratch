#include "vector.hpp"
#include <bits/stdc++.h>
#include "random.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

// ------------------- MATH VECTOR -------------------
void mathVector::scalarMultiplication(float scalar) {
    // In-place: overwrite existing elements
    for (int i = 0; i < size(); i++) {
        (*this)[i] *= scalar;
    }
}

mathVector mathVector::operator+(const mathVector& other) const {
    if (size() != other.size())
        THROW_INVALID_ARG("Vectors must be of the same dimension");

    mathVector result;
    result.reserve(size());
    for (int i = 0; i < size(); i++)
        result.push((*this)[i] + other[i]);
    return result;
}

mathVector mathVector::operator-(const mathVector& other) const {
    if (size() != other.size())
        THROW_INVALID_ARG("Vectors must be of the same dimension");

    mathVector result;
    result.reserve(size());
    for (int i = 0; i < size(); i++)
        result.push((*this)[i] - other[i]);
    return result;
}

float mathVector::dotProduct(const mathVector& other) const {
    if (size() != other.size())
        THROW_INVALID_ARG("Dot product dimensions must match");

    float val = 0.0f;
    for (int i = 0; i < size(); i++)
        val += (*this)[i] * other[i];
    return val;
}

// ------------------- MATRIX -------------------
mathVector matrix::shape() const {
    mathVector v;
    v.push(size());
    if (size() > 0) v.push((*this)[0].size());
    else v.push(0);
    return v;
}

matrix matrix::transpose() const {
    matrix m;
    if (size() == 0) return m;

    const int rows = size();
    const int cols = (*this)[0].size();

    for (int i = 0; i < cols; i++) {
        mathVector v;
        v.reserve(rows);
        for (int j = 0; j < rows; j++)
            v.push((*this)[j][i]);
        m.push(v);
    }
    return m;
}

void matrix::print() const {
    if (size() == 0) {
        std::cout << "(empty)\n";
        return;
    }
    const int cols = (*this)[0].size();
    for (int i = 0; i < size(); i++) {
        for (int j = 0; j < cols; j++)
            std::cout << (*this)[i][j] << '|';
        std::cout << std::endl;
    }
}

// ------------------- MATRIX OPERATIONS -------------------
matrix matrix::operator*(const matrix& other) const {
    if (size() == 0 || other.size() == 0)
        THROW_INVALID_ARG("Matrix multiply on empty matrix is undefined");

    const int r1 = size();
    const int c1 = (*this)[0].size();
    const int r2 = other.size();
    const int c2 = other[0].size();

    if (c1 != r2)
        throw std::invalid_argument("Incompatible shapes for matmul: (" +
                                    std::to_string(r1) + "x" + std::to_string(c1) + ") * (" +
                                    std::to_string(r2) + "x" + std::to_string(c2) + ")");

    matrix B_T = other.transpose();

    matrix result;
    result.reserve(r1);
    for (int i = 0; i < r1; i++) {
        const mathVector& rowA = (*this)[i];
        mathVector row;
        row.reserve(c2);
        for (int j = 0; j < c2; j++)
            row.push(rowA.dotProduct(B_T[j]));
        result.push(row);
    }
    return result;
}

matrix matrix::operator+(const matrix& other) const {
    if (!(shape() == other.shape())) {
				std::ostringstream oss;
				oss << "Matrix addition shape mismatch: "
						<< "lhs shape = " << shape()
						<< ", rhs shape = " << other.shape();
				throw std::invalid_argument(oss.str());
		}

    matrix result;
    result.reserve(size());
    for (int i = 0; i < size(); i++) {
        const mathVector& rowA = (*this)[i];
        const mathVector& rowB = other[i];
        mathVector row;
        row.reserve(rowA.size());
        for (int j = 0; j < rowA.size(); j++)
            row.push(rowA[j] + rowB[j]);
        result.push(row);
    }
    return result;
}

matrix matrix::operator-(const matrix& other) const {
    if (!(shape() == other.shape()))
        THROW_INVALID_ARG("Both dimensions must match for subtraction");

    matrix result;
    result.reserve(size());
    for (int i = 0; i < size(); i++) {
        const mathVector& rowA = (*this)[i];
        const mathVector& rowB = other[i];
        mathVector row;
        row.reserve(rowA.size());
        for (int j = 0; j < rowA.size(); j++)
            row.push(rowA[j] - rowB[j]);
        result.push(row);
    }
    return result;
}

matrix matrix::hadamard(const matrix& other) const {
    if (!(shape() == other.shape()))
        THROW_INVALID_ARG("Both dimensions must match for Hadamard product");

    matrix result;
    result.reserve(size());
    for (int i = 0; i < size(); i++) {
        const mathVector& rowA = (*this)[i];
        const mathVector& rowB = other[i];
        mathVector row;
        row.reserve(rowA.size());
        for (int j = 0; j < rowA.size(); j++)
            row.push(rowA[j] * rowB[j]);
        result.push(row);
    }
    return result;
}

// ------------------- RANDOM INITIALIZATION -------------------
void matrix::fill_uniform(Random& rng, float min, float max) {
    for (int i = 0; i < size(); i++) {
        mathVector& row = (*this)[i];
        for (int j = 0; j < row.size(); j++)
            row[j] = rng.uniform(min, max);
    }
}

void matrix::fill_zeroes() {
    for (int i = 0; i < size(); i++) {
        mathVector& row = (*this)[i];
        for (int j = 0; j < row.size(); j++)
            row[j] = 0;
    }
}

void matrix::fill_identity() {
    if (size() == 0 || (*this)[0].size() == 0)
        throw std::invalid_argument("Cannot fill empty matrix as identity");

    if (size() != (*this)[0].size())
        throw std::invalid_argument("Identity matrix must be square.");

    for (int i = 0; i < size(); i++) {
        mathVector& row = (*this)[i];
        for (int j = 0; j < row.size(); j++)
            row[j] = (i == j) ? 1.0f : 0.0f;
    }
}

void matrix::fill_xavier(Random& rng, int fan_in, int fan_out) {
    for (int i = 0; i < size(); i++) {
        mathVector& row = (*this)[i];
        for (int j = 0; j < row.size(); j++)
            row[j] = rng.xavier_uniform(fan_in, fan_out);
    }
}

matrix matrix::scalarMultiply(float scalar) const {
    matrix result;
    result.reserve(size());
    for (int i = 0; i < size(); i++) {
        const mathVector& row = (*this)[i];
        mathVector outputRow;
        outputRow.reserve(row.size());
        for (int j = 0; j < row.size(); j++)
            outputRow.push(row[j] * scalar);
        result.push(outputRow);
    }
    return result;
}

matrix matrix::scalarAddition(float scalar) const {
    matrix result;
    result.reserve(size());
    for (int i = 0; i < size(); i++) {
        const mathVector& row = (*this)[i];
        mathVector outputRow;
        outputRow.reserve(row.size());
        for (int j = 0; j < row.size(); j++)
            outputRow.push(row[j] + scalar);
        result.push(outputRow);
    }
    return result;
}
