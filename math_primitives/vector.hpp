#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <initializer_list>
#include <stdexcept>
#include <iostream>
#include "random.hpp"
#define THROW_INVALID_ARG(msg) \
    throw std::invalid_argument(std::string(msg) + " (at " + __FILE__ + ":" + std::to_string(__LINE__) + ")")

// ------------------- TEMPLATE VECTOR -------------------
template <typename T>
class vectorClass {
protected:
    T* arr;
    int capacity;
    int current;

public:
    vectorClass() : capacity(1), current(0) {
        arr = new T[capacity];
    }


    vectorClass(const vectorClass& other) : capacity(other.capacity), current(other.current) {
        arr = new T[capacity];
        for (int i = 0; i < current; i++)
            arr[i] = other.arr[i];
    }

    vectorClass& operator=(const vectorClass& other) {
        if (this != &other) {
            delete[] arr;
            capacity = other.capacity;
            current = other.current;
            arr = new T[capacity];
            for (int i = 0; i < current; i++)
                arr[i] = other.arr[i];
        }
        return *this;
    }

    ~vectorClass() { delete[] arr; }

    void push(T data) {
        if (current == capacity) {
            T* temp = new T[2 * capacity];
            for (int i = 0; i < capacity; i++) temp[i] = arr[i];
            delete[] arr;
            capacity *= 2;
            arr = temp;
        }
        arr[current++] = data;
    }

    void push(T data, int index) {
        if (index == capacity) push(data);
        else arr[index] = data;
    }

    void pop() { if(current > 0) current--; }
    void clear() { current = 0; }
		void reserve(int new_capacity) {
				if (new_capacity > capacity) {
						T* temp = new T[new_capacity];
						for (int i = 0; i < current; i++)
								temp[i] = arr[i];
						delete[] arr;
						arr = temp;
						capacity = new_capacity;
				}
		}

    int size() const { return current; }
    int getcapacity() const { return capacity; }

    bool search(T key) const {
        for (int i = 0; i < size(); i++)
            if ((*this)[i] == key) return true;
        return false;
    }

    

    const T& operator[](int index) const {
        if (index >= current || index < 0)
            throw std::out_of_range("Index out of range");
        return arr[index];
    }

    bool operator==(const vectorClass& other) const {
        if (other.size() != size()) return false;
        for (int i = 0; i < size(); i++)
            if ((*this)[i] != other[i]) return false;
        return true;
    }
};

// ------------------- MATH VECTOR -------------------
class mathVector : public vectorClass<float> {
public:
    using vectorClass<float>::vectorClass;
    explicit mathVector(int n) {
        for (int i = 0; i < n; i++) push(0.0f);
    }
    friend std::ostream& operator<<(std::ostream& os, const mathVector& vec) {
        os << "[";
        for (int i = 0; i < vec.size(); i++) {
            os << vec[i];
            if (i != vec.size() - 1) os << ", ";
        }
        os << "]";
        return os;
    }
    mathVector(std::initializer_list<float> list) {  // NEW
        for (auto v : list) push(v);
    }
    void scalarMultiplication(float scalar);
    mathVector operator+(const mathVector& other) const;
    mathVector operator-(const mathVector& other) const;
    float dotProduct(const mathVector& other) const;
};

// ------------------- MATRIX -------------------

class matrix : public vectorClass<mathVector> {
public:
    using vectorClass<mathVector>::vectorClass;
    using vectorClass<mathVector>::size;
    using vectorClass<mathVector>::operator[];
    matrix() = default;

    matrix(int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            mathVector row;
            row.reserve(cols);
            for (int j = 0; j < cols; j++)
                row.push(0.0f);
            this->push(row);  // use the push function defined below
        }
    }
		matrix(const std::initializer_list<mathVector>& list) {
        for (const auto& row : list)
            this->push(row);
    }

    void push(const mathVector& row) {
        if(this->size() > 0) {
            if(row.size() != (*this)[0].size()) {
                throw std::invalid_argument("All rows in a matrix must have the same length");
            }
        }
        vectorClass<mathVector>::push(row);
    }
		mathVector shape() const;
    matrix transpose() const;
    void print() const;

    matrix operator*(const matrix& other) const;
    matrix operator+(const matrix& other) const;
    matrix operator-(const matrix& other) const;
    matrix hadamard(const matrix& other) const;

    void fill_uniform(Random& rng, float min, float max);
    void fill_xavier(Random& rng, int fan_in, int fan_out);
    void fill_zeroes();
    void fill_identity();

    matrix scalarMultiply(float scalar) const;
    matrix scalarAddition(float scalar) const;
};
#endif
