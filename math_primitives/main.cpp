#include <iostream>
#include "vector.hpp"
#include "random.hpp"
#include "matrix_io.hpp"
#include <chrono>
int main() {
    using namespace std;

    // Create two mathVectors
    mathVector v1; 
    v1.push(1); v1.push(2); v1.push(3);

    mathVector v2; 
    v2.push(4); v2.push(5); v2.push(6);

    // Create two matrices
    matrix m1; 
    m1.push(v1); 
    m1.push(v2);

    matrix m2; 
    m2.push(v2); 
    m2.push(v1); 
    m2.push(v1);  // irregular shape, but will test multiplication rules

    cout << "Matrix m1:" << endl;
    m1.print();
    cout << endl;

    cout << "Matrix m2:" << endl;
    m2.print();
    cout << endl;

    // Multiply matrices (auto-transpose if needed)
    try {
				auto start = std::chrono::high_resolution_clock::now();
    
				matrix m3 = m1 * m2;  // matrix multiplication
    
				auto end = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double, std::milli> duration_ms = end - start;

				std::cout << "m1 * m2 took " << duration_ms.count() << " ms" << std::endl;
    
				std::cout << "Result of m1 * m2:" << std::endl;
				m3.print();
				std::cout << std::endl;



    } catch (const exception& e) {
        cout << "Multiplication error: " << e.what() << endl;
    }

    // Random initialization
    Random rng(42);  // deterministic seed
    matrix m_rand;
    m_rand.push(mathVector(3));  // 1st row
    m_rand.push(mathVector(3));  // 2nd row
    m_rand.fill_uniform(rng, -1.0f, 1.0f);

    cout << "Randomly initialized matrix:" << endl;
    m_rand.print();
    cout << endl;

    // Xavier initialization
    matrix m_xavier;
    m_xavier.push(mathVector(3));
    m_xavier.push(mathVector(3));
    m_xavier.fill_xavier(rng, 3, 3);

    cout << "Xavier-initialized matrix:" << endl;
    m_xavier.print();
    cout << endl;
		

		MatrixIO::saveBinary(m1, "matrix1.bin");

		matrix loaded = MatrixIO::loadBinary("matrix1.bin");
		std::cout << "Loaded matrix:" << std::endl;
		loaded.print();


    return 0;
}
