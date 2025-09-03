#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <algorithm>

#include <zlib.h>

int main () {
		// Creates a file buffer
		std::ifstream in("image.png", std::ios::binary);

		
		std::istreambuf_iterator<char> end;

		// Outputs each byte and increments the iterator by 1 until the iterator is an end-of-stream iterator
		for (; it != end; ++it) {
				
				std::cout << std::setw(2) << (static_cast<unsigned>(*it) & 0xFF) << " ";

				
		}
		return 0;
}
