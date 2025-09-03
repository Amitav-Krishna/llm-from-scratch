#include <iostream>
#include <cmath>
#include <vector>

int hash_code(std::vector<char> chars) {
		int hash;
		int a{33};
		for (int i = 0; i < chars.size(); i++) {
				hash += static_cast<unsigned>(chars[i]) * (pow(a, i));
		}
		return hash;
}

int main() {
		char x = 'a';
		char y = 'b';
		char z = 'c';
		std::vector<char> a;
		a = {x, y, z};
		int b = hash_code(a);
		std::cout << b << std::endl;
		return 0;
}
