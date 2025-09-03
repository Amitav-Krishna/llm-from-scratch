#include <iostream>
#include "tokenizer.hpp"
#include <iomanip>

int main() {
		std::string filename = "clean.txt";
		auto tokens = tokenizeFile(filename);

		if (tokens.size() < 2) {
				throw std::invalid_argument("Bi-gram requires at least 2 tokens");
		}
		vectorClass<vectorClass<vectorClass<char>>> pairs;
		vectorClass<vectorClass<char>> pair;
	  for (int i = 0; i < tokens.size()-1; i++) {
				pair.push(tokens[i]);
				pair.push(tokens[i+1]);
				pairs.push(pair);
				pair.clear();
		}

		for (int i = 0; i < pairs.size(); i++) {
				for (int j = 0; j < 2; j++) {
						for (int h = 0; h < pairs[i][j].size(); h++) {
								std::cout << pairs[i][j][h];
						}
						std::cout << ' ';
				}
				std::cout << '\n';
		}
		vectorClass<int> counts;
		for (int i = 0; i < pairs.size()-1; i++) {
				counts.push(0);
				for (int j = i+1; j < pairs.size(); j++) {
						if (pairs[i][0] == pairs[j][0] && pairs[i][1] == pairs[j][1]) {
								counts[i] = counts[i] + 1;
						}
				} 
		}
		

		return 0;
}
