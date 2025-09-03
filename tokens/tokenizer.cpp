#include <sstream>

#include <iostream>
#include <fstream>
#include "tokenizer.hpp"
#include <iomanip>
#include <iterator>
#include <algorithm>
vectorClass<vectorClass<char>> tokenizeFile(const std::string& filename) {
		std::ifstream in(filename, std::ios::binary);
		if (!in) {
				throw std::runtime_error("Could not open file " + filename);
		}
		std::istreambuf_iterator<char> it (in);
		std::istreambuf_iterator<char> end;
		vectorClass<char> word;
		vectorClass<vectorClass<char>> tokens;
		for (; it != end; it++) {
				char c = *it;
				if (c == '.' || c == ' ' || c == '\n' || c == '?' || c == '!' || c == ';' || c == ':' || c == ',') {
						if (word.size() > 0) {
								tokens.push(word);
								word = vectorClass<char>();
						}
						
				}
				if (c != ' ' && c != '\n') {
						word.push(c);
				}
				

			  
				
		}
		if (word.size() > 0) {
				tokens.push(word);
		}
		return tokens;
}
