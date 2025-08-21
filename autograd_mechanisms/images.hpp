#ifndef IMAGES_HPP
#define IMAGES_HPP

#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include "../phase_1/vector.hpp"
#include "autograd.hpp"

matrix load_ascii_image(const std::string& filepath);
int extract_label(const std::string& filename);
#endif
