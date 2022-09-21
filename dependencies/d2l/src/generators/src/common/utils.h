
#pragma once

#include <string>
#include <fstream>

std::ofstream get_ofstream(const std::string &filename);
std::ifstream get_ifstream(const std::string &filename);
