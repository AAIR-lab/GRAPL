

#include <common/utils.h>
#include <blai/utils.h>


std::ofstream get_ofstream(const std::string &filename) {
    std::ofstream stream(filename.c_str());
    if(stream.fail()) {
        throw std::runtime_error(Utils::error() + "opening file '" + filename + "'");
    }
    return stream;
}

std::ifstream get_ifstream(const std::string &filename) {
    std::ifstream stream(filename.c_str());
    if(stream.fail()) {
        throw std::runtime_error(Utils::error() + "opening file '" + filename + "'");
    }
    return stream;
}