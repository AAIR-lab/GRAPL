
#pragma once

#include <boost/functional/hash.hpp>

namespace sltp::utils {

template<typename Container>
struct container_hash {
    std::size_t operator()(const Container &c) const {
        return boost::hash_range(c.begin(), c.end());
    }
};

}  // namespaces