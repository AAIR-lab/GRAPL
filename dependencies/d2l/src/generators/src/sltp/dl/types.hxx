
#pragma once

#include <cassert>
#include <vector>


namespace sltp {


// A state denotation for concept C is a subset of objects
// implemented as a bitmap (ie. vector<bool>). These are
// cached so that at most one copy of a bitmap exists.
struct state_denotation_t : public std::vector<bool> {
    state_denotation_t(size_t n, bool value) : std::vector<bool>(n, value) { }
    [[nodiscard]] size_t cardinality() const {
        size_t n = 0;
        for( size_t i = 0; i < size(); ++i )
            n += (*this)[i];
        return n;
    }
};

// A (full) sample denotation for concept C is a vector of
// state denotations, one per each state in the sample.
// Since we cache state denotations, a sample denotation
// is implemented as a vector of pointers to bitmaps.
struct sample_denotation_t : public std::vector<const state_denotation_t*> {
    [[nodiscard]] const state_denotation_t& get(std::size_t i, std::size_t expected_size) const {
        const state_denotation_t *sd = this->operator[](i);
        assert((sd != nullptr) && (sd->size() == expected_size));
        return *sd;
    }
};

using feature_state_denotation_t = int;
using feature_sample_denotation_t = std::vector<feature_state_denotation_t>;


}  // namespaces