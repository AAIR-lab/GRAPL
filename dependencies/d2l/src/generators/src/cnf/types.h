
#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/functional/hash.hpp>
#include <common/base.h>

namespace sltp::cnf {

struct transition_pair {
    transition_pair(transition_id_t tx1, transition_id_t tx2) :
            tx1(tx1), tx2(tx2)
    {}

    transition_id_t tx1;
    transition_id_t tx2;
};

bool operator<(const transition_pair& x, const transition_pair& y);


struct transition_denotation {
    uint8_t value_s_gt_0:1;
    uint8_t increases:1;
    uint8_t decreases:1;

    transition_denotation()
        : value_s_gt_0(0), increases(0), decreases(0)
    {}

    transition_denotation(bool value_s_gt_0, int sign)
        : value_s_gt_0(value_s_gt_0), increases((sign > 0) ? 1 : 0), decreases((sign < 0) ? 1 : 0)
    {}

    [[nodiscard]] bool nils() const { return increases == 0 && decreases == 0; }
};
static_assert(sizeof(transition_denotation) == 1);

inline bool operator==(const transition_denotation& x, const transition_denotation& y) {
    return x.value_s_gt_0 == y.value_s_gt_0 && x.increases == y.increases && x.decreases == y.decreases;
}

inline std::ostream& operator<<(std::ostream &os, const transition_denotation& o) {
    os << (unsigned) o.value_s_gt_0 << ":";
    if (o.nils()) os << "N";
    else if (o.increases) os << "U";
    else os << "D";
    return os;
}


struct transition_trace {
//    std::array<transition_denotation, 40> data; // This could be useful if we decide to compile the code on the fly

    //! A list of denotations, one denotation for every transition
    std::vector<transition_denotation> denotations;

    transition_trace()
        : denotations()
    {}

    explicit transition_trace(unsigned nfeatures)
        : denotations(nfeatures)
    {}
};

inline bool operator==(const transition_trace& x, const transition_trace& y) { return x.denotations == y.denotations; }

inline std::ostream& operator<<(std::ostream &os, const transition_trace& o) {
    auto sz = o.denotations.size();
    for (unsigned i = 0; i < sz; ++i) {
        os << o.denotations[i];
        if (i < sz-1) os << " ";
    }
    return os;
}

std::size_t hash_value(const transition_pair& x);
std::size_t hash_value(const transition_denotation& x);

} // namespaces

// Specializations of std::hash
namespace std {

template<> struct hash<sltp::cnf::transition_pair> {
    std::size_t operator()(const sltp::cnf::transition_pair& x) const {
        return hash_value(x);
    }
};

template<> struct hash<sltp::cnf::transition_denotation> {
    std::size_t operator()(const sltp::cnf::transition_denotation& x) const {
        return hash_value(x);
    }
};

template<> struct hash<sltp::cnf::transition_trace> {
    std::size_t operator()(const sltp::cnf::transition_trace& x) const {
        return boost::hash_range(x.denotations.begin(), x.denotations.end());
    }
};
}