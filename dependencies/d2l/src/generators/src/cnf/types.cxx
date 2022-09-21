
#include "types.h"


namespace sltp::cnf {

std::size_t hash_value(const transition_denotation& x) {
    std::size_t seed = 0;
    boost::hash_combine(seed, x.value_s_gt_0);
    boost::hash_combine(seed, x.increases);
    boost::hash_combine(seed, x.decreases);
    return seed;
}


std::size_t hash_value(const transition_pair& x) {
    std::size_t seed = 0;
    boost::hash_combine(seed, x.tx1);
    boost::hash_combine(seed, x.tx2);
    return seed;
}


bool operator<(const transition_pair& x, const transition_pair& y) {
    return std::tie(x.tx1, x.tx2) < std::tie(y.tx1, y.tx2);
}

} // namespaces
