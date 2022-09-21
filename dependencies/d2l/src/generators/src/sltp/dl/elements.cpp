
#include <sltp/algorithms.hxx>
#include "elements.hxx"

namespace sltp::dl {

unsigned long DLBaseElement::global_id = 0;


int DistanceFeature::value(const Cache &cache, const Sample &sample, const State &state) const {
    assert(sample.state(state.id()).id() == state.id());
    const auto m = sample.num_states();
    if (!valid_cache_) {
        const sample_denotation_t& start_d = cache.find_sample_denotation(*start_, m);
        const sample_denotation_t& end_d = cache.find_sample_denotation(*end_, m);
        const sample_denotation_t& role_d = cache.find_sample_denotation(*role_, m);

        cached_distances_ = std::vector<int>(m, std::numeric_limits<int>::max());
        for (unsigned i = 0; i < m; ++i ) {
            const auto n = sample.num_objects(i);

            const state_denotation_t& start_sd = start_d.get(i, n);
            const state_denotation_t& end_sd = end_d.get(i, n);
            const state_denotation_t& role_sd = role_d.get(i, n * n);
            int distance = compute_distance(n, start_sd, end_sd, role_sd);
            cached_distances_[i] = distance;
        }
        valid_cache_ = true;
    }
    return cached_distances_[state.id()];
}


} // namespaces