
#include "cache.hxx"
#include <common/base.h>
#include <sltp/dl/elements.hxx>
#include <sltp/dl/types.hxx>

namespace sltp::dl {

const state_denotation_t &Cache::retrieveDLDenotation(
        const DLBaseElement &element, const State &state, std::size_t expected_size) const {
    const sample_denotation_t &sd = find_sample_denotation(element, expected_size);
    return *sd[state.id()];
}

const sample_denotation_t&
Cache::find_sample_denotation(const DLBaseElement& element, std::size_t expected_size) const {
        auto it = cache2_.find(element.id());
        assert (it != cache2_.end());
        assert(it->second != nullptr);
        assert(it->second->size() == expected_size);
        return *it->second;
    }

}