
#pragma once

#include <sltp/dl/types.hxx>

#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sltp {
    class State;
}

namespace sltp::dl {

class DLBaseElement;


// We cache sample and state denotations. The latter are cached
// into a simple hash (i.e. unordered_set). The former are cached
// using two hash maps (i.e. unordered_map): one that maps sample
// denotations to concept names, and the other that maps concept
// names to sample denotations.
//
// We also cache atoms that are used to represent states in the sample.
class Cache {
public:
    struct cache_support_t {
        // cache for sample denotations
        bool operator()(const sample_denotation_t *d1, const sample_denotation_t *d2) const {
            assert((d1 != nullptr) && (d2 != nullptr));
            assert(d1->size() == d2->size()); // number of states is fixed in sample
            return *d1 == *d2;
        }

        size_t operator()(const sample_denotation_t *obj) const {
            assert(obj != nullptr);
            size_t hash = (*this)((*obj)[0]);
            for (auto elem:*obj)
                hash = hash ^ (*this)(elem);
            return hash;
        }

        // cache for state denotations
        bool operator()(const state_denotation_t *sd1, const state_denotation_t *sd2) const {
            assert((sd1 != nullptr) && (sd2 != nullptr));
            // dimension of sd1 and sd2 may not be equal since they may
            // be for states with different number of objects, or for
            // denotation of concept/roles
            return *sd1 == *sd2;
        }

        size_t operator()(const state_denotation_t *obj) const {
            assert(obj != nullptr);
            std::hash<std::vector<bool> > hasher;
            return hasher(*static_cast<const std::vector<bool>*>(obj));
        }
    };

    //! A map from sample denotations to the concept/role ID with that denotation
    using cache1_t = std::unordered_map<const sample_denotation_t*, unsigned long, cache_support_t, cache_support_t>;

    //! A map from concept/role IDs to their denotation
    using cache2_t = std::unordered_map<unsigned long, const sample_denotation_t*>;

    //! A set of state denotations that we'll keep as a register, to avoid having duplicate state_denotation_t
    //! objects around.
    using cache3_t = std::unordered_set<const state_denotation_t*, cache_support_t, cache_support_t>;

protected:
    cache1_t cache1_;
    cache2_t cache2_;
    cache3_t cache3_;

public:
    Cache() = default;
    ~Cache() = default;

    // cache1: (full) sample denotations for concepts
    //
    // We maintain a hash of (full) sample denotations so that one copy
    // exists for each such denotations. This hash is implemented with
    // two tables, one that provides mappings from sample denotations
    // to concept names, and the other is the inverse
    //
    // sample denotation -> concept name
    const cache1_t& cache1() const {
        return cache1_;
    }

    bool contains(const sample_denotation_t* d) const {
        return cache1_.find(d) != cache1_.end();
    }

    const sample_denotation_t* find_or_insert_sample_denotation(const sample_denotation_t &d, unsigned long id) {
        auto it = cache1_.find(&d);
        if( it == cache1_.end() ) {
            assert(cache2_.find(id) == cache2_.end());
            const sample_denotation_t *nd = new sample_denotation_t(d);
            cache1_.emplace(nd, id);
            cache2_.emplace(id, nd);
            return nd;
        } else {
            return it->first;
        }
    }

    // Return the sample denotation corresponding to the DL element with given ID
    const sample_denotation_t& find_sample_denotation(const DLBaseElement& element, std::size_t expected_size) const;

    //! Find or insert the state denotation with given extension; and return it.
    const state_denotation_t* find_or_insert_state_denotation(const state_denotation_t &sd) {
        auto it = cache3_.find(&sd);
        if( it == cache3_.end() ) {
            const state_denotation_t *nsd = new state_denotation_t(sd);
            cache3_.insert(nsd);
            return nsd;
        } else {
            return *it;
        }
    }

    const state_denotation_t& retrieveDLDenotation(
            const DLBaseElement& element, const State& state, std::size_t expected_size) const;
};


} // namespaces
