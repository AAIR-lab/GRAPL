
#include <common/helpers.h>
#include "generator.h"
#include "types.h"

namespace sltp::cnf {

//! Return a sorted vector with those features that d1-distinguish s from t
std::vector<feature_t> compute_d1_distinguishing_features(const TrainingSet& sample, unsigned s, unsigned t) {
    std::vector<unsigned> features;
    const auto& mat = sample.matrix();
    for (unsigned f = 0; f < mat.num_features(); ++f) {
        auto sf = mat.entry(s, f);
        auto tf = mat.entry(t, f);
        if ((sf == 0) != (tf == 0)) {
            features.push_back(f);
        }
    }
    return features;
}

//! Return a sorted vector with those features that d2-distinguish transition (s, s') from (t, t')
std::vector<feature_t> compute_d2_distinguishing_features(const TrainingSet& sample,
                                                          unsigned s, unsigned sprime, unsigned t, unsigned tprime) {

    std::vector<unsigned> features;
    const auto& mat = sample.matrix();

    for (unsigned f = 0; f < mat.num_features(); ++f) {
        // Store those features that d2-distinguish (s, s') from (t, t'), but do _not_ d1-distinguish s from t
        int sf = mat.entry(s, f);
        int tf = mat.entry(t, f);
        if ((sf == 0) != (tf == 0)) continue; // f d1-distinguishes s from t

        int sprime_f = mat.entry(sprime, f);
        int tprime_f = mat.entry(tprime, f);

        if (transition_sign(sf, sprime_f) != transition_sign(tf, tprime_f)) {
            features.push_back(f);
        }
    }

    return features;
}

//! Return a sorted vector with those features that either d1-distinguish or d2-distinguish (s, s') from (t, t')
std::vector<feature_t> compute_d1d2_distinguishing_features(
        const std::vector<unsigned>& feature_ids,
        const TrainingSet& sample,
        unsigned s, unsigned sprime,
        unsigned t, unsigned tprime)
{
    std::vector<unsigned> features;
    const auto& mat = sample.matrix();

    for (unsigned f:feature_ids) {
        if (are_transitions_d1d2_distinguished(
                mat.entry(s, f), mat.entry(sprime, f), mat.entry(t, f), mat.entry(tprime, f))) {
            features.push_back(f);
        }
    }

    return features;
}


//! Check whether t appears isomorphic to s, and in that case, add it to the given list of isomorphisms
void check_isomorphic(const TrainingSet& sample, unsigned s, unsigned t, isomorphism_t& isomorphisms) {
    // if either s or t are isomorphic of some other state no need to recheck, will be detected in due time
    if (isomorphisms.find(s) != isomorphisms.end() || isomorphisms.find(t) != isomorphisms.end()) return;

    auto distinguishing = compute_d1_distinguishing_features(sample, s, t);
    if (!distinguishing.empty()) return; // s and t are distinguishable, ergo not isomorphic

    if (sample.matrix().goal(s) != sample.matrix().goal(t)) {
        // Only one of the two is a goal: the SAT theory will be unsat
        undist_goal_warning(s, t);
        return;
    }

    if (sample.is_deadend(s) != sample.is_deadend(t)) {
        // Only one of the two is a deadend: the SAT theory will be unsat
        undist_deadend_warning(s, t);
        return;
    }

    if (all_tx_have_analogs(sample, s, t) && all_tx_have_analogs(sample, t, s)) {
        isomorphisms.emplace(t, s);  // t can be pruned in favor of s
    }
}


isomorphism_t compute_redundant_states(const TrainingSet& sample) {
    isomorphism_t isomorphisms;

    // Collect optimal and non-optimal states
    std::unordered_set<unsigned> optimal_states;
    for (const auto& tx:sample.transitions().marked_transitions()) {
        optimal_states.insert(tx.first);
    }

    std::vector<unsigned> nonoptimal_states;
    for (unsigned s=0; s < sample.transitions().num_states(); ++s) {
        if (optimal_states.find(s) == optimal_states.end()) nonoptimal_states.push_back(s);
    }


    // Check isomorphism between an optimal and a non-optimal state
    for (unsigned s:optimal_states) {
        for (unsigned t:nonoptimal_states) {
            if (s != t) {
                check_isomorphic(sample, s, t, isomorphisms);
            }
        }
    }

    // Check isomorphism between two non-optimal states
    for (unsigned i=0; i < nonoptimal_states.size(); ++i) {
        for (unsigned j=i+1; j < nonoptimal_states.size(); ++j) {
            unsigned s = nonoptimal_states[i], t = nonoptimal_states[j];
            check_isomorphic(sample, s, t, isomorphisms);
        }
    }
    return isomorphisms;
}

bool all_tx_have_analogs(const TrainingSet& sample, unsigned s, unsigned t) {

    for (unsigned sprime:sample.transitions().successors(s)) {
        bool tx_has_analog = false;
        for (unsigned tprime:sample.transitions().successors(t)) {
            auto d2distinguishing = compute_d2_distinguishing_features(sample, s, sprime, t, tprime);
            if (d2distinguishing.empty()) tx_has_analog = true;
        }

        if (!tx_has_analog) return false;
    }
    return true;
}

} // namespaces
