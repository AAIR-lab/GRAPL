
#pragma once

#include <cassert>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <boost/functional/hash.hpp>
#include <common/base.h>


namespace sltp {

class TransitionSample {
public:
    using transition_list_t = std::vector<state_pair>;
    using transition_set_t = std::unordered_set<state_pair, boost::hash<state_pair>>;

protected:
    const std::size_t num_states_;
    const std::size_t num_transitions_;
    const std::size_t num_marked_transitions_;
    // trdata_[s] contains the IDs of all neighbors of s in the state space
    std::vector<std::vector<unsigned>> trdata_;
    std::vector<bool> is_state_alive_;

    std::vector<unsigned> alive_states_;

    std::vector<int> vstar_;

    transition_set_t marked_transitions_;
    transition_list_t unmarked_transitions_;
    transition_list_t unmarked_and_alive_transitions_;
    std::vector<transition_list_t> unmarked_transitions_from_;

public:
    TransitionSample(std::size_t num_states, std::size_t num_transitions, std::size_t num_marked_transitions)
            : num_states_(num_states),
              num_transitions_(num_transitions),
              num_marked_transitions_(num_marked_transitions),
              trdata_(num_states),
              is_state_alive_(num_states, false),
              alive_states_(),
              marked_transitions_(),
              unmarked_transitions_(),
              unmarked_and_alive_transitions_(),
              unmarked_transitions_from_(num_states)
     {
         if (num_states_ > std::numeric_limits<state_id_t>::max()) {
             throw std::runtime_error("Number of states too high - revise source code and change state_id_t datatype");
         }

         if (num_transitions_ > std::numeric_limits<transition_id_t>::max()) {
             throw std::runtime_error("Number of states too high - revise source code and change transition_id_t datatype");
         }
     }

    ~TransitionSample() = default;
    TransitionSample(const TransitionSample&) = default;
    TransitionSample(TransitionSample&&) = default;

    std::size_t num_states() const { return num_states_; }
    std::size_t num_transitions() const { return num_transitions_; }
    std::size_t num_marked_transitions() const { return num_marked_transitions_; }

    int vstar(unsigned sid) const { return vstar_.at(sid); }

    const std::vector<unsigned>& successors(unsigned s) const {
        return trdata_.at(s);
    }

    const transition_set_t& marked_transitions() const {
        return marked_transitions_;
    }

    const transition_list_t& unmarked_transitions() const {
        return unmarked_transitions_;
    }

    const transition_list_t& unmarked_and_alive_transitions() const {
        return unmarked_and_alive_transitions_;
    }

    const transition_list_t& unmarked_transitions_starting_at(unsigned s) const {
        return unmarked_transitions_from_[s];
    }

    bool marked(const state_pair& p) const {
        return marked_transitions_.find(p) != marked_transitions_.end();
    }
    bool marked(unsigned src, unsigned dst) const {
        return marked(std::make_pair(src, dst));
    }

    bool is_alive(unsigned state) const {
        return is_state_alive_.at(state);
    }

    const std::vector<unsigned>& all_alive() const { return alive_states_; }

    //! Print a representation of the object to the given stream.
    friend std::ostream& operator<<(std::ostream &os, const TransitionSample& o) { return o.print(os); }
    std::ostream& print(std::ostream &os) const {
        os << "Transition sample [states: " << num_states_ << ", transitions: " << num_transitions_;
        os << " (" << num_marked_transitions_ << " marked)]" << std::endl;
//        for (unsigned s = 0; s < num_states_; ++s) {
//            const auto& dsts = trdata_[s];
//            if (!dsts.empty()) os << "state " << s << ":";
//            for (auto dst:dsts) os << " " << dst;
//            os << std::endl;
//        }
        return os;
    }

    // readers
    void read(std::istream &is) {
        marked_transitions_.reserve(num_marked_transitions_);
        for(unsigned i = 0; i < num_marked_transitions_; ++i) {
            unsigned src = 0, dst = 0;
            is >> src >> dst;
            assert(src < num_states_ && dst < num_states_);
            marked_transitions_.emplace(src, dst);
        }

        // read number of states that have been expanded, for thich we'll have one state per line next
        unsigned num_records = 0;
        is >> num_records;
        unsigned n_total_transitions = 0;

        // read transitions, in format: source_id, num_successors, succ_1, succ_2, ...
        for( unsigned i = 0; i < num_records; ++i ) {
            unsigned src = 0, count = 0, dst = 0;
            is >> src >> count;
            assert(src < num_states_ && 0 <= count);
            if( count > 0 ) {
                std::vector<bool> seen(num_states_, false);
                trdata_[src].reserve(count);
                for( unsigned j = 0; j < count; ++j ) {
                    is >> dst;
                    assert(dst < num_states_);
                    if (seen.at(dst)) throw std::runtime_error("Duplicate transition");
                    trdata_[src].push_back(dst);
                    seen[dst] = true;

                    if (!marked(src, dst)) {
                        unmarked_transitions_.push_back(std::make_pair(src, dst));
                        unmarked_transitions_from_[src].push_back(std::make_pair(src, dst));
                    }
                    n_total_transitions++;
                }
            }
        }

        assert(n_total_transitions == unmarked_transitions_.size() + marked_transitions_.size());

        // Validate that marked transitions are indeed transitions
        for (const auto &marked : marked_transitions_) {
            unsigned src = marked.first;
            unsigned dst = marked.second;

            // Check that the marked transition is indeed a transition
            bool valid = false;
            for (unsigned t:trdata_[src]) {
                if (dst == t) {
                    valid = true;
                    break;
                }
            }

            if (!valid) {
                throw std::runtime_error("Invalid marked transition");
            }
        }

        // Store which states are alive (i.e. solvable, reachable, and not a goal)
        unsigned count = 0, s = 0;
        is >> count;
        assert(0 <= count && count <= num_states_);
        if(count > 0) {
            for(unsigned j = 0; j < count; ++j) {
                is >> s;
                assert(s < num_states_);
                is_state_alive_[s] = true;
                alive_states_.push_back(s);
            }
        }

        for (const auto tx:unmarked_transitions_) {
            if (is_state_alive_[tx.first]) unmarked_and_alive_transitions_.push_back(tx);
        }

        // Store the value of V^*(s) for each state s
        int vstar;
        vstar_.reserve(num_states_);
        for (s=0; s < num_states_; ++s) {
            is >> vstar;
            vstar_.push_back(vstar);
        }
    }

    static TransitionSample read_dump(std::istream &is, bool verbose) {
        unsigned num_states = 0, num_transitions = 0, num_marked_transitions = 0;
        is >> num_states >> num_transitions >> num_marked_transitions;
        TransitionSample transitions(num_states, num_transitions, num_marked_transitions);
        transitions.read(is);
        if( verbose ) {
            std::cout << "TransitionSample::read_dump: #states=" << transitions.num_states()
                      << ", #transitions=" << transitions.num_transitions()
                      << ", #marked-transitions=" << transitions.marked_transitions_.size()
                      << std::endl;
        }
        return transitions;
    }

    //! Project the m states in selected, assumed to be a subset of [0, n], to the range
    //! [0, m], applying the given mapping
    TransitionSample resample(const std::unordered_set<unsigned>& selected,
            const std::unordered_map<unsigned, unsigned>& mapping) const {

        auto nstates = mapping.size();
        unsigned ntransitions = 0;

        std::vector<std::vector<unsigned>> trdata(nstates);
        transition_set_t marked_transitions;
        std::vector<bool> alive_states(nstates, false);

        for (unsigned s:selected) {
            unsigned mapped_s = mapping.at(s);
            assert(mapped_s < nstates);

            if (is_state_alive_.at(s)) alive_states.at(mapped_s) = true;

            for (unsigned sprime:successors(s)) {
                // mapping must contain all successors of the states in selected
                assert(mapping.find(sprime) != mapping.end());
                auto mapped_sprime = mapping.at(sprime);

                trdata.at(mapped_s).push_back(mapped_sprime);
                if (marked(s, sprime)) {
                    marked_transitions.emplace(mapped_s, mapped_sprime);
                }

                ++ntransitions;
            }
        }

        TransitionSample transitions(nstates, ntransitions, marked_transitions.size());
        transitions.trdata_ = std::move(trdata);
        transitions.marked_transitions_ = std::move(marked_transitions);
        return transitions;
    }
};

} // namespaces
