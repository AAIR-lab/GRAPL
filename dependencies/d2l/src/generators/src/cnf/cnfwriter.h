#pragma once

#include <utility>
#include <cstdint>
#include <vector>
#include <ostream>

using cnfvar_t = uint32_t;
using cnflit_t = int64_t;  // To make sure we can negate any variable
using cnfclause_t = std::vector<cnflit_t>;


class CNFWriter {
protected:
    uint32_t next_var_id_;

    ulong accumulated_weight_;

    ulong nclauses_;

    std::ostream& os_;

    std::ostream* varnamestream_;

public:
    // Variable IDs must start with 1
    explicit CNFWriter(std::ostream &os, std::ostream* varnamestream = nullptr)
        : next_var_id_(1), accumulated_weight_(0), nclauses_(0), os_(os), varnamestream_(varnamestream)
    {
    }

    cnfvar_t variable(const std::string& name = "") {
        const auto id = next_var_id_;
        if (varnamestream_ != nullptr) {
            *varnamestream_ << id << "\t" << name << std::endl;
        }
        next_var_id_ += 1;
        return id;
    }

    [[nodiscard]] uint32_t nvars() const { return next_var_id_ - 1; }

    [[nodiscard]] ulong nclauses() const { return nclauses_; }

    static inline cnflit_t literal(cnfvar_t var, bool polarity) {
        auto res = static_cast<cnflit_t>(var);
        return polarity ? res : -1*res;
    }

    [[nodiscard]] ulong top() const { return accumulated_weight_ + 1; }

    //! Print the given clause - if weight is negative, it is assumed to be TOP
    void print_clause(const cnfclause_t& clause, unsigned weight = std::numeric_limits<unsigned int>::max()) {
        //  w <literals> 0
        assert(!clause.empty());
        if (weight == 0) throw std::runtime_error("Cannot use weight-0 clauses");
        if (weight == std::numeric_limits<unsigned int>::max()) os_ << "TOP ";
        else {
            accumulated_weight_ += weight;
            os_ << weight << " ";
        }

        auto size = clause.size();
        for (unsigned i = 0; i < size; ++i) {
            os_ << clause[i] << " ";
        }
        os_ << "0" << std::endl;

        nclauses_ += 1;
    }

    //! A couple shorthands
    cnfvar_t var(const std::string& name) { return variable(name); }
    static inline cnflit_t lit(cnfvar_t var, bool polarity) { return literal(var, polarity); }
    inline void cl(const cnfclause_t& clause, unsigned weight = std::numeric_limits<unsigned int>::max()) { print_clause(clause, weight); }


};