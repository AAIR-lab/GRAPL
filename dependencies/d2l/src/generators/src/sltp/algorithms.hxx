
#pragma once

namespace sltp {

class state_denotation_t;

//! Compute the distance from the concept denotation represented in start_sd to that represented in end_sd,
//! along the role denotation represented in role_sd
int compute_distance(unsigned num_objects,
        const state_denotation_t &start_sd, const state_denotation_t &end_sd, const state_denotation_t &role_sd);

}  // namespaces