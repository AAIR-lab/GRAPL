
#pragma once

#include <blai/transitions.h>
#include <blai/matrix.h>
#include "base.h"

[[nodiscard]] sltp::TransitionSample read_transition_data(const std::string& workspace, bool verbose);

[[nodiscard]] sltp::FeatureMatrix read_feature_matrix(const std::string& workspace, bool verbose);

[[nodiscard]] sltp::Sample read_input_sample(const std::string& workspace);

[[nodiscard]] std::vector<std::string> read_nominals(const std::string& workspace);

[[nodiscard]] int transition_sign(int s_f, int sprime_f);

[[nodiscard]] bool are_transitions_d1d2_distinguished(int s_f, int sprime_f, int t_f, int tprime_f);
