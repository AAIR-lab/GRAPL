import heapq
import itertools
import random

import torch
from torch.distributions import Categorical

from generalized_learning.qlearning.evaluator.base_evaluator import BaseEvaluator
from generalized_learning.qlearning.ran import RAN
from neural_net.nn import NN
import torch.nn.functional as F
from util import file


class PolicyEvaluator(BaseEvaluator):

    def __init__(self, phase_dict, parent_dir):

        super(PolicyEvaluator, self).__init__(phase_dict)
        self._parent_dir = parent_dir

        ran_dir = file.get_relative_path(
            self._phase_dict["model_dir"],
            self._parent_dir)

        self.ran = RAN.load(ran_dir, self._phase_dict["nn_name"])

    def _get_action_score(self, abstraction, abstract_state, action,
                          nn_output_pkg, max_action_params):

        action_score = 1.0
        param_list = action.get_param_list()

        for i in range(max_action_params):

            if i < len(param_list):

                obj_name = param_list[i]
            else:

                obj_name = None

            param_vector = torch.sigmoid(
                nn_output_pkg.decode("action_param_%u" % (i)))

            count = abstraction.get_correct_param_count(abstract_state,
                                                        obj_name,
                                                        param_vector)
            action_score *= count / len(param_vector)

        # Make it negative so that we can do a heappop to get the best one.
        return -action_score

    def get_best_scoring_action(self, abstraction, abstract_state,
                                matched_actions,
                                nn_output_pkg):

        max_action_params = abstraction.get_max_action_params()

        scores = []
        counter = itertools.count()
        for matched_action in matched_actions:

            action_score = self._get_action_score(abstraction, abstract_state,
                                                  matched_action,
                                                  nn_output_pkg,
                                                  max_action_params)

            heapq.heappush(
                scores,
                (action_score, next(counter), matched_action))

        action_score, _, action = heapq.heappop(scores)
        return action

    def get_action(self, problem, current_state):

        abstraction = self.ran.get_abstraction()
        policy_nn = self.ran.policy_nn

        abstract_state = abstraction.create_abstract_state(
            problem,
            current_state)

        nn_input_pkg = abstraction.encode_nn_input(
            policy_nn.nn_input_call_set,
            abstract_state=abstract_state,
            nn_inputs=policy_nn.nn_inputs,
            nn_outputs=policy_nn.nn_outputs)

        nn_output_pkg = policy_nn.fit_pkgs([nn_input_pkg])[0]

        action_probabilities = F.softmax(
            nn_output_pkg.decode("action"), dim=-1)
        action_probabilities = Categorical(action_probabilities)

        action_index = int(action_probabilities.sample())
        action_name = abstraction.get_index_action(
            action_index)

        applicable_actions = problem.get_applicable_actions(current_state)
        random.shuffle(applicable_actions)

        matched_actions = []
        for action in applicable_actions:

            if action.get_name() == action_name:

                matched_actions.append(action)

        if len(matched_actions) == 0:

            return random.choice(applicable_actions)
        else:

            return self.get_best_scoring_action(abstraction, abstract_state,
                                                matched_actions,
                                                nn_output_pkg)
