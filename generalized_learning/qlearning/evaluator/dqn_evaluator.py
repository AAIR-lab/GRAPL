import random

from generalized_learning.qlearning.evaluator.base_evaluator import BaseEvaluator
import numpy as np


class DQNEvaluator(BaseEvaluator):

    @staticmethod
    def update_q_values(applicable_actions, current_state, nn,
                        problem, q_values, weight):

        abstract_state = nn._abstract_domain.create_abstract_state(
            problem,
            current_state)

        nn_pkg_list = []
        for i in range(len(applicable_actions)):

            nn_input_pkg = nn._abstract_domain.encode_nn_input(
                nn.nn_input_call_set,
                abstract_state=abstract_state,
                action=applicable_actions[i])
            nn_pkg_list.append(nn_input_pkg)

        nn_output_pkgs = nn.fit_pkgs(nn_pkg_list)
        for i in range(len(nn_output_pkgs)):

            q_values[i] += weight * nn_output_pkgs[i].decode("q_s_a")

    def __init__(self, phase_dict):

        super(DQNEvaluator, self).__init__(phase_dict)

    def get_action(self, problem, current_state, **kwargs):

        raise NotImplementedError

        nn_wts = kwargs["nns"]

        applicable_actions = problem.get_applicable_actions(current_state)
        random.shuffle(applicable_actions)

        q_values = np.zeros(len(applicable_actions))

        for nn, weight in nn_wts:

            DQNEvaluator.update_q_values(applicable_actions,
                                         current_state,
                                         nn,
                                         problem,
                                         q_values,
                                         weight)

        return applicable_actions[np.argmax(q_values)]
