import random

from generalized_learning.qlearning.evaluator.base_evaluator import BaseEvaluator
import numpy as np


class ConcreteQEvaluator(BaseEvaluator):

    def __init__(self, phase_dict):

        super(ConcreteQEvaluator, self).__init__(phase_dict)

    def setup_globals(self, problem, **kwargs):

        self.q_table = kwargs["q_table"]

    def get_action(self, problem, current_state, **kwargs):

        applicable_actions = problem.get_applicable_actions(current_state)
        random.shuffle(applicable_actions)

        q_values = []
        for action in applicable_actions:

            dict_key = (current_state, action)

            q_value = self.q_table.get(dict_key, float("inf"))
            q_values.append(q_value)

        return applicable_actions[np.argmin(q_values)]
