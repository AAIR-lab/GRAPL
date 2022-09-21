
import random


from generalized_learning.qlearning.evaluator.base_evaluator import BaseEvaluator


class RandomPolicyEvaluator(BaseEvaluator):

    def __init__(self, phase_dict):

        super(RandomPolicyEvaluator, self).__init__(phase_dict)

    def get_action(self, problem, current_state):

        applicable_actions = problem.get_applicable_actions(current_state)
        return random.choice(applicable_actions)
