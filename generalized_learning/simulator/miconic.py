
import random

from generalized_learning.concretized.state import State


class MiconicSimulator:

    def __init__(self):

        pass

    def apply_action(self, problem, current_state, action):

        if action.get_name() in ["up", "down"] and random.random() <= 0.2:

            next_state = State(current_state.get_atom_set())
        else:
            next_state = action.apply(current_state)

        done = False

        if problem.is_goal_satisfied(next_state):

            done = True

        return next_state, -1, done
