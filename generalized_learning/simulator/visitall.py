
import random

from generalized_learning.concretized.state import State


class VisitallSimulator:

    def __init__(self):

        pass

    def apply_action(self, problem, current_state, action):

        # Visitall has a single action "move".
        if random.random() <= 0.2:

            next_state = State(current_state.get_atom_set())
        else:
            next_state = action.apply(current_state)

        done = False

        if problem.is_goal_satisfied(next_state):

            done = True

        return next_state, -1, done
