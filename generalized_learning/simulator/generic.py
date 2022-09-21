

class GenericSimulator:

    def __init__(self):

        pass

    def apply_action(self, problem, current_state, action):

        next_state = action.apply(current_state)

        done = False

        if problem.is_goal_satisfied(next_state):

            done = True

        return next_state, -1, done
