
from gym import spaces
from generalized_learning.neural_net.nn import NNPkg


class Env:

    def __init__(self, abstract_domain, problem):

        self.abstraction = abstract_domain
        self.action_space = spaces.Discrete(
            abstract_domain.get_nn_shape("action"))
        self.observation_space = spaces.MultiBinary(
            abstract_domain.get_nn_shape("state"))
        self.reward_range = [-1, 0]
        self.metadata = {}
        self.problem = problem

    def reset(self):

        self.state = self.problem.get_initial_state()

        nn_pkg = NNPkg()
        self.abstraction._encode_state(nn_pkg, abstract_state=self.state)
        return nn_pkg.decode("state")

    def step(self, action_index):

        action_name = self.abstraction._index_action_dict[action_index]
        action = self.problem.get_action(action_name)

        reward = -1
        done = False

        if action.is_applicable(self.state):

            self.state = action.apply(self.state)

            if self.problem.is_goal_satisfied(self.state):

                print(self.state)
                print("goal")
                reward = 0
                done = True

        nn_pkg = NNPkg()
        self.abstraction._encode_state(nn_pkg, abstract_state=self.state)

        return nn_pkg.decode("state"), reward, done, {}
