'''
Created on Oct 21, 2020

@author: anonymous
'''

from collections import deque
from collections import namedtuple
import copy
import random

import torch
from torch.distributions import Categorical

from concretized.problem import Problem
from generalized_learning.abstraction.canonical import CanonicalAbstractionMLP
from generalized_learning.abstraction.concrete import ConcreteMLP
from generalized_learning.abstraction.description_logic import DescriptionLogicMLP
from neural_net.nn import NN
import numpy as np
import torch.nn.functional as F
from util import constants
from util import file
from util import state_explorer


abstraction_func = None

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()


class A2C:

    DEFAULTS = {
        "num_episodes": 1000000,
        "timesteps_per_episode": 250,
        "epsilon": 1.0,
        "min_epsilon": 0.01,
        "epsilon_decay_rate": 0.9,
        "replay_memory_size": 2000,
        "batch_size": 32,
        "gamma": 1.0,
        "model_dir": None,
        "abstraction": "description_logic_mlp",
        "feature_file": None,
        "model_dir": None,

    }

    def _get_value(self, config_dict, key):

        try:

            return config_dict[key]
        except KeyError:

            return A2C.DEFAULTS[key]

    def __init__(self, parent_dir, phase_dict):

        self._parent_dir = parent_dir
        self._phase_dict = phase_dict

        self.reset()

        self.saved_actions = []
        self.rewards = []

    def reset(self):

        self._num_episodes = self._get_value(
            self._phase_dict,
            "num_episodes")

        self._timesteps_per_episode = self._get_value(
            self._phase_dict,
            "timesteps_per_episode")

        self._epsilon = self._get_value(
            self._phase_dict,
            "epsilon")

        self._min_epsilon = self._get_value(
            self._phase_dict,
            "min_epsilon")

        self._epsilon_decay_rate = self._get_value(
            self._phase_dict,
            "epsilon_decay_rate")

        replay_memory_size = self._get_value(
            self._phase_dict,
            "replay_memory_size")
        self._replay_memory = deque(maxlen=replay_memory_size)

        self._batch_size = self._get_value(
            self._phase_dict,
            "batch_size")

        self._gamma = self._get_value(
            self._phase_dict,
            "gamma")

        self._abstraction = self._get_value(
            self._phase_dict,
            "abstraction")

        self._feature_file = self._get_value(
            self._phase_dict,
            "feature_file")

        self._behavior_policy_dir = self._get_value(
            self._phase_dict,
            "model_dir")

        if self._behavior_policy_dir is not None:
            self._behavior_policy_dir = file.get_relative_path(
                self._behavior_policy_dir,
                self._parent_dir)

    def prepare_abstraction(self, domain_filepath, problem_list):

        if self._abstraction == "description_logic_mlp":

            abstraction = ConcreteMLP.create(
                domain_filepath,
                problem_list)
        else:

            raise Exception("Unknown abstraction function")

        abstraction.initialize_nn(nn_inputs=["state"],
                                  nn_outputs=["action", "v_s"])
        return abstraction

    def _sample_action(self, abstraction, problem, current_state, nn_output_pkg):

        action_vector = nn_output_pkg.decode("action")

        action_scores = []
        applicable_actions = problem.get_applicable_actions(current_state)

        for action in applicable_actions:

            index = abstraction.get_action_index(action.get_name())
            action_scores.append(action_vector[index])

        # Convert the action scores to a tensor.
        action_scores = torch.tensor(action_scores).float()

        m = Categorical(action_scores)

        action = m.sample()

        value = nn_output_pkg.decode("v_s")
        self.saved_actions.append(SavedAction(m.log_prob(action), value))

        return applicable_actions[action.item()]

    def get_action(self, problem, current_state, abstraction, nn):

        abstract_state = abstraction.create_abstract_state(problem,
                                                           current_state)

        nn_input_pkg = abstraction.encode_nn_input(
            abstract_state=abstract_state)

        state_tensor = nn_input_pkg.decode("state")
        state_tensor = torch.tensor(state_tensor).float()

        action_vector, value = nn._model(state_tensor)

        action_scores = []
        applicable_actions = problem.get_applicable_actions(current_state)

        string = ""

        for action in applicable_actions:

            index = abstraction.get_action_index(str(action))
            action_scores.append(action_vector[index])
            string += "(%s, %.2f, %.2f)" % (str(action),
                                            action_vector[index], value)

        # Convert the action scores to a tensor.
        action_scores = torch.tensor(action_scores).float()

        m = Categorical(action_scores)

        action = m.sample()

        self.saved_actions.append(SavedAction(m.log_prob(action), value))

        print("Selected", applicable_actions[action.item()], " | ", string)
        return applicable_actions[action.item()], False

    def train(self, nn):

        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self._gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.mse_loss(value, torch.tensor([R])))

        # reset gradients
        nn._optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        nn._optimizer.step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

        pass

    def apply_action(self, problem, current_state, action):

        next_state = action.apply(current_state)
        done = False
        reward = 0

        if problem.is_goal_satisfied(next_state):

            done = True
            reward = 1

        return next_state, reward, done

    def simulate(self, problem, abstraction, nn):

        current_state = problem.get_initial_state()

        time_step = 0
        done = False
        while not done \
                and time_step < self._timesteps_per_episode:

            time_step += 1

            action, _ = self.get_action(problem, current_state,
                                        abstraction, nn)
            next_state, reward, done = self.apply_action(problem,
                                                         current_state,
                                                         action)

            self.rewards.append(reward)

            current_state = next_state

        self.train(nn)

        return time_step

    def qlearn(self, domain_filepath, problem_filepath, abstraction, nn):

        problem = Problem(domain_filepath.name, problem_filepath.name,
                          problem_filepath.parent)

        if self._behavior_policy is not None:

            self._behavior_policy._abstract_domain._cache_problem(problem)

        for episode in range(self._num_episodes):

            time_steps = self.simulate(problem, abstraction, nn)

            print("******** [EPISODE] Episode %3u ended in %3u timesteps with epsilon %.2f" % (
                episode,
                time_steps,
                self._epsilon))

            if len(self._replay_memory) >= self._batch_size:
                self._epsilon *= self._epsilon_decay_rate
                self._epsilon = max(self._min_epsilon, self._epsilon)

    def run(self, input_dir):

        problem_list = file.get_file_list(input_dir,
                                          constants.PROBLEM_FILE_REGEX)

        domain_list = file.get_file_list(input_dir,
                                         constants.DOMAIN_FILE_REGEX)

        assert len(domain_list) == 1
        domain_filepath = domain_list[0]

        if self._behavior_policy_dir is not None:

            self._behavior_policy = NN.load(
                self._behavior_policy_dir, "py_a2c")
            nn = NN.load(self._behavior_policy_dir, "py_a2c")
            abstraction = nn._abstract_domain

        else:

            abstraction = self.prepare_abstraction(domain_filepath,
                                                   problem_list)
            nn = NN.get_instance(abstraction, "generic", "py_a2c")
            self._behavior_policy = None

        for problem_filepath in problem_list:

            self.qlearn(domain_filepath, problem_filepath, abstraction, nn)

            self._replay_memory.clear()
            nn.soft_save(input_dir)

            import tensorflow as tf
            tf.keras.backend.clear_session()

            self._behavior_policy = NN.load(input_dir, "py_a2c")
