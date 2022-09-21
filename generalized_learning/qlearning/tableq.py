'''
Created on Oct 21, 2020

@author: anonymous
'''

from collections import deque
import copy
import random

from abstraction.domain import AbstractDomain
from abstraction.state import AbstractState
from concretized.problem import Problem
from neural_net.nn import NN
import numpy as np
from util import constants
from util import file
from util import reproducibility
from util import state_explorer


class TableQ:

    def __init__(self, num_episodes=500,
                 timesteps_per_episode=75,
                 epsilon=1.0,
                 min_epsilon=0.01,
                 epsilon_decay_rate=0.9,
                 alpha=0.3,
                 gamma=0.9):

        self._num_episodes = num_episodes
        self._timesteps_per_episode = timesteps_per_episode
        self._epsilon = epsilon
        self._min_epsilon = min_epsilon
        self._epsilon_decay_rate = epsilon_decay_rate
        self._alpha = alpha
        self._gamma = gamma
        self._q_table = {}

    def get_action(self, problem, current_state):

        applicable_actions = problem.get_applicable_actions(current_state)
        assert len(applicable_actions) > 0
        random_action = random.sample(applicable_actions, 1)[0]

        string = ""
        for action in applicable_actions:

            string += "%s " % (action)

        if random.random() <= self._epsilon:

            print("[RANDOM]", random_action, "A:", string)
            return random_action
        else:

            q_action = self.get_max_q_value(problem, current_state)[1]
            print("[     Q]", q_action, "A:", string)
            return q_action

    def apply_action(self, problem, current_state, action):

        next_state = action.apply(current_state)
        reward = 0
        done = False

        if problem.is_goal_satisfied(next_state):

            reward = 1
            done = True

        return next_state, reward, done

    def get_max_q_value(self, problem, state):

        BASE_COST = 0

        abstract_state = AbstractState(problem, state)
        applicable_actions = problem.get_applicable_actions(state)
        random.shuffle(applicable_actions)
        random_action = random.sample(applicable_actions, 1)[0]

        max_action = None

        try:

            max_q_value = BASE_COST
            q_s_dict = self._q_table[abstract_state]
            for action in problem.get_applicable_actions(state):

                try:

                    q_s_a_dict = q_s_dict[action]

                    role_tuple = ()
                    for param in action.get_param_list():

                        role = abstract_state.get_role(param)
                        role_tuple += (role, )

                    if q_s_a_dict[role_tuple] > max_q_value:

                        max_q_value = q_s_a_dict[role_tuple]
                        max_action = action

                except Exception:

                    pass
        except Exception:

            pass

        if max_action is not None:

            return max_q_value, max_action
        else:

            return BASE_COST, random_action

    def q_update(self, problem, current_state, action, next_state, reward):

        BASE_COST = 0

        current_abstract_state = AbstractState(problem, current_state)

        role_tuple = ()
        for param in action.get_param_list():

            role = current_abstract_state.get_role(param)
            role_tuple += (role, )

        q_s_dict = self._q_table.setdefault(current_abstract_state, {})
        q_s_a_dict = q_s_dict.setdefault(action, {})
        q_s_a_value = q_s_a_dict.get(role_tuple, BASE_COST)

        max_q_s_dash = self.get_max_q_value(problem, next_state)[0]

        q_s_a_value += self._alpha * \
            (reward + self._gamma * (max_q_s_dash - q_s_a_value))

        q_s_a_dict[role_tuple] = q_s_a_value

    def simulate(self, problem):

        current_state = problem.get_initial_state()

        time_step = 0
        done = False
        while not done \
                and time_step < self._timesteps_per_episode:

            time_step += 1

            action = self.get_action(problem, current_state)
            next_state, reward, done = self.apply_action(problem,
                                                         current_state,
                                                         action)

            self.q_update(problem, current_state, action, next_state, reward)

            current_state = next_state

        return time_step

    def qlearn(self, domain_filepath, problem_filepath):

        problem = Problem(domain_filepath.name, problem_filepath.name,
                          problem_filepath.parent)

        for episode in range(self._num_episodes):

            time_steps = self.simulate(problem)

            print("******** [EPISODE] Episode %3u ended in %3u timesteps with epsilon %.2f" % (
                episode,
                time_steps,
                self._epsilon))

            self._epsilon *= self._epsilon_decay_rate
            self._epsilon = max(self._min_epsilon, self._epsilon)

    def run(self, input_dir):

        reproducibility.set_random_seeds()
        problem_list = file.get_file_list(input_dir,
                                          constants.PROBLEM_FILE_REGEX)

        domain_list = file.get_file_list(input_dir,
                                         constants.DOMAIN_FILE_REGEX)

        assert len(domain_list) == 1
        domain_filepath = domain_list[0]

        assert len(problem_list) == 1
        for problem_filepath in problem_list:

            self.qlearn(domain_filepath, problem_filepath)
