'''
Created on Oct 21, 2020

@author: anonymous
'''

from collections import deque
import copy
import random

from tensorflow.python.ops import state_grad
import tqdm

from abstraction.domain import AbstractDomain
from abstraction.state import AbstractState
from concretized.problem import Problem
from concretized.solution import Solution
from neural_net.nn import NN
import numpy as np
from search.ff import FF
from util import constants
from util import file
from util import reproducibility
from util import state_explorer


class NNTD:

    def __init__(self, num_episodes=500,
                 timesteps_per_episode=75,
                 epsilon=1.0,
                 min_epsilon=0.01,
                 epsilon_decay_rate=0.9,
                 learning_rate=0.3,
                 gamma=1.0,
                 lambda_value=0):

        self._num_episodes = num_episodes
        self._timesteps_per_episode = timesteps_per_episode
        self._epsilon = epsilon
        self._min_epsilon = min_epsilon
        self._epsilon_decay_rate = epsilon_decay_rate
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._lambda = lambda_value
        self._debug_print = True

        self._q_table = {}

    def prepare_abstract_domain(self, problem):

        action_names = problem.get_actions_in_domain()
        max_action_params = problem.get_max_action_params_in_domain()
        abstract_domain = AbstractDomain(action_names=action_names,
                                         max_action_params=max_action_params)

        random_action = problem.get_applicable_actions(
            problem.get_initial_state())[0]

        states = state_explorer.explore_state_space(problem)

        for state in states:

            abstract_state = AbstractState(problem, state)
            abstract_domain.encode_nn_input(abstract_state, random_action)

        abstract_domain.initialize_nn_parameters()
        return abstract_domain

    def print_q_table_stats(self, problem, state, msg, abstract_domain=None,
                            nn=None):

        if not self._debug_print:

            return

        q_action_dict = self._q_table.get(state, {})

        q_values = []
        string = ""
        applicable_actions = problem.get_applicable_actions(state)
        for action in applicable_actions:

            string += "%s " % (action)
            q_values.append("%.2f" % (q_action_dict.get(action, 0)))

        print("*****", msg, "A:", string, "Q:", q_values)

        if abstract_domain is not None:

            abstract_state = AbstractState(problem, state)
            q_values = self.get_q_values(abstract_domain, nn, abstract_state,
                                         applicable_actions)
            print("#####", msg, "A:", string, "Q:", q_values)

    def get_action(self, problem, abstract_domain, nn, current_state):

        applicable_actions = problem.get_applicable_actions(current_state)
        assert len(applicable_actions) > 0
        random_action = random.sample(applicable_actions, 1)[0]

        string = ""
        for action in applicable_actions:

            string += "%s " % (action)

        self.print_q_table_stats(problem, current_state, "get", abstract_domain,
                                 nn)

        if random.random() <= self._epsilon:

            if self._debug_print:
                print("[RANDOM]", random_action, "A:", string)
            return random_action
        else:

            q_value, q_action = self.get_best_q_value(problem, current_state,
                                                      abstract_domain, nn)

            if self._debug_print:
                print("[     Q]", str(q_action), q_value, "A:", string)
            return q_action

    def get_best_q_value(self, problem, state, abstract_domain=None, nn=None):

        if problem.is_goal_satisfied(state):

            return 0, None
        elif abstract_domain is not None:

            return self.get_best_q_value_nn(problem, state, abstract_domain, nn)
        else:
            best_q_value = float("inf")
            best_action = None

            q_action_dict = self._q_table.setdefault(state, {})
            for action in problem.get_applicable_actions(state):

                q_action_value = q_action_dict.setdefault(action, 0)
                if q_action_value < best_q_value:

                    best_q_value = q_action_value
                    best_action = action

            return best_q_value, best_action

    def get_q_values(self, abstract_domain, nn, abstract_state,
                     applicable_actions):

        nn_pkg_list = [abstract_domain.encode_nn_input(
            abstract_state,
            applicable_actions[0])]
        for i in range(1, len(applicable_actions)):

            nn_input_pkg = abstract_domain.encode_nn_input(abstract_state,
                                                           applicable_actions[i])

#             nn_input_pkg = copy.deepcopy(nn_pkg_list[0])
#             abstract_domain.override_nn_input_action(nn_input_pkg,
#                                                      abstract_state,
#                                                      applicable_actions[i])
            nn_pkg_list.append(nn_input_pkg)

        nn_output_pkgs = nn.fit_pkgs(nn_pkg_list)
        q_values = []
        for nn_output_pkg in nn_output_pkgs:

            q_values.append(nn_output_pkg.decode("plan_length"))

        return q_values

    def get_best_q_value_nn(self, problem, state, abstract_domain, nn):

        if problem.is_goal_satisfied(state):

            if self._debug_print:

                import time
                print("XXXXXXXXX GOAL XXXXXXXXXXX")
                print("")
                print("")
                print("")
                time.sleep(5)
            return 0, None
        else:

            abstract_state = AbstractState(problem, state)
            applicable_actions = problem.get_applicable_actions(state)
            random.shuffle(applicable_actions)
            q_values = self.get_q_values(abstract_domain, nn, abstract_state,
                                         applicable_actions)

            return np.amin(q_values), applicable_actions[np.argmin(q_values)]

    def apply_action(self, current_state, action):

        return action.apply(current_state), action.get_cost()

    def td_update(self, problem, current_state, action, next_state, reward):

        q_action_dict = self._q_table.setdefault(current_state, {})
        q_action_value = q_action_dict.setdefault(action, 0)

        q_dash, _ = self.get_best_q_value(problem, next_state)

        q_action_dict[action] = q_action_value + self._learning_rate \
            * (reward + self._gamma * q_dash - q_action_value)

    def nn_update(self, problem, abstract_domain, nn, current_state, action,
                  next_state, reward):

        if self._debug_print:
            print("========= NN UPDATE ==========")
        self.print_q_table_stats(problem, current_state, "before_nn", abstract_domain,
                                 nn)
        abstract_state = AbstractState(problem, current_state)

        self.print_q_table_stats(problem, next_state, "  next_nn", abstract_domain,
                                 nn)

        q_dash, _ = self.get_best_q_value_nn(problem, next_state,
                                             abstract_domain, nn)

        if problem.is_goal_satisfied(next_state):
            epochs = 10
        else:
            epochs = 10

        target = reward + self._gamma * q_dash
        nn_pkg = abstract_domain.encode_nn_training_data(abstract_state,
                                                         action,
                                                         target)

        nn_pkgs = [nn_pkg]

        applicable_actions = problem.get_applicable_actions(current_state)
        q_values = self.get_q_values(abstract_domain, nn, abstract_state,
                                     applicable_actions)

        string = ""
        target_q = []
        for i in range(len(applicable_actions)):

            string += "%s " % (str(applicable_actions[i]))

            if applicable_actions[i] is not action:

                nn_new_pkg = abstract_domain.encode_nn_training_data(abstract_state,
                                                                     applicable_actions[i],
                                                                     q_values[i])

                target_q.append(q_values[i])

                nn_pkgs.append(nn_new_pkg)
            else:

                target_q.append(target)

        if self._debug_print:
            print("***** training: A: %s Q %s" % (string, target_q))

        nn.train(nn_pkgs, epochs=epochs, batch_size=25, shuffle=True)

        self.print_q_table_stats(problem, current_state, "after_nn", abstract_domain,
                                 nn)

        if self._debug_print:
            if q_dash == 0:

                import time
                time.sleep(5)

            print("")
            print("")

    def rollout(self, problem, abstract_domain, nn, current_state, progress_bar):

        import tempfile
        import pathlib

        domain_filepath = pathlib.Path(problem.get_domain_filepath())
        file_handle = tempfile.NamedTemporaryFile("w+", delete=False,
                                                  dir=domain_filepath.parent)

        problem.write_problem(file_handle, current_state)
        file_handle.close()

        ff = FF("ff", {})
        ff.search(domain_filepath,
                  pathlib.Path(file_handle.name))

        solution_filepath = "%s.ff.%s" % (
            file_handle.name, constants.SOLUTION_FILE_EXT)
        solution = Solution.parse(solution_filepath)

        action_list = solution.get_action_list()

        for action_name in action_list:

            action = problem.get_action(action_name)
            progress_bar.update(1)

            next_state, reward = self.apply_action(current_state, action)

            self.td_update(problem, current_state, action, next_state, reward)
            self.nn_update(problem, abstract_domain, nn, current_state,
                           action, next_state, reward)
            current_state = next_state

        assert problem.is_goal_satisfied(current_state)

        import os
        os.remove(file_handle.name)
        os.remove(solution_filepath)

        pass

    def simulate(self, problem, abstract_domain, nn):

        current_state = problem.get_initial_state()

        time_step = 0
        progress_bar = tqdm.tqdm(unit=" steps")
        while not problem.is_goal_satisfied(current_state) \
                and time_step < self._timesteps_per_episode:

            time_step += 1
            progress_bar.update(1)

            action = self.get_action(
                problem, abstract_domain, nn, current_state)
            next_state, reward = self.apply_action(current_state, action)

            self.td_update(problem, current_state, action, next_state, reward)
            self.nn_update(problem, abstract_domain, nn, current_state,
                           action, next_state, reward)
            current_state = next_state

        progress_bar.close()

        if not problem.is_goal_satisfied(current_state):

            self.rollout(problem, abstract_domain, nn,
                         current_state, progress_bar)

#             self._steps += 1
#             if self._steps % 75 == 0:
#
#                 self._epsilon *= self._epsilon_decay_rate
#                 self._epsilon = max(self._min_epsilon, self._epsilon)
#                 print("========= epsilon", self._epsilon)
        return time_step

    def qlearn(self, domain_filepath, problem_filepath):

        problem = Problem(domain_filepath.name, problem_filepath.name,
                          problem_filepath.parent)

        abstract_domain = self.prepare_abstract_domain(problem)
        nn = NN.get_instance(abstract_domain, "generic", "gms")
        self._steps = 0

        for episode in range(self._num_episodes):

            time_steps = self.simulate(problem, abstract_domain, nn)

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
