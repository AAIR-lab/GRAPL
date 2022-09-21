'''
Created on Oct 21, 2020

@author: anonymous
'''

from collections import deque
import copy
import random

import tqdm

from concretized.problem import Problem
from generalized_learning import simulator
from generalized_learning.abstraction.canonical import CanonicalAbstractionMLP
from generalized_learning.abstraction.concrete import ConcreteMLP
from generalized_learning.abstraction.description_logic import DescriptionLogicMLP
from generalized_learning.qlearning.evaluator.dqn_evaluator import DQNEvaluator
from generalized_learning.qlearning.ran import RAN
from generalized_learning.util import constants
from neural_net.nn import NN
import numpy as np
from util import file
from util import state_explorer


abstraction_func = None


class ConcreteDQN:

    DEFAULTS = {
        "episodes_per_epoch": 50,
        "training_interval": 1,
        "total_train_iterations": 1,
        "max_timesteps": 7500,
        "num_episodes": 1500,
        "timesteps_per_episode": 50,
        "epsilon": 0.1,
        "min_epsilon": 0.1,
        "epsilon_decay_rate": 0.99,
        "replay_memory_size": 2500,
        "epochs": 1,
        "batch_size": 256,
        "gamma": 1.0,
        "experiment_name": None,
        "nn_type": "torch_generic",
        "nn_name": "vanilla_dqn",
        "simulator_type": "generic",
        "debug_print": False,
        "ignore_stats_till": 0,
        "start_episode": 0,
    }

    def _get_value(self, config_dict, key):

        try:

            return config_dict[key]
        except KeyError:

            return ConcreteDQN.DEFAULTS[key]

    def __init__(self, parent_dir, phase_dict):

        self._parent_dir = parent_dir
        self._phase_dict = phase_dict

        self.reset()

    def reset(self):

        assert "experiment_name" in self._phase_dict
        self._experiment_name = self._get_value(
            self._phase_dict,
            "experiment_name")

        self._num_episodes = self._get_value(
            self._phase_dict,
            "num_episodes")

        self._episodes_per_epoch = self._get_value(
            self._phase_dict,
            "episodes_per_epoch")
        assert self._num_episodes % self._episodes_per_epoch == 0

        self._timesteps_per_episode = self._get_value(
            self._phase_dict,
            "timesteps_per_episode")

        self._max_timesteps = self._get_value(
            self._phase_dict,
            "max_timesteps")

        self._training_interval = self._get_value(
            self._phase_dict,
            "training_interval")

        self._total_train_iterations = self._get_value(
            self._phase_dict,
            "total_train_iterations")

        self._epsilon = self._get_value(
            self._phase_dict,
            "epsilon")

        self._min_epsilon = self._get_value(
            self._phase_dict,
            "min_epsilon")

        self._epsilon_decay_rate = self._get_value(
            self._phase_dict,
            "epsilon_decay_rate")

        self._epochs = self._get_value(
            self._phase_dict,
            "epochs")

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

        self._nn_type = self._get_value(
            self._phase_dict,
            "nn_type")

        self._nn_name = self._get_value(
            self._phase_dict,
            "nn_name")

        self._simulator_type = self._get_value(
            self._phase_dict,
            "simulator_type")

        self._debug_print = self._get_value(
            self._phase_dict,
            "debug_print")

        self._ignore_stats_till = self._get_value(
            self._phase_dict,
            "ignore_stats_till")

        self._start_episode = self._get_value(
            self._phase_dict,
            "start_episode")

    def get_action(self, problem, current_state, episode, episode_time_step,
                   time_step):

        applicable_actions = problem.get_applicable_actions(current_state)
        assert len(applicable_actions) > 0

        # Shuffle the actions.
        random.shuffle(applicable_actions)

        random_action = random.sample(applicable_actions, 1)[0]

        q_values = np.zeros(len(applicable_actions))
        DQNEvaluator.update_q_values(applicable_actions,
                                     current_state,
                                     self.concrete_nn,
                                     problem,
                                     q_values,
                                     1)

        if self._debug_print:

            print("==== get_action(): Episode: %3u (step: %u), Timestep: %3u, Epsilon: %1.2f ===="
                  % (episode, episode_time_step, time_step, self._epsilon))

            for i in range(len(applicable_actions)):
                print(
                    "[%3u]: Action: %30s, Concrete: %4.2f" % (
                        i,
                        applicable_actions[i],
                        q_values[i]))

        if random.random() <= self._epsilon:

            if self._debug_print:

                print("Selected: [XX]: %s" % (random_action))

            return random_action, True
        else:

            if self._debug_print:

                print("Selected: [%u]: %s" % (
                    np.argmax(q_values),
                    applicable_actions[np.argmax(q_values)]))

            return applicable_actions[np.argmax(q_values)], False

    def get_single_nn_targets(self, problem, nn, mini_batch):

        targets = []

        for i in range(len(mini_batch)):

            _, _, next_state, _, done, _ = mini_batch[i]

            if done:

                targets.append(0)
            else:

                applicable_actions = problem.get_applicable_actions(next_state)
                q_values = np.zeros(len(applicable_actions))

                DQNEvaluator.update_q_values(applicable_actions,
                                             next_state,
                                             nn,
                                             problem,
                                             q_values,
                                             1.0)

                targets.append(np.amax(q_values))

        return targets

    def get_nn_targets(self, problem, mini_batch):

        targets = []

        for i in range(len(mini_batch)):

            _, _, next_state, _, done, _ = mini_batch[i]

            if done:

                targets.append(0)
            else:
                applicable_actions = problem.get_applicable_actions(next_state)

                q_values = np.zeros(len(applicable_actions))
                DQNEvaluator.update_q_values(applicable_actions,
                                             next_state,
                                             self.concrete_nn,
                                             problem,
                                             q_values,
                                             1.0)

                targets.append(np.amax(q_values))

        return targets

    def train_network_on_minibatch(self, problem, mini_batch, nn,
                                   use_other=False):

        abstraction = nn._abstract_domain

        if use_other:

            q_values = self.get_nn_targets(problem, mini_batch)
        else:
            q_values = self.get_single_nn_targets(problem, nn, mini_batch)

        nn_pkgs_list = []
        for i in range(len(mini_batch)):

            current_state, action, _, reward, done, _ = mini_batch[i]

            abstract_state = abstraction.create_abstract_state(
                problem,
                current_state)

            target = reward + self._gamma * q_values[i] * (1 - done)

            # Always set the target to below 0.
            # Since we use argmax and amax, once targets go above 0, the
            # network will never recover.
            if not done:

                target = min(-1, target)

            nn_pkg = abstraction.encode_nn_training_data(
                nn.nn_input_call_set,
                nn.nn_output_call_set,
                abstract_state=abstract_state,
                action=action,
                q_s_a=target)

            nn_pkgs_list.append(nn_pkg)

        nn.train(nn_pkgs_list, epochs=self._epochs,
                 batch_size=self._batch_size,
                 shuffle=True)

    def replay(self, problem):

        if len(self._replay_memory) < self._batch_size:

            return

        mini_batch = random.sample(self._replay_memory, self._batch_size)

        # Train the DQN.
        self.train_network_on_minibatch(problem, mini_batch,
                                        self.concrete_nn)

    def qlearn_with_episode_limit(self, problem, evaluator,
                                  cost_results, episode_results):

        if problem.is_goal_satisfied(problem.get_initial_state()):

            return

        episode_progress = tqdm.tqdm(disable=self._debug_print,
                                     total=self._num_episodes,
                                     unit=" episodes",
                                     leave=False,
                                     position=0)

        sim = simulator.get_simulator(self._simulator_type)

        total_reward = 0
        episodes_successful = 0
        global_time_step = 0
        for episode in range(self._start_episode,
                             self._num_episodes + self._start_episode):

            ignore_stats = episode < self._ignore_stats_till

            timestep_progress = tqdm.tqdm(disable=self._debug_print,
                                          total=self._timesteps_per_episode,
                                          unit=" timesteps",
                                          leave=False,
                                          position=1)

            current_state = problem.get_initial_state()
            done = False
            episode_reward = 0
            time_step = 0
            while not done and time_step < self._timesteps_per_episode:

                action, is_random = self.get_action(
                    problem,
                    current_state,
                    episode,
                    time_step,
                    global_time_step + time_step)
                next_state, reward, done = sim.apply_action(problem,
                                                            current_state,
                                                            action)
                episode_reward += reward

                self._replay_memory.append((current_state, action, next_state,
                                            reward, done, is_random))

                time_step += 1
                timestep_progress.update(1)

                # HER
                if time_step == self._timesteps_per_episode:
                    self._replay_memory.append((current_state, action, next_state,
                                                reward, True, False))

                current_state = next_state

                if (global_time_step + time_step) % self._training_interval == 0:
                    for _ in range(self._total_train_iterations):

                        self.replay(problem)

            timestep_progress.close()
            episode_progress.update(1)

            # Update the global time step.
            global_time_step += time_step

            # Decay epsilon.
            self._epsilon *= self._epsilon_decay_rate
            self._epsilon = max(self._min_epsilon, self._epsilon)

            if done:

                episodes_successful += 1

            total_reward += episode_reward

            stat_epi_no = episode - self._ignore_stats_till

            # Add the episode cost to the data tracker.
            cost_results.add_data(problem, self._experiment_name,
                                  (stat_epi_no, episode_reward))

            if (stat_epi_no + 1) % self._episodes_per_epoch == 0:

                epoch_no = int(stat_epi_no / self._episodes_per_epoch)
                success_rate = (episodes_successful * 100.0) \
                    / self._episodes_per_epoch
                avg_cost = (total_reward * 1.0) / self._episodes_per_epoch

                episode_results.add_data(
                    problem, self._experiment_name,
                    (epoch_no, success_rate, avg_cost))

                total_reward = 0
                episodes_successful = 0

        episode_progress.close()

    def run(self, input_dir, cost_results, episode_results):

        problem_list = sorted(file.get_file_list(input_dir,
                                                 constants.PROBLEM_FILE_REGEX))

        domain_list = file.get_file_list(input_dir,
                                         constants.DOMAIN_FILE_REGEX)

        assert len(domain_list) == 1
        domain_filepath = domain_list[0]

        evaluator = DQNEvaluator(self._phase_dict)

        for problem_filepath in problem_list:

            # Reset epsilon to 1 for each problem.
            self._epsilon = self._get_value(
                self._phase_dict,
                "epsilon")

            problem = Problem(domain_filepath.name, problem_filepath.name,
                              problem_filepath.parent)

            self.concrete_nn = RAN.get_instance("concrete",
                                                domain_filepath,
                                                problem_list,
                                                self._nn_type,
                                                "concrete.%s" % (self._nn_name))
            self.concrete_nn = self.concrete_nn.q_nn

            self.qlearn_with_episode_limit(problem, evaluator,
                                           cost_results, episode_results)

            self._replay_memory.clear()

            import tensorflow as tf
            tf.keras.backend.clear_session()
