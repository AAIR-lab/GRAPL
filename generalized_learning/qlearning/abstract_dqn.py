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


class AbstractDQN:

    DEFAULTS = {
        "episodes_per_epoch": 50,
        "training_interval": 32,
        "total_train_iterations": 1,
        "max_timesteps": 7500,
        "num_episodes": 50,
        "timesteps_per_episode": 50,
        "epsilon": 0.1,
        "min_epsilon": 0.1,
        "epsilon_decay_rate": 0.99,
        "abstraction_weight": 1.0,
        "min_abstraction_weight": 0.1,
        "abstraction_weight_decay_rate": 0.9,
        "replay_memory_size": 5000,
        "epochs": 1,
        "batch_size": 256,
        "gamma": 1.0,
        "model_dir": None,
        "abstraction": "description_logic_mlp",
        "feature_file": None,
        "model_dir": None,
        "experiment_name": None,
        "nn_type": "torch_generic",
        "nn_name": "vanilla",
        "simulator_type": "generic",
        "pretrain_steps": 200 * 250,
        "pretrain_epochs": 25,
        "pretrain_batch_size": 512,
        "debug_print": False,
        "start_episode": 0,

    }

    def _get_value(self, config_dict, key):

        try:

            return config_dict[key]
        except KeyError:

            return AbstractDQN.DEFAULTS[key]

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

        self._abstraction_weight = self._get_value(
            self._phase_dict,
            "abstraction_weight")

        self._min_abstraction_weight = self._get_value(
            self._phase_dict,
            "min_abstraction_weight")

        self._abstraction_weight_decay_rate = self._get_value(
            self._phase_dict,
            "abstraction_weight_decay_rate")

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

        self._abstraction = self._get_value(
            self._phase_dict,
            "abstraction")

        self._feature_file = self._get_value(
            self._phase_dict,
            "feature_file")

        self._old_ran_dir = self._get_value(
            self._phase_dict,
            "model_dir")

        if self._old_ran_dir is not None:
            self._old_ran_dir = file.get_relative_path(
                self._old_ran_dir,
                self._parent_dir)

        self._nn_type = self._get_value(
            self._phase_dict,
            "nn_type")

        self._nn_name = self._get_value(
            self._phase_dict,
            "nn_name")

        self._simulator_type = self._get_value(
            self._phase_dict,
            "simulator_type")

        self._pretrain_steps = self._get_value(
            self._phase_dict,
            "pretrain_steps")

        self._pretrain_epochs = self._get_value(
            self._phase_dict,
            "pretrain_epochs")

        self._pretrain_batch_size = self._get_value(
            self._phase_dict,
            "pretrain_batch_size")

        self._debug_print = self._get_value(
            self._phase_dict,
            "debug_print")

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

            old_q_nn = self.old_ran.q_nn
            new_q_nn = self.new_ran.q_nn

            concrete_q_values = np.zeros(len(applicable_actions))
            DQNEvaluator.update_q_values(applicable_actions,
                                         current_state,
                                         self.concrete_nn,
                                         problem,
                                         concrete_q_values,
                                         1)

            old_q_values = np.zeros(len(applicable_actions))

            if old_q_nn is not None:
                DQNEvaluator.update_q_values(applicable_actions,
                                             current_state,
                                             old_q_nn,
                                             problem,
                                             old_q_values,
                                             1.0)

            new_q_values = np.zeros(len(applicable_actions))
            DQNEvaluator.update_q_values(applicable_actions,
                                         current_state,
                                         new_q_nn,
                                         problem,
                                         new_q_values,
                                         1.0)

            for i in range(len(applicable_actions)):
                print(
                    "[%3u]: Action: %30s, Abs. weight: %.2f, Total: %4.2f, Concrete: %4.2f, Old: %4.2f, New: %4.2f" % (
                        i,
                        applicable_actions[i],
                        self._abstraction_weight,
                        q_values[i],
                        concrete_q_values[i],
                        old_q_values[i],
                        new_q_values[i]
                    ))

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

            _, _, next_state, _, done, is_random = mini_batch[i]

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

    def train_using_ran(self, problem, mini_batch, nn):

        abstraction = nn._abstract_domain

        nn_pkgs_list = []
        for i in range(len(mini_batch)):

            current_state, _, _, _, _, _ = mini_batch[i]

            applicable_actions = problem.get_applicable_actions(current_state)
            q_values = np.zeros(len(applicable_actions))
            DQNEvaluator.update_q_values(applicable_actions,
                                         current_state,
                                         self.old_ran.q_nn,
                                         problem,
                                         q_values,
                                         1.0)

            abstract_state = abstraction.create_abstract_state(
                problem,
                current_state)

            for i in range(len(applicable_actions)):

                nn_pkg = abstraction.encode_nn_training_data(
                    nn.nn_input_call_set,
                    nn.nn_output_call_set,
                    abstract_state=abstract_state,
                    action=applicable_actions[i],
                    q_s_a=min(-1, q_values[i]))

                nn_pkgs_list.append(nn_pkg)

        nn.train(nn_pkgs_list, epochs=self._epochs,
                 batch_size=self._batch_size,
                 shuffle=True)

    def train_network_on_minibatch(self, problem, mini_batch, nn):

        abstraction = nn._abstract_domain

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

    def train_ran(self, problem, mini_batch):

        nn_pkgs_list = []
        for i in range(len(mini_batch)):

            current_state, applied_action, _, _, _, _ = mini_batch[i]

#             applicable_actions = problem.get_applicable_actions(current_state)
            applicable_actions = [applied_action]

            q_values = np.zeros(len(applicable_actions))

            DQNEvaluator.update_q_values(applicable_actions,
                                         current_state,
                                         self.concrete_nn,
                                         problem,
                                         q_values,
                                         1.0)

            abstract_state = self.new_ran.q_nn._abstract_domain.create_abstract_state(
                problem,
                current_state)

            for i in range(len(applicable_actions)):
                nn_pkg = self.new_ran.q_nn._abstract_domain.encode_nn_training_data(
                    self.new_ran.q_nn.nn_input_call_set,
                    self.new_ran.q_nn.nn_output_call_set,
                    abstract_state=abstract_state,
                    action=applicable_actions[i],
                    q_s_a=min(-1, q_values[i]))
                nn_pkgs_list.append(nn_pkg)

        self.new_ran.q_nn.train(nn_pkgs_list, epochs=1,
                                batch_size=self._batch_size,
                                shuffle=True)

    def replay(self, problem):

        if len(self._replay_memory) < self._batch_size:

            return

        mini_batch = random.sample(self._replay_memory, self._batch_size)

        # Train the DQN.
        if False and self.old_ran.q_nn is not None and random.random() < self._abstraction_weight:
            self.train_using_ran(problem, mini_batch, self.concrete_nn)
        else:
            self.train_network_on_minibatch(problem, mini_batch,
                                            self.concrete_nn)

        # Also train the new ran.
        self.train_ran(problem, mini_batch)

    def initialize_concrete_nn(self, problem, sim):

        if self.old_ran.q_nn is None:

            return

        current_state = problem.get_initial_state()

        nn_pkgs_list = []

        progress_bar = tqdm.tqdm(total=self._pretrain_steps, unit=" states",
                                 leave=False)

        for _ in range(self._pretrain_steps):

            progress_bar.update(1)

            applicable_actions = problem.get_applicable_actions(current_state)
            action = random.choice(applicable_actions)

            next_state, _, done = sim.apply_action(problem,
                                                   current_state,
                                                   action)

            if done:
                q_values = [-1]
            else:
                q_values = np.zeros(len([action]))

            DQNEvaluator.update_q_values([action],
                                         current_state,
                                         self.old_ran.q_nn,
                                         problem,
                                         q_values,
                                         1.0)

            nn_pkg = self.concrete_nn._abstract_domain.encode_nn_training_data(
                self.concrete_nn.nn_input_call_set,
                self.concrete_nn.nn_output_call_set,
                abstract_state=self.concrete_nn._abstract_domain.create_abstract_state(
                    problem,
                    current_state),
                action=action,
                q_s_a=min(-1, q_values[0]))
            nn_pkgs_list.append(nn_pkg)

            if done:

                current_state = problem.get_initial_state()
            else:

                current_state = next_state

        progress_bar.close()

        self.concrete_nn.train(nn_pkgs_list,
                               epochs=self._pretrain_epochs,
                               batch_size=self._pretrain_batch_size,
                               shuffle=True)

    def qlearn_with_episode_limit(self, problem, evaluator,
                                  cost_results, episode_results):

        if problem.is_goal_satisfied(problem.get_initial_state()):

            return

        sim = simulator.get_simulator(self._simulator_type)

        self.initialize_concrete_nn(problem, sim)

        episode_progress = tqdm.tqdm(disable=self._debug_print,
                                     total=self._num_episodes,
                                     unit=" episodes",
                                     leave=False,
                                     position=0)

        total_reward = 0
        episodes_successful = 0
        global_time_step = 0
        for episode in range(self._start_episode,
                             self._num_episodes + self._start_episode):

            timestep_progress = tqdm.tqdm(disable=self._debug_print,
                                          total=self._timesteps_per_episode,
                                          unit=" timesteps",
                                          leave=False,
                                          position=1)

            current_state = problem.get_initial_state()
            done = False
            time_step = 0
            episode_reward = 0
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

            if done:

                episodes_successful += 1

            total_reward += episode_reward

            # Add the episode cost to the data tracker.
            cost_results.add_data(problem, self._experiment_name,
                                  (episode, episode_reward))

            if (episode + 1) % self._episodes_per_epoch == 0:

                epoch_no = int(episode / self._episodes_per_epoch)
                success_rate = (episodes_successful * 100.0) \
                    / self._episodes_per_epoch
                avg_cost = (total_reward * 1.0) / self._episodes_per_epoch

                episode_results.add_data(
                    problem, self._experiment_name,
                    (epoch_no, success_rate, avg_cost))

                total_reward = 0
                episodes_successful = 0

            # Update the global time step.
            global_time_step += time_step

            # Decay epsilon.
            self._epsilon *= self._epsilon_decay_rate
            self._epsilon = max(self._min_epsilon, self._epsilon)

            self._abstraction_weight *= self._abstraction_weight_decay_rate
            if self._abstraction_weight < self._min_abstraction_weight:
                self._abstraction_weight = 0

        episode_progress.close()

    def run(self, input_dir, cost_results, episode_results):

        problem_list = sorted(file.get_file_list(input_dir,
                                                 constants.PROBLEM_FILE_REGEX))

        domain_list = file.get_file_list(input_dir,
                                         constants.DOMAIN_FILE_REGEX)

        assert len(domain_list) == 1
        domain_filepath = domain_list[0]

        evaluator = DQNEvaluator(self._phase_dict)

        if self._old_ran_dir is not None:

            # All leapfrog RAN's must have the same name but reside in a
            # different directory currently.
            self.old_ran = RAN.load(self._old_ran_dir, self._nn_name)
            self.new_ran = RAN.load(self._old_ran_dir, self._nn_name)
        else:

            self.old_ran = RAN()

            self.new_ran = RAN.get_instance(self._abstraction,
                                            domain_filepath,
                                            problem_list,
                                            self._nn_type,
                                            self._nn_name,
                                            feature_file=self._feature_file)

        for problem_filepath in problem_list:

            # Reset epsilon to 1 for each problem.
            self._epsilon = self._get_value(
                self._phase_dict,
                "epsilon")

            self._abstraction_weight = self._get_value(
                self._phase_dict,
                "abstraction_weight")

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
            self.new_ran.soft_save(input_dir)

            import tensorflow as tf
            tf.keras.backend.clear_session()

            # Reload the old ran as the new network.
            self.old_ran = RAN.load(input_dir, self._nn_name)
