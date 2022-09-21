from collections import deque
import random
import statistics

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


class AbstractQL:

    DEFAULTS = {
        "episodes_per_epoch": 50,
        "training_interval": 256,
        "total_train_iterations": 1,
        "max_timesteps": 7500,
        "num_episodes": 1500,
        "timesteps_per_episode": 250,
        "epsilon": 0.1,
        "min_epsilon": 0.01,
        "epsilon_decay_rate": 1.0,
        "replay_memory_size": 5000,
        "epochs": 25,
        "batch_size": 256,
        "gamma": 1.0,
        "experiment_name": "abstract_ql_0",
        "lambda": 0,
        "learning_rate": 0.8,
        "model_dir": None,
        "abstraction": "description_logic_mlp",
        "feature_file": None,
        "experiment_name": None,
        "nn_type": "torch_generic",
        "nn_name": "vanilla",
        "simulator_type": "generic",
        "debug_print": False,
        "start_episode": 0,
    }

    def _get_value(self, config_dict, key):

        try:

            return config_dict[key]
        except KeyError:

            return AbstractQL.DEFAULTS[key]

    def __init__(self, parent_dir, phase_dict):

        self._parent_dir = parent_dir
        self._phase_dict = phase_dict

        self.reset()

    def reset(self):

        assert "experiment_name" in self._phase_dict
        self._experiment_name = self._get_value(
            self._phase_dict,
            "experiment_name")

        self._max_timesteps = self._get_value(
            self._phase_dict,
            "max_timesteps")

        self._training_interval = self._get_value(
            self._phase_dict,
            "training_interval")

        self._total_train_iterations = self._get_value(
            self._phase_dict,
            "total_train_iterations")

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

        self._epsilon = self._get_value(
            self._phase_dict,
            "epsilon")

        self._min_epsilon = self._get_value(
            self._phase_dict,
            "min_epsilon")

        self._epsilon_decay_rate = self._get_value(
            self._phase_dict,
            "epsilon_decay_rate")

        self._gamma = self._get_value(
            self._phase_dict,
            "gamma")

        self._lambda = self._get_value(
            self._phase_dict,
            "lambda")

        self._learning_rate = self._get_value(
            self._phase_dict,
            "learning_rate")

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

        self._debug_print = self._get_value(
            self._phase_dict,
            "debug_print")

        self._start_episode = self._get_value(
            self._phase_dict,
            "start_episode")

    def prepare_abstraction(self, domain_filepath, problem_list):

        if self._abstraction == "description_logic_mlp":

            abstraction = DescriptionLogicMLP.create(
                domain_filepath,
                problem_list,
                feature_file=self._feature_file)
        elif self._abstraction == "canonical_abstraction_mlp":

            abstraction = CanonicalAbstractionMLP.create(domain_filepath,
                                                         problem_list)
        else:

            raise Exception("Unknown abstraction function")

        abstraction.initialize_nn()
        return abstraction

    def get_q_table_key(self, state, action):

        return (state, action)

    def initialize_q_table(self, problem, state, ran):

        applicable_actions = problem.get_applicable_actions(state)

        q_values = np.zeros(len(applicable_actions))

        if ran.q_nn is not None:
            DQNEvaluator.update_q_values(applicable_actions,
                                         state,
                                         ran.q_nn,
                                         problem,
                                         q_values,
                                         1.0)

        for i in range(len(applicable_actions)):

            action = applicable_actions[i]
            dict_key = self.get_q_table_key(state, action)

            self.q_table[dict_key] = q_values[i]

    def get_q_table_value(self, problem, state, action, ran):

        dict_key = self.get_q_table_key(state, action)

        try:

            return self.q_table[dict_key]
        except KeyError:

            self.initialize_q_table(problem, state, ran)
            return self.q_table[dict_key]

    def get_eligibility_table_value(self, _problem, state, action):

        dict_key = self.get_q_table_key(state, action)

        return self.e_table.setdefault(dict_key, 0)

    def set_eligibility_table_value(self, _problem, state, action, value):

        dict_key = self.get_q_table_key(state, action)
        self.e_table[dict_key] = value

    def get_eligibility_condition(self, problem, next_state,
                                  selected_action, ran):

        applicable_actions = problem.get_applicable_actions(next_state)
        max_value = self.get_q_table_value(problem, next_state,
                                           selected_action, ran)

        for action in applicable_actions:

            value = self.get_q_table_value(problem, next_state, action, ran)

            if value > max_value:

                return False

        return True

    def get_q_values(self, problem, state, applicable_actions, ran):

        q_values = []
        for action in applicable_actions:

            q_values.append(self.get_q_table_value(
                problem, state, action, ran))

        return q_values

    def get_best_q_value(self, problem, state, ran):

        applicable_actions = problem.get_applicable_actions(state)
        random.shuffle(applicable_actions)

        q_values = self.get_q_values(problem, state, applicable_actions, ran)
        return np.amax(q_values)

    def set_q_value(self, problem, state, action, value):

        dict_key = self.get_q_table_key(state, action)
        self.q_table[dict_key] = value

    def td_update(self, problem, current_state, action, next_state, reward,
                  done, ran, eligibility, episode_s_a):

        q_value = self.get_q_table_value(problem, current_state, action,
                                         ran)

        if done:

            q_dash = 0
        else:

            q_dash = self.get_best_q_value(problem, next_state, ran)

        td_error = (reward + self._gamma * q_dash - q_value)

        e_s_a = self.get_eligibility_table_value(problem,
                                                 current_state, action)
        e_s_a += 1

        self.set_eligibility_table_value(problem, current_state,
                                         action, e_s_a)

        for state, action in episode_s_a:

            dict_key = self.get_q_table_key(state, action)

            q_value = self.q_table[dict_key]
            e_s_a = self.e_table.setdefault(dict_key, 0)
            q_value += self._learning_rate * td_error * e_s_a

            q_value = min(-1, q_value)
            self.q_table[dict_key] = q_value

            if not eligibility:

                self.e_table[dict_key] = 0
            else:

                e_s_a = self.e_table.setdefault(dict_key, 0)
                e_s_a *= self._gamma * self._lambda

                self.e_table[dict_key] = e_s_a

    def get_action(self, problem, current_state, old_ran, new_ran, episode, episode_time_step,
                   time_step):

        applicable_actions = problem.get_applicable_actions(current_state)
        assert len(applicable_actions) > 0

        # Shuffle the actions.
        random.shuffle(applicable_actions)

        random_action = random.sample(applicable_actions, 1)[0]

        q_values = self.get_q_values(
            problem, current_state, applicable_actions, old_ran)

        if self._debug_print:

            print("==== get_action(): Episode: %3u (step: %u), Timestep: %3u, Epsilon: %1.2f ===="
                  % (episode, episode_time_step, time_step, self._epsilon))

            old_q_values = np.zeros(len(applicable_actions))

            if old_ran.q_nn is not None:
                DQNEvaluator.update_q_values(applicable_actions,
                                             current_state,
                                             old_ran.q_nn,
                                             problem,
                                             old_q_values,
                                             1.0)

            new_q_values = np.zeros(len(applicable_actions))
            DQNEvaluator.update_q_values(applicable_actions,
                                         current_state,
                                         new_ran.q_nn,
                                         problem,
                                         new_q_values,
                                         1.0)

            for i in range(len(applicable_actions)):
                print(
                    "[%3u]: Action: %30s, Concrete: %4.2f, Old: %4.2f, New: %4.2f" % (
                        i,
                        applicable_actions[i],
                        q_values[i],
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

    def get_nn_targets(self, problem, nn, mini_batch):

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

    def train_network_on_minibatch(self, problem, new_ran, mini_batch, old_ran):

        abstraction = new_ran.get_abstraction()

        nn_pkgs_list = []
#         q_values = self.get_nn_targets(problem, new_ran.q_nn, mini_batch)

        for i in range(len(mini_batch)):

            current_state, applied_action, _, reward, done, _ = mini_batch[i]

            abstract_state = abstraction.create_abstract_state(
                problem,
                current_state)

            target = self.get_q_table_value(
                problem, current_state, applied_action, old_ran)

#             target = reward + self._gamma * q_values[i] * (1 - done)
            nn_pkg = abstraction.encode_nn_training_data(
                new_ran.q_nn.nn_input_call_set,
                new_ran.q_nn.nn_output_call_set,
                abstract_state=abstract_state,
                action=applied_action,
                q_s_a=target)

            nn_pkgs_list.append(nn_pkg)

        new_ran.q_nn.train(nn_pkgs_list, epochs=self._epochs,
                           batch_size=self._batch_size,
                           shuffle=True)

    def replay(self, problem, old_ran, new_ran):

        if len(self._replay_memory) < self._batch_size:

            return

        mini_batch = random.sample(self._replay_memory, self._batch_size)
        self.train_network_on_minibatch(problem, new_ran, mini_batch,
                                        old_ran)

    def qlearn_with_timestep_limit(self, problem, old_ran, new_ran, evaluator,
                                   cost_results, episode_results):

        sim = simulator.get_simulator(self._simulator_type)

        current_state = problem.get_initial_state()
        episodes_completed = 0
        episode_costs = []
        episode_time_step = 0

        for time_step in range(self._max_timesteps):

            action, is_random = self.get_action(problem, current_state,
                                                old_ran, new_ran,
                                                episodes_completed,
                                                episode_time_step,
                                                time_step)
            next_state, reward, done = sim.apply_action(problem,
                                                        current_state,
                                                        action)

            episode_time_step += 1
            self.td_update(problem, current_state, action, next_state, reward,
                           done, is_random, old_ran)

            self._replay_memory.append((current_state, action, next_state,
                                        reward, done, is_random))

            if time_step % self._training_interval == 0:
                for _ in range(self._total_train_iterations):

                    self.replay(problem, old_ran, new_ran)

            if done:

                episode_costs.append(episode_time_step)
                episodes_completed += 1
                episode_time_step = 0
                current_state = problem.get_initial_state()
            else:
                current_state = next_state

            if len(episode_costs) > 0:
                cost_results.add_data(problem, self._experiment_name,
                                      (time_step,
                                       statistics.mean(episode_costs)))

            episode_results.add_data(problem, self._experiment_name,
                                     (time_step, episodes_completed))

            self._epsilon *= self._epsilon_decay_rate
            self._epsilon = max(self._min_epsilon, self._epsilon)

    def qlearn_with_episode_limit(self, problem, old_ran, new_ran, evaluator,
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

            timestep_progress = tqdm.tqdm(disable=self._debug_print,
                                          total=self._timesteps_per_episode,
                                          unit=" timesteps",
                                          leave=False,
                                          position=1)

            current_state = problem.get_initial_state()
            done = False
            episode_reward = 0
            time_step = 0

            self.e_table = {}
            episode_s_a = []

            action, _ = self.get_action(
                problem,
                current_state,
                old_ran, new_ran,
                episode,
                time_step,
                global_time_step + time_step)

            while time_step < self._timesteps_per_episode:

                episode_s_a.append((current_state, action))

                next_state, reward, done = sim.apply_action(problem,
                                                            current_state,
                                                            action)

                time_step += 1
                timestep_progress.update(1)

                if done:

                    next_action = None
                    eligibility = True
                else:
                    next_action, _ = self.get_action(
                        problem,
                        next_state,
                        old_ran, new_ran,
                        episode,
                        time_step,
                        global_time_step + time_step + 1)
                    eligibility = self.get_eligibility_condition(problem,
                                                                 next_state,
                                                                 next_action,
                                                                 old_ran)

                episode_reward += reward

                self.td_update(problem, current_state, action, next_state, reward,
                               done, old_ran, eligibility, episode_s_a)

                self._replay_memory.append((current_state, action, next_state,
                                            reward, done, False))

                if done:

                    break
                else:
                    current_state = next_state
                    action = next_action

                if (global_time_step + time_step) % self._training_interval == 0:
                    for _ in range(self._total_train_iterations):

                        self.replay(problem, old_ran, new_ran)

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

        episode_progress.close()

    def qlearn(self, problem, old_ran, new_ran, evaluator, results):

        #         evaluator.setup_globals(problem)

        for episode in range(self._num_episodes):

            # Evaluate before the episode starts.
            #             stats = evaluator.evaluate(problem, nns=[(nn, 1.0)])
            #             results.add_data(problem, self._experiment_name, episode,
            #                              stats["mean"], stats["stdev"])

            time_steps = self.simulate(problem, old_ran, new_ran, episode)

            results.add_data(problem, self._experiment_name, episode,
                             time_steps, 0)

            print("******** [EPISODE] Episode %3u ended in %3u timesteps with epsilon %.2f" % (
                episode,
                time_steps,
                self._epsilon))

            self._epsilon *= self._epsilon_decay_rate
            self._epsilon = max(self._min_epsilon, self._epsilon)

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
            old_ran = RAN.load(self._old_ran_dir, self._nn_name)
            new_ran = RAN.load(self._old_ran_dir, self._nn_name)
        else:

            old_ran = RAN()

            new_ran = RAN.get_instance(self._abstraction,
                                       domain_filepath,
                                       problem_list,
                                       self._nn_type,
                                       self._nn_name,
                                       feature_file=self._feature_file)

        for problem_filepath in problem_list:

            self.q_table = {}

            # Reset epsilon to 1 for each problem.
            self._epsilon = self._get_value(
                self._phase_dict,
                "epsilon")

            problem = Problem(domain_filepath.name, problem_filepath.name,
                              problem_filepath.parent)

            self.qlearn_with_episode_limit(problem, old_ran, new_ran, evaluator,
                                           cost_results, episode_results)

            self._replay_memory.clear()
            new_ran.soft_save(input_dir)

            import tensorflow as tf
            tf.keras.backend.clear_session()

            # Reload the old ran as the new network.
            old_ran = RAN.load(input_dir, self._nn_name)
