import random
import statistics

import tqdm

from concretized.problem import Problem
from generalized_learning import simulator
from generalized_learning.concretized.state import State
from generalized_learning.util import constants
import numpy as np
from util import file


class ConcreteQL:

    DEFAULTS = {
        "episodes_per_epoch": 50,
        "max_timesteps": 7500,
        "num_episodes": 1500,
        "timesteps_per_episode": 250,
        "epsilon": 0.1,
        "min_epsilon": 0.01,
        "epsilon_decay_rate": 1.0,
        "gamma": 1.0,
        "experiment_name": "concrete_ql_0",
        "lambda": 0,
        "learning_rate": 0.8,
        "simulator_type": "generic",
        "debug_print": False,
        "ignore_stats_till": 0,
        "start_episode": 0,
    }

    def _get_value(self, config_dict, key):

        try:

            return config_dict[key]
        except KeyError:

            return ConcreteQL.DEFAULTS[key]

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

    def get_q_table_key(self, state, action):

        return (state, action)

    def initialize_q_table(self, problem, state):

        applicable_actions = problem.get_applicable_actions(state)

        for action in applicable_actions:

            dict_key = self.get_q_table_key(state, action)

            self.q_table[dict_key] = 0.0

    def get_q_table_value(self, problem, state, action):

        dict_key = self.get_q_table_key(state, action)

        try:

            return self.q_table[dict_key]
        except KeyError:

            self.initialize_q_table(problem, state)
            return self.q_table[dict_key]

    def get_eligibility_table_value(self, _problem, state, action):

        dict_key = self.get_q_table_key(state, action)

        return self.e_table.setdefault(dict_key, 0)

    def set_eligibility_table_value(self, _problem, state, action, value):

        dict_key = self.get_q_table_key(state, action)
        self.e_table[dict_key] = value

    def get_eligibility_condition(self, problem, next_state,
                                  selected_action):

        applicable_actions = problem.get_applicable_actions(next_state)
        max_value = self.get_q_table_value(problem, next_state,
                                           selected_action)

        for action in applicable_actions:

            value = self.get_q_table_value(problem, next_state, action)

            if value > max_value:

                return False

        return True

    def get_q_values(self, problem, state, applicable_actions):

        q_values = []
        for action in applicable_actions:

            q_values.append(self.get_q_table_value(problem, state, action))

        return q_values

    def get_best_q_value(self, problem, state):

        applicable_actions = problem.get_applicable_actions(state)
        random.shuffle(applicable_actions)

        q_values = self.get_q_values(problem, state, applicable_actions)
        return np.amax(q_values)

    def set_q_value(self, problem, state, action, value):

        dict_key = self.get_q_table_key(state, action)
        self.q_table[dict_key] = value

    def td_update(self, problem, current_state, action, next_state, reward,
                  done, eligibility, episode_s_a):

        q_value = self.get_q_table_value(problem, current_state, action)

        if done:

            q_dash = 0
        else:

            q_dash = self.get_best_q_value(problem, next_state)

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

        pass

    def get_action(self, problem, current_state, episode, episode_time_step,
                   time_step):

        applicable_actions = problem.get_applicable_actions(current_state)
        assert len(applicable_actions) > 0

        # Shuffle the actions.
        random.shuffle(applicable_actions)

        random_action = random.sample(applicable_actions, 1)[0]

        q_values = self.get_q_values(
            problem, current_state, applicable_actions)

        if self._debug_print:

            print("==== get_action(): Episode: %3u (step: %u), Timestep: %3u, Epsilon: %1.2f ===="
                  % (episode, episode_time_step, time_step, self._epsilon))

            for i in range(len(applicable_actions)):
                print(
                    "[%3u]: Action: %30s, Concrete: %4.2f" % (
                        i,
                        applicable_actions[i],
                        q_values[i],
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

    def simulate(self, problem, episode):

        current_state = problem.get_initial_state()

        time_step = 0
        done = False
        while not done \
                and time_step < self._timesteps_per_episode:

            time_step += 1

            action, is_random = self.get_action(problem, current_state,
                                                episode, time_step)
            next_state, reward, done = self.apply_action(problem,
                                                         current_state,
                                                         action)

            self.td_update(problem, current_state, action, next_state, reward,
                           done, is_random)
            current_state = next_state

        return time_step

    def qlearn_with_timestep_limit(self, problem, evaluator,
                                   cost_results, episode_results):

        sim = simulator.get_simulator(self._simulator_type)

        current_state = problem.get_initial_state()
        episodes_completed = 0
        episode_time_step = 0
        episode_costs = []
        for time_step in range(self._max_timesteps):

            action, is_random = self.get_action(problem, current_state,
                                                episodes_completed,
                                                episode_time_step,
                                                time_step)
            next_state, reward, done = sim.apply_action(problem,
                                                        current_state,
                                                        action)

            episode_time_step += 1
            self.td_update(problem, current_state, action, next_state, reward,
                           done, is_random)

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

            self.e_table = {}
            episode_s_a = []

            action, _ = self.get_action(
                problem,
                current_state,
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
                        episode,
                        time_step,
                        global_time_step + time_step + 1)
                    eligibility = self.get_eligibility_condition(problem,
                                                                 next_state,
                                                                 next_action)

                episode_reward += reward

                self.td_update(problem, current_state, action, next_state, reward,
                               done, eligibility, episode_s_a)

                if done:

                    break
                else:
                    current_state = next_state
                    action = next_action

            timestep_progress.close()
            episode_progress.update(1)

            # Update the global time step.
            global_time_step += time_step

            # Decay epsilon.
            self._epsilon *= self._epsilon_decay_rate
            self._epsilon = max(self._min_epsilon, self._epsilon)

            if ignore_stats:

                continue

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

    def qlearn(self, problem, evaluator, results):

        #         evaluator.setup_globals(problem)

        for episode in range(self._num_episodes):

            # Evaluate before the episode starts.
            #             stats = evaluator.evaluate(problem, nns=[(nn, 1.0)])
            #             results.add_data(problem, self._experiment_name, episode,
            #                              stats["mean"], stats["stdev"])

            time_steps = self.simulate(problem, episode)

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

#         evaluator = DQNEvaluator(self._phase_dict)
        evaluator = None

        for problem_filepath in problem_list:

            self.q_table = {}

            # Reset epsilon to 1 for each problem.
            self._epsilon = self._get_value(
                self._phase_dict,
                "epsilon")

            problem = Problem(domain_filepath.name, problem_filepath.name,
                              problem_filepath.parent)

            self.qlearn_with_episode_limit(problem, evaluator,
                                           cost_results, episode_results)
