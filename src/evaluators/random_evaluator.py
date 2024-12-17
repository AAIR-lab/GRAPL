
import pathlib
import sys
sys.path.append("%s/../" % (pathlib.Path(__file__).parent))
import config

import time
import random
import numpy as np
from pddlgym.core import PDDLEnv
from evaluators.result_logger import DiffResultsLogger
import os
import shutil
from utils.file_utils import FileUtils
import tqdm
import math
class StepExponentialDecay:

    def __init__(self, total_steps,
                 horizon,
                 start_epsilon=1.0,
                 min_epsilon=0.01):

        self.start_epsilon = start_epsilon

        if self.start_epsilon > 0:
            self.min_epsilon = min_epsilon

            self.decay_rate = -math.log(min_epsilon/start_epsilon) / (total_steps /
                                                                horizon)
            self.horizon = horizon

    def get_epsilon(self, total_steps):

        return 1.0

class RandomEvaluator:

    NAME = "random"

    def __init__(self, base_dir,
                 num_simulations=50,
                 max_reward=0,
                 debug=True,
                 enable_time_thread=False):

        self.base_dir = base_dir
        self.total_goals_reached = 0
        self.task_no = 0
        self.num_simulations = num_simulations

        assert max_reward == 0
        self.max_reward = 0
        self.reset_task()

        self.results_logger = DiffResultsLogger(self.base_dir,
                                                self.get_name(),
                                                self.get_logging_data,
                                                clean=True)

        self.enable_time_thread = enable_time_thread

        if self.enable_time_thread:
            self.results_logger.start()

        self.debug = debug


    def reset_task(self):

        self.qtable = None

        self.task_no = None
        self.horizon = None
        self.problem_name = None
        self.simulator = None
        self.action_space = None
        self.simulator_budget = None
        self.domain_file = None
        self.problem_file = None

        self.epsilon = None
        self.min_epsilon = None
        self.learning_rate = None
        self.gamma = None
        self.epsilon_decay = None

    def get_name(self):

        return RandomEvaluator.NAME

    def get_logging_data(self):

        avg_sr, avg_reward = self.get_success_rate_and_avg_reward()
        return {
            DiffResultsLogger.METADATA: None,
            DiffResultsLogger.HORIZON: self.horizon,
            DiffResultsLogger.TASK_NAME: self.problem_name,
            DiffResultsLogger.TASK_NO: self.task_no,
            DiffResultsLogger.NUM_SIMULATIONS: self.num_simulations,
            DiffResultsLogger.SIM_BUDGET: self.simulator_budget,
            DiffResultsLogger.SUCCESS_RATE: avg_sr,
            DiffResultsLogger.REWARD: avg_reward,
            DiffResultsLogger.GOALS: self.total_goals_reached

        }

    def get_q_value(self, s, a):

        return self.qtable.setdefault((s.literals, a), self.max_reward)

    def set_q_value(self, s, a, q_value):

        self.qtable[(s.literals, a)] = q_value

    def get_01_reward(self):

        return -1

    def get_max_q_value(self, s):

        sa_pairs = [(s, a) for a in self.action_space]

        q_values = [self.get_q_value(s, a) for s, a in sa_pairs]
        idx = np.argmax(q_values)

        return sa_pairs[idx], q_values[idx]

    def td_update(self, s, a, r, s_dash, done,
                  learning_rate, gamma):

        if done:
            q_dash = 0
        else:
            _, q_dash = self.get_max_q_value(s_dash)

        q_value = self.get_q_value(s, a)
        assert r == -1

        q_value = q_value + learning_rate * (r + gamma * q_dash - q_value)
        self.set_q_value(s, a, q_value)

    def get_action(self, s, action_space, epsilon):

        if random.random() < epsilon:

            return random.choice(action_space)
        else:

            sa_pair, _ = self.get_max_q_value(s)
            return sa_pair[1]

    def decay_epsilon(self, epsilon, epsilon_decay_rate, min_epsilon):

        epsilon = epsilon * epsilon_decay_rate
        epsilon = max(min_epsilon, epsilon)
        return epsilon

    def get_simulator(self, domain_file, problem_file,
                      shared_instance=True,
                      progress_bar=None,
                      step_callback=None):

        simulator = PDDLEnv(domain_file, problem_file,
                            operators_as_actions=True,
                            dynamic_action_space=False,
                            progress_bar=progress_bar,
                            shared_instance=shared_instance,
                            step_callback=step_callback)
        simulator.reset()

        return simulator

    def get_task_dir(self, simulator):

        task_dir = "%s/%s/task%u/" % (self.base_dir,
                                  self.get_name(),
                                  self.task_no)
        return task_dir

    def qlearn_episode(self, simulator, action_space, max_steps,
                       epsilon_decay,
                       learning_rate, gamma):

        sim_state = simulator.save_state()
        simulator.reset()

        episode_step = 0
        done = False
        while simulator.get_total_steps() < max_steps \
            and episode_step < self.horizon:

            s = simulator.get_state()
            epsilon = epsilon_decay.get_epsilon(simulator.get_total_steps())
            a = self.get_action(s, action_space, epsilon)
            s_dash, _, done, _ = simulator.step(a, True)
            r = self.get_01_reward()

            episode_step += 1
            # self.td_update(s, a, r, s_dash, done, learning_rate, gamma)

            if done:

                self.total_goals_reached += 1
                break

        simulator.restore_state(*sim_state)
        return -1 * episode_step, done

    def do_qlearning(self, simulator, action_space, max_steps,
                     epsilon_decay,
                     learning_rate, gamma):

        while simulator.get_total_steps() < max_steps:

            self.qlearn_episode(simulator, action_space, max_steps,
                                epsilon_decay,
                                learning_rate, gamma)

    def get_action_space(self, simulator):

        action_space = list(self.simulator.get_all_grounded_actions())
        return action_space

    def get_success_rate_and_avg_reward(self):

        if self.simulator is None or self.qtable is None:

            return 0.0, self.horizon
        else:

            simulator = self.get_simulator(self.domain_file,
                                           self.problem_file)
            action_space = self.get_action_space(simulator)

            total_reward = 0.0
            total_success = 0.0
            progress_bar = tqdm.tqdm(total=self.num_simulations,
                                     leave=False, position=1,
                                     disable=self.debug,
                                     unit=" simulations", ascii="  *",
                                     ncols=75, colour="red")
            epsilon_decay = StepExponentialDecay(None, None,
                                                 start_epsilon=0)
            for _ in range(self.num_simulations):

                reward, done = self.qlearn_episode(simulator, action_space,
                                                   float("inf"),
                                                   epsilon_decay,
                                                   0,
                                                   self.gamma)
                progress_bar.update(1)

                total_reward += reward
                total_success += int(done)

            progress_bar.close()
            return total_success / self.num_simulations, \
                total_reward / self.num_simulations

    def setup_task_dir(self):

        self.task_dir = self.get_task_dir(self.simulator)
        FileUtils.initialize_directory(self.task_dir, clean=True)

        with open("%s/%s.task" % (self  .task_dir, self.problem_name), "w") as fh:

            pass

        shutil.copy(self.domain_file, self.task_dir)
        shutil.copy(self.problem_file, self.task_dir)


    def solve_task(self, task_no, domain_file, problem_file,
                   horizon=40, simulator_budget=5000, epsilon=1.0,
                   min_epsilon=0.01,
                   learning_rate=0.3, gamma=0.9):

        self.task_no = task_no
        self.domain_file = os.path.abspath(domain_file)
        self.problem_file = os.path.abspath(problem_file)
        self.horizon = horizon
        self.simulator_budget = simulator_budget

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.qtable = {}

        self.epsilon_decay = StepExponentialDecay(simulator_budget, horizon,
                                                  start_epsilon=epsilon,
                                                  min_epsilon=min_epsilon)

        progress_bar = tqdm.tqdm(total=simulator_budget, unit=" steps",
                                 ascii="░▒█", disable=self.debug,
                                 leave=False, delay=0.05, colour="green")

        simulator = self.get_simulator(
            domain_file, problem_file,
            shared_instance=True,
            step_callback=self.results_logger.log_step_data,
            progress_bar=progress_bar)
        simulator.reset()

        self.setup_task_dir()
        self.simulator = simulator
        self.action_space = self.get_action_space(self.simulator)
        self.problem_name = self.simulator.get_problem().problem_name

        progress_bar.write("%s: Solving task %s" % (self.get_name(),
                                       self.problem_name))


        self.do_qlearning(
            self.simulator, self.action_space, self.simulator_budget,
            self.epsilon_decay,
            self.learning_rate, self.gamma)

        progress_bar.close()

    def shutdown(self):

        if self.enable_time_thread:
            self.results_logger.cancel()
            self.results_logger.join()
        pass

    def switch_to_last_good_model(self):

        pass

if __name__ == "__main__":

    # domain_file = "%s/tireworld/domain.pddl" % (config.BENCHMARKS_DIR)
    # problem_file = "%s/tireworld/training_problem.pddl" % (config.BENCHMARKS_DIR)
    #
    # qlearning = QLearningEvaluator("/tmp/results/")
    # qlearning.solve_task(domain_file, problem_file)

    total_steps = 60000
    horizon = 40
    decay = StepExponentialDecay(total_steps, horizon)
    for step in range(total_steps):

        if step % 1000 == 0:
            print("Step", step, " epsilon", decay.get_epsilon(step))


