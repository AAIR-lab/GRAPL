import os.path
import pathlib
import sys
sys.path.append("%s/../" % (pathlib.Path(__file__).parent))
import config

from pddlgym.core import PDDLEnv
from agent import PRPAgent
from model import Model
from utils import learning_utils
from interrogation.saia import AgentInterrogation
from utils.file_utils import FileUtils
from evaluators.result_logger import DiffResultsLogger
from planner import laostar
import shutil
import tqdm

class GTEvaluator:

    NAME = "oracle"

    def __init__(self, base_dir, num_simulations=50,
                 debug=True,
                 enable_time_thread=False):

        self.base_dir = base_dir
        self.total_goals_reached = 0
        self.task_no = 0
        self.num_simulations = num_simulations

        self.sampling_count = 5

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

        self.task_no = None
        self.horizon = None
        self.problem_name = None
        self.simulator = None
        self.simulator_budget = None
        self.domain_file = None
        self.problem_file = None
        self.policy = None

        self.model_no = None

    def get_name(self):

        return GTEvaluator.NAME

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

    def get_model_dir(self):

        return "%s/model%u" % (self.task_dir, self.model_no)

    def compute_policy(self, simulator):

        problem_file = "%s/lao.pddl" % (self.get_model_dir())
        problem = simulator.get_problem()
        fh = open(problem_file, "w")

        self.learned_model.write(fh, with_probabilities=True, close=False)
        problem.write(fh, fast_downward_order=True)
        fh.close()

        policy_file = "%s/lao.policy.txt" % (self.get_model_dir())
        solver = laostar.LAOStar(simulator, model=self.learned_model)
        policy = solver.solve(self.task_dir,
                     problem_file,
                     problem.problem_name,
                     policy_file)

        return policy

    def execute_episode(self, policy, simulator, max_steps):

        sim_state = simulator.save_state()
        simulator.reset()

        node_idx = policy.get_init_node_idx()

        episode_step = 0
        done = False
        while simulator.get_total_steps() < max_steps \
            and episode_step < self.horizon:

            state = simulator.get_state()
            assert policy.holds(node_idx, state.literals)

            action = policy.get_action(node_idx)
            if action is None:

                assert policy.is_deadend(node_idx)
                break
            else:

                # Update the step count first, since the callback triggers
                # on step.
                episode_step += 1

                next_state, _, done, _ = simulator.step(action, True)
                execution_status =  simulator.get_step_execution_status()
                assert execution_status
                assert self.learned_model.is_transition_conformant(
                    state, action, next_state, execution_status)

                node_idx = policy.get_sucessor_node_idx(node_idx,
                                                        next_state.literals)

                if done:
                    assert policy.is_goal(node_idx)
                    self.total_goals_reached += 1
                    break

        simulator.restore_state(*sim_state)
        return -1 * episode_step, done

    def get_success_rate_and_avg_reward(self):

        if self.simulator is None or self.policy is None:

            return 0.0, self.horizon
        else:

            simulator = self.get_simulator(self.domain_file,
                                           self.problem_file)

            progress_bar = tqdm.tqdm(total=self.num_simulations,
                                     leave=False, position=1,
                                     disable=self.debug,
                                     unit=" simulations", ascii="  *",
                                     ncols=75, colour="red")

            total_reward = 0.0
            total_success = 0.0
            for _ in range(self.num_simulations):

                reward, done = self.execute_episode(self.policy,
                                                  simulator,
                                                  max_steps=float("inf"))

                progress_bar.update(1)

                total_reward += reward
                total_success += int(done)

            progress_bar.close()
            return total_success / self.num_simulations, \
                total_reward / self.num_simulations

    def execute_policy(self, policy, simulator, max_steps=5000):

        while simulator.get_total_steps() < max_steps:

            self.execute_episode(policy, simulator, max_steps)

    def setup_task_dir(self):

        self.task_dir = self.get_task_dir(self.simulator)
        FileUtils.initialize_directory(self.task_dir, clean=True)

        with open("%s/%s.task" % (self.task_dir, self.problem_name), "w") as fh:

            pass

        shutil.copy(self.domain_file, self.task_dir)
        shutil.copy(self.problem_file, self.task_dir)


    def solve_task(self, task_no, domain_file, problem_file, horizon=40,
                   simulator_budget=5000):

        self.task_no = task_no
        self.domain_file = os.path.abspath(domain_file)
        self.problem_file = os.path.abspath(problem_file)
        self.horizon = horizon
        self.simulator_budget = simulator_budget

        progress_bar = tqdm.tqdm(total=simulator_budget, unit=" steps",
                                 ascii="░▒█", disable=self.debug,
                                 leave=False, delay=0.05, colour="green")

        simulator = self.get_simulator(domain_file, problem_file,
            shared_instance=True,
            step_callback=self.results_logger.log_step_data)
        simulator.reset()

        self.setup_task_dir()
        self.simulator = simulator
        self.problem_name = self.simulator.get_problem().problem_name

        progress_bar.write("%s: Solving task %s" % (self.get_name(),
                                                    self.problem_name))

        self.model_no = 0
        self.learned_model = Model(simulator.get_domain())
        self.learned_model = self.learned_model.flatten(with_copy=False)
        self.learned_model = self.learned_model.optimize(with_copy=False)
        FileUtils.initialize_directory(self.get_model_dir(), clean=True)

        policy = self.compute_policy(simulator)
        assert policy.has_path_to_goal()
        self.policy = policy

        self.execute_policy(self.policy, simulator,
                            max_steps=self.simulator_budget)

        progress_bar.close()

    def shutdown(self):

        if self.enable_time_thread:
            self.results_logger.cancel()
            self.results_logger.join()
        pass

    def switch_to_last_good_model(self):

        pass

if __name__ == "__main__":

    domain_file = "%s/tireworld/domain.pddl" % (config.BENCHMARKS_DIR)
    problem_file = "%s/tireworld/training_problem.pddl" % (config.BENCHMARKS_DIR)

    gt = GTEvaluator("/tmp/results")
    gt.solve_task(domain_file, problem_file, simulator_budget=10000)
