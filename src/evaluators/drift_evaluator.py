import copy
import pathlib
import sys
import random

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
import tqdm
import os
import shutil
from model import UnconformantPreconditionException
from model import UnconformantEffectException
from exploration import drift_aware_bfs
from exploration import drift_explore
import pickle
from pddlgym.core import SimulatorOutOfBudgetException
from planner.prp import PRPPolicyNotFoundException
import collections
import numpy as np
import scipy

DEBUG = False

class DriftEvaluator:

    NAME = "drift"

    def __init__(self, base_dir,
                 sampling_count=5,
                 explore_mode="random_walk",
                 num_simulations=50,
                 num_rw_tries=25,
                 debug=True,
                 enable_time_thread=False,
                 failure_threshold=10):

        self.base_dir = base_dir
        self.total_goals_reached = 0
        self.task_no = 0
        self.num_simulations = num_simulations
        self.failure_threshold = failure_threshold

        self.randomize_pal = True

        self.sampling_count = sampling_count
        self.explore_mode = explore_mode

        self.model_no = 0
        self.model = None
        self.policy = None
        self.assessment_module = None

        self.reset_task()

        self.results_logger = DiffResultsLogger(self.base_dir,
                                                self.get_name(),
                                                self.get_logging_data,
                                                clean=True)

        self.debug = debug
        self.enable_time_thread = enable_time_thread
        self.last_good_model = None
        self.num_rw_tries = num_rw_tries

        self.lao_time_limit_in_sec = 90

        if self.enable_time_thread:
            self.results_logger.start()

        self.total_exceptions = 0
        self.experimental_effects_learning = True
        self.enable_gof_updates = True
        self.total_gof_updates = 0

    def reset_task(self):

        self.task_no = None
        self.horizon = None
        self.problem_name = None
        self.simulator = None
        self.simulator_budget = None
        self.domain_file = None
        self.problem_file = None
        self.policy = None
        self.gof_samples = {}
        self.gof_simulator = None
        self.total_gof_updates = 0

    def get_name(self):

        return DriftEvaluator.NAME

    def get_logging_data(self):

        avg_sr, avg_reward = self.get_success_rate_and_avg_reward()
        return {
            DiffResultsLogger.METADATA: (self.total_exceptions, self.total_gof_updates),
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
                      step_callback=None,
                      max_steps=float("inf")):

        simulator = PDDLEnv(domain_file, problem_file,
                            operators_as_actions=True,
                            dynamic_action_space=False,
                            progress_bar=progress_bar,
                            shared_instance=shared_instance,
                            step_callback=step_callback,
                            max_steps=max_steps)
        simulator.reset()

        return simulator

    def get_task_dir(self, simulator):

        task_dir = "%s/%s/task%u/" % (self.base_dir,
                                  self.get_name(),
                                  self.task_no)
        return task_dir

    def get_model_dir(self):

        return "%s/model%u" % (self.task_dir, self.model_no)

    def get_interrogation_params_for_task(self):

        agent = PRPAgent(self.simulator, {"horizon": self.horizon})

        gt_model = Model(agent.get_domain(), clean=True)

        return agent, gt_model


    def initialize_interrogation_algorithm(self):

        agent, gt_model = self.get_interrogation_params_for_task()
        domain = agent.get_domain()
        problem = agent.get_problem()

        abstract_actions = learning_utils.abstract_action(domain.operators)
        abstract_model = Model(domain, actions=abstract_actions)

        model_dir = self.get_model_dir()
        FileUtils.initialize_directory(model_dir, clean=True)

        self.assessment_module = AgentInterrogation(agent, abstract_model,
                                               problem, model_dir, gt_model,
                                               None, self.sampling_count,
                                               randomize=self.randomize_pal,
                                               should_count_sdm_samples=True,
                                               explore_mode=self.explore_mode,
                                               drift_mode=True)

    def setup_assessment_module_for_task(self):

        self.model_no += 1
        model_dir = self.get_model_dir()
        FileUtils.initialize_directory(model_dir, clean=True)

        agent, gt_model = self.get_interrogation_params_for_task()
        domain = agent.get_domain()
        problem = agent.get_problem()

        self.assessment_module.reset_for_task(agent, problem, gt_model,
                                              model_dir)

    def perform_gof(self, state, action, next_state,
                execution_status):

        try:
            self._perform_gof(state, action, next_state, execution_status)
        except Exception as e:

            pass

    def _perform_gof(self, state, action, next_state,
                     execution_status):

        if not self.enable_gof_updates:
            return

        action_dict = self.gof_samples.setdefault(action.predicate.name, {})
        effect_idx = self.model.get_effect_idx(state, action, next_state,
                                               execution_status)

        if effect_idx is None:
            self.gof_samples[action.predicate.name] = {}
            return

        effect_list = action_dict.setdefault(effect_idx, collections.deque(maxlen=500))
        effect_list.append((state, action, next_state, execution_status))

        samples = []
        probabilities = np.asarray(
            self.model.actions[action.predicate.name].effects.probabilities)
        if probabilities[-1] == 0:
            probabilities = probabilities[:-1]
        for i, probability in enumerate(probabilities):

            assert probability > 0
            samples.append(len(action_dict.get(i, [])))

        total_samples = sum(samples)

        if total_samples > 100:
            distribution = probabilities * total_samples
            chi_test = scipy.stats.chisquare(f_obs=samples, f_exp=distribution)
            if chi_test.pvalue < 0.05:

                self.total_gof_updates += 1
                lifted_effect =  self.model.actions[action.predicate.name].effects
                for i, sample in enumerate(samples):
                    lifted_effect.probabilities[i] = sample / total_samples

                lifted_effect.normalize_probabilities()
                self.compute_policy()

    def explore_and_learn(self):

        action = None
        for rw_attempt in range(self.num_rw_tries):

            print("drift-rw: Attempt %s/%s" % (
                (rw_attempt + 1),
                self.num_rw_tries))
            action, state, learn_all = drift_explore.explore(
                self.simulator,
                self.assessment_module.applicable_action_state_cache,
                self.model,
                self.assessment_module.taboo_state_dict,
                total_tries=1)

            if action is not None:
                break

        if action is None:

            print("Switching to bfs exploration.")
            action, state, learn_all = drift_aware_bfs.explore(
                self.simulator,
                self.assessment_module.applicable_action_state_cache,
                self.model,
                self.assessment_module.taboo_state_dict)

            assert action is not None
            print("drift-bfs-explore:", action, learn_all)
        else:
            print("drift-rw-explore:", action, learn_all)

        self.learn_action(action, state, learn_all)

    def learn_full_model(self):

        action_cache = self.assessment_module.applicable_action_state_cache

        while len(action_cache) != len(self.assessment_module.actions):

            self.explore_and_learn()

    def optimized_effects_learning(self, state, action):

        action_name = action.predicate.name
        action_vars = []
        for var in action.variables:

            action_vars.append(var.name)

        samples = []
        for _ in range(self.sampling_count):

            self.simulator.set_state(state)
            next_state, _, _, _ = self.simulator.step(action, True)
            execution_status = self.simulator.get_step_execution_status()
            assert execution_status

            sample = [(state, [action_name, action_vars], next_state,
                       execution_status)]
            samples.append(sample)

        self.assessment_module.analyze_samples(
            samples,
            self.assessment_module.abstract_model)


    def learn_action(self, action, state, learn_all):

        name = action.predicate.name
        self.assessment_module.set_all_pals_to_true(name)

        skip_interrogation = False
        if learn_all:

            self.assessment_module.add_to_taboo(name)
            self.assessment_module.prepare_action_for_complete_learning(name,
                                                                        state)
        else:

            if self.experimental_effects_learning:
                self.optimized_effects_learning(state, action)
                skip_interrogation = True
            else:
                self.assessment_module.prepare_action_for_effects_learning(name,
                                                                           state)

        try:

            if not skip_interrogation:
                model = self.assessment_module.agent_interrogation_algo(name)
            else:
                model = self.assessment_module.abstract_model
        except PRPPolicyNotFoundException as e:

            self.total_exceptions += 1
            self.assessment_module.add_to_taboo(name)
            return False
        except Exception as e:

            raise e

        model = model.flatten(with_copy=True)
        model = model.optimize(with_copy=False)

        if self.model is None:
            self.model = model
        else:
            self.model.actions[name] = copy.deepcopy(model.actions[name])

        return True

    def get_success_rate_and_avg_reward(self):

        if self.simulator is None or self.policy is None:

            return 0.0, -self.horizon
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

                try:
                    reward, done = self.execute_episode(
                        self.policy, simulator,
                        max_steps=float("inf"),
                        do_gof=False)
                except Exception:

                    reward = -self.horizon
                    done = False

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


    def compute_policy(self):

        model_dir = self.get_model_dir()

        # Create the model dir for this task just in case we are using
        # a previously used model.
        FileUtils.initialize_directory(model_dir, clean=False)

        problem_file = "%s/lao.pddl" % (model_dir)
        policy_file = "%s/lao.policy.txt" % (model_dir,)
        log_file = "%s.log" % (problem_file)

        FileUtils.remove_file(problem_file)
        FileUtils.remove_file(policy_file)
        FileUtils.remove_file(log_file)

        problem = self.simulator.get_problem()
        fh = open(problem_file, "w")

        self.model.write(fh, with_probabilities=True, close=False)
        problem.write(fh, fast_downward_order=True)
        fh.close()

        solver = laostar.LAOStar(self.simulator, model=self.model)
        self.policy = solver.solve(model_dir,
                     problem_file,
                     problem.problem_name,
                     policy_file,
                     time_limit_in_sec=self.lao_time_limit_in_sec)

    def execute_episode(self, policy, simulator, max_steps,
                        do_gof=True):

        sim_state = simulator.save_state()
        simulator.reset()

        node_idx = policy.get_init_node_idx()

        episode_step = 0
        done = False
        unconformant_effect_data = None

        while policy.has_path_to_goal() \
            and simulator.get_total_steps() < max_steps \
            and episode_step < self.horizon \
            and node_idx is not None:

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
                execution_status = simulator.get_step_execution_status()

                precondition_conformant, effect_conformant = \
                    self.model.is_transition_conformant(state, action,
                                                        next_state,
                                                        execution_status)
                if not precondition_conformant:

                    raise UnconformantPreconditionException(
                        state, action, next_state, execution_status)
                elif not effect_conformant:

                    if unconformant_effect_data is None:
                        unconformant_effect_data = (state, action, next_state,
                                                    execution_status)
                    raise UnconformantEffectException(
                        state, action, next_state, execution_status)
                elif do_gof:
                    self.perform_gof(state, action, next_state,
                                                    execution_status)

                node_idx = policy.get_sucessor_node_idx(node_idx,
                                                        next_state.literals)

                if done:
                    assert policy.is_goal(node_idx)
                    self.total_goals_reached += 1
                    break

        simulator.restore_state(*sim_state)

        if not done:

            if unconformant_effect_data is not None:
                raise UnconformantEffectException(*unconformant_effect_data)

            return -self.horizon, done
        else:
            return -1 * episode_step, done

    def execute_policy(self, policy, simulator, max_steps=5000):

        while simulator.get_total_steps() < max_steps:

            reward, done = self.execute_episode(policy, simulator, max_steps)

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
            step_callback=self.results_logger.log_step_data,
            max_steps=simulator_budget)
        simulator.reset()

        self.gof_simulator = self.get_simulator(domain_file, problem_file,
            shared_instance=False)
        self.gof_simulator.reset()

        self.simulator = simulator
        self.problem_name = self.simulator.get_problem().problem_name
        self.setup_task_dir()

        progress_bar.write("%s: Solving task %s" % (self.get_name(),
                                                    self.problem_name))

        if self.assessment_module is None:

            self.initialize_interrogation_algorithm()
        else:
            self.setup_assessment_module_for_task()


        if DEBUG and task_no == 0 and os.path.exists("/tmp/fr.pkl"):

            with open("/tmp/fr.pkl", "rb") as fh:
                self.assessment_module = pickle.load(fh)

            with open("/tmp/rs.pkl", "rb") as fh:
                rs = pickle.load(fh)
                random.setstate(rs)

            self.model = copy.deepcopy(self.assessment_module.abstract_model)
            self.model = self.model.flatten(with_copy=True)
            self.model = self.model.optimize(with_copy=False)
            self.setup_assessment_module_for_task()

        if self.model is None:
            self.learn_full_model()

        # Override the name of the model to the current task.
        self.model.domain_name = simulator.get_domain().domain_name

        self.compute_policy()
        failed_count = 0
        while self.simulator.get_total_steps() < self.simulator_budget:

            if not self.policy.has_path_to_goal() \
                    or failed_count > self.failure_threshold:

                self.explore_and_learn()
                self.compute_policy()
                failed_count = 0

            try:
                reward, done = self.execute_episode(self.policy, self.simulator,
                                    max_steps=self.simulator_budget)
                if not done:
                    failed_count += 1
                else:
                    failed_count = 0
            except UnconformantPreconditionException as e:

                print("Learning preconditions")
                self.assessment_module.add_to_taboo(e.a.predicate.name)
                self.gof_samples[e.a.predicate.name] = {}
                self.learn_full_model()
                self.compute_policy()
                failed_count = 0
            except UnconformantEffectException as e:

                print("Learning effects")
                self.gof_samples[e.a.predicate.name] = {}
                self.learn_action(e.a, e.s, False)
                self.compute_policy()
                failed_count = 0
            except SimulatorOutOfBudgetException as e:

                break

        progress_bar.close()
        self.reset_task()

        if DEBUG:

            self.assessment_module.agent = None
            self.assessment_module.am = None

            with open("/tmp/fr.pkl", "wb") as fh:
                pickle.dump(self.assessment_module, fh)

            with open("/tmp/rs.pkl", "wb") as fh:
                pickle.dump(random.getstate(), fh)

        self.last_good_model = copy.deepcopy(self.model)

    def switch_to_last_good_model(self):

        self.model = copy.deepcopy(self.last_good_model)

    def shutdown(self):

        if self.enable_time_thread:
            self.results_logger.cancel()
            self.results_logger.join()
        pass

if __name__ == "__main__":

    domain_file = "%s/tireworld/domain.pddl" % (config.BENCHMARKS_DIR)
    problem_file = "%s/tireworld/training_problem.pddl" % (config.BENCHMARKS_DIR)

    drift = DriftEvaluator("/tmp/results")
    drift.solve_task(domain_file, problem_file)
