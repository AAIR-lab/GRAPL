
import docker

import pickle
import random
import os

from utils import FileUtils
from pddlgym.structs import LiteralConjunction
from pddlgym.structs import State
import pathlib

class TMPPolicy:

    ID_KEY = "tmp_py3_policy"
    ROOT_NODE_ID = 0

    def __init__(self, policy_filepath):

        self.policy = {}
        self.goal = None

        try:
            file_handle = open(policy_filepath, "rb")
            self.policy = pickle.load(file_handle, encoding="latin1")
            assert self.policy[TMPPolicy.ID_KEY] == True
            self._is_valid = True
        except Exception:

            self._is_valid = False

    def get_goal_state(self):

        if not self.is_goal_reachable():
            return None
        else:
            for node_idx in self.policy:

                if isinstance(node_idx, str):
                    continue

                if len(self.policy[node_idx]["child_ids"]) == 0:
                    return self._get_state_from_node(node_idx)

            assert False
            return None

    def is_valid(self):

        return self._is_valid

    def is_goal_reachable(self):

        return self._is_valid and self.policy["is_goal_reachable"]

    def get_state(self, node_idx):

        try:

            return self._get_state_from_node(node_idx)
        except KeyError:

            return None

    def _get_state_from_node(self, node_idx):

        return TMPState.make_state(
            self.policy[node_idx]["ll_state"],
            self.policy[node_idx]["hl_state"])

    def generate_transitions(self, sim, total_steps=200,
                             continue_on_goal=False,
                             reset_sim_to_original_state=True):
        if not self._is_valid:

            return [], 0, True

        assert not continue_on_goal
        assert reset_sim_to_original_state
        assert self.goal is None

        transitions = []
        step = 0
        parent_idx = TMPPolicy.ROOT_NODE_ID
        assert self.policy[parent_idx]["simulation"] is None

        while step < total_steps:

            child_ids = self.policy[parent_idx]["child_ids"]

            if len(child_ids) == 0:
                break

            probabilities = self.policy[parent_idx]["probabilities"]
            child_idx = random.choices(child_ids, weights=probabilities, k=1)[0]

            child_node = self.policy[child_idx]
            is_goal = child_node["is_goal"]

            parent_state = self._get_state_from_node(parent_idx)
            idxs = [parent_idx, child_idx]
            if self.policy["has_simulation"]:

                state = TMPState(child_node["simulation"][0][0],
                                 child_node["simulation"][0][1])
                transitions.append((parent_state, None, state, idxs, is_goal))
                for i in range(1, len(child_node["simulation"])):

                    prev_state = TMPState(child_node["simulation"][i-1][0],
                                          child_node["simulation"][i-1][1])
                    state = TMPState(child_node["simulation"][i][0],
                                     child_node["simulation"][i][1])
                    transitions.append((prev_state, None, state, idxs, is_goal))
            else:

                state = self._get_state_from_node(child_idx)
                transitions.append((parent_state, None, state, idxs, is_goal))

            parent_idx = child_idx

        return transitions, 1, False

class TMPState:

    @staticmethod
    def make_state(ll_state_dict, hl_state_set):

        return TMPState(ll_state_dict, hl_state_set)

    @staticmethod
    def _convert_str_to_literal(predicate_str, domain):

        predicate_str = predicate_str.strip()
        predicate_str = predicate_str.replace("(", "")
        predicate_str = predicate_str.replace(")", "")

        predicate_str = predicate_str.split(" ")
        predicate = domain.predicates[predicate_str[0]]

        variables = []
        for i in range(len(predicate.var_types)):

            typed_arg = predicate.var_types[i](predicate_str[i+1])
            variables.append(typed_arg)

        return predicate(*variables)

    @staticmethod
    def convert_to_abstract(state, domain, with_copy=False):

        assert isinstance(state, TMPState)
        if with_copy:
            abstract_state = TMPState(None, None)
        else:
            abstract_state = state

        ll_state = state.ll_state
        literals = []
        for predicate_str in state.hl_state:

            literal = TMPState._convert_str_to_literal(predicate_str,
                                                       domain)
            literals.append(literal)

        abstract_state.hl_state = literals
        abstract_state.is_abstracted = True
        return abstract_state

    @staticmethod
    def convert_to_gym_state(state, objects):

        assert state.is_abstracted

        # Store a reference to this state in the "goal"
        # field of the gym state.
        return State(frozenset(state.hl_state),
                     objects,
                     state)

    def __init__(self, ll_state, hl_state):

        self.is_abstracted = False
        self.ll_state = ll_state
        self.hl_state = hl_state

class TMPSim:

    DOCKER_IMG_NAME = "jedai:dev"
    DOCKER_TMP_ROOT_DIR = "/root/git/TMP"
    DOCKER_TMP_EXEC_SCRIPT = "docker_tmp.sh"
    DOCKER_CONTROL_EXEC_SCRIPT = "docker_tmp_exec_control.sh"
    TMP_REFINED_TREE_PY3_NAME = "refined_tree.py3.pkl"

    @staticmethod
    def get_docker_image(docker_client):

        for image in docker_client.images.list():
            for tag in image.tags:

                if TMPSim.DOCKER_IMG_NAME == tag:
                    return image

        return None

    def __init__(self, domain_name, base_dir, domain, problem,
                 actions,
                 env_path=None,
                 problem_path=None):

        self.base_dir = base_dir
        self.domain = domain
        self.problem = problem
        self.actions = actions

        self.domain_name = domain_name
        self.call_idx = 0
        self.step_counter = 0

        self.assume_refinable = True
        self.store_simulated_executions = True

        self.env_path = env_path
        self.problem_path = problem_path

        docker_client = docker.from_env(timeout=None)
        image = TMPSim.get_docker_image(docker_client)
        assert image is not None

        self.docker_results_dir = "%s/%s" % (
            TMPSim.DOCKER_TMP_ROOT_DIR,
            self.domain_name
        )

        assert os.path.isdir(self.base_dir)
        self.container = docker_client.containers.create(
            image,
            tty=True,
            stdin_open=True,
            volumes={self.base_dir: {"bind": self.docker_results_dir,
                                     "mode": "rw"}})
        self.container.start()

        policy = self.compute_policy(output_dir_suffix="initial")
        assert policy.is_goal_reachable()

        self.initial_state = policy.get_state(TMPPolicy.ROOT_NODE_ID)
        self.initial_state = TMPState.convert_to_abstract(self.initial_state,
                                                          self.domain)
        self.initial_state = self.convert_to_gym_state(self.initial_state)
        self.current_state = self.initial_state
        self.execution_status = False

    def _get_docker_tmp_cmd(self, script_name, output_dir,
                            timeout_in_sec,
                            log_filename="output.log",
                            host_problem_filepath=None,
                            host_ll_filepath=None,
                            policy_filename=None):

        tmp_cmd = "%s/%s" % (TMPSim.DOCKER_TMP_ROOT_DIR,
                             script_name)

        tmp_cmd += "  %s" % (output_dir)
        tmp_cmd += "  %s" % (log_filename)
        tmp_cmd += " %s" % (timeout_in_sec)
        tmp_cmd += " --domain %s" % (self.domain_name)

        if self.assume_refinable:
            tmp_cmd += " --assume-refinable"

        tmp_cmd += " --store-policy-tree"

        if self.store_simulated_executions:
            tmp_cmd += " --store-simulated-executions"

        tmp_cmd += " --output-dir %s" % (output_dir)

        if host_problem_filepath is not None:

            problem_file = host_problem_filepath.name
            tmp_cmd += " --problem-file %s/%s" % (output_dir, problem_file)

        if host_ll_filepath is not None:

            ll_file = host_ll_filepath.name
            tmp_cmd += " --ll-file %s/%s" % (output_dir, ll_file)

        if policy_filename is not None:

            tmp_cmd += " --policy-file" % (output_dir, ll_file)

        tmp_cmd += " --env-path %s" % (self.env_path)

        return tmp_cmd

    def reset(self):

        self.current_state = self.initial_state
        self.execution_status = False
        return self.current_state, None

    def step(self, action):

        next_state, execution_status = self.execute_control(action, self.get_state())
        self.execution_status = execution_status
        self.set_state(next_state)
        return next_state, None, None, None

    def get_step_execution_status(self):

        return self.execution_status

    def get_actions(self):

        return self.actions

    def get_initial_state(self):

        return self.initial_state

    def set_state(self, state):

        assert isinstance(state, State)
        assert isinstance(state.goal, TMPState)
        assert state.goal.is_abstracted
        self.current_state = state

    def get_applicable_actions(self):

        raise NotImplementedError

    def get_domain(self):

        return self.domain

    def get_goal(self):

        raise NotImplementedError

    def get_state(self):

        return self.current_state

    def is_goal_reached(self, state, goal=None):

        raise NotImplementedError

    def write_problem(self, output_dir,
                      initial_state=None,
                      goal=None,
                      combined=False):

        if initial_state is None and goal is None:

            return None, None

        assert combined is False

        problem_filepath = pathlib.Path("%s/problem.pddl" % (output_dir))
        ll_filepath = pathlib.Path("%s/problem.pddl.ll.pkl" % (output_dir))

        FileUtils.remove_file(problem_filepath)
        FileUtils.remove_file(ll_filepath)

        if initial_state is not None:

            assert isinstance(initial_state, State)
            assert isinstance(initial_state.goal, TMPState)
            assert initial_state.goal.is_abstracted

            assert initial_state.goal.ll_state is not None
            ll_fh = open(ll_filepath, "wb")
            pickle.dump(initial_state.goal.ll_state, ll_fh, protocol=2)
            ll_fh.close()
        else:

            initial_state = State(frozenset(), None, None)
            ll_filepath = None

        if goal is not None:

            assert isinstance(goal, State)
            assert isinstance(goal.goal, TMPState)
            assert goal.goal.is_abstracted
            goal = LiteralConjunction(goal.goal.hl_state)

        self.problem.write(problem_filepath.as_posix(),
                           initial_state=initial_state.literals,
                           goal=goal,
                           fast_downward_order=True)

        return problem_filepath, \
            ll_filepath

    def compute_policy(self, initial_state=None, goal=None,
                       timeout_in_sec=240, output_dir_suffix=None):

        if output_dir_suffix is None:
            output_dir_suffix = ""

        host_output_dir = "%s/%s" % (self.base_dir, output_dir_suffix)
        docker_output_dir = "%s/%s" % (self.docker_results_dir, output_dir_suffix)

        policy_filepath = "%s/%s" % (host_output_dir,
                                     TMPSim.TMP_REFINED_TREE_PY3_NAME)
        log_filename = "output.log"

        FileUtils.initialize_directory(host_output_dir, clean=False)

        FileUtils.remove_file(policy_filepath)
        FileUtils.remove_file("%s/%s" % (host_output_dir, log_filename))

        host_problem_filepath, host_ll_filepath = self.write_problem(
            host_output_dir,
            initial_state,
            goal)

        tmp_cmd = self._get_docker_tmp_cmd(
            TMPSim.DOCKER_TMP_EXEC_SCRIPT,
            docker_output_dir,
            timeout_in_sec,
            log_filename=log_filename,
            host_problem_filepath=host_problem_filepath,
            host_ll_filepath=host_ll_filepath)

        print("*********", self.container.name)
        print(tmp_cmd)

        _ = self.container.exec_run(
            tmp_cmd,
            stderr=False,
            stdout=True,
            workdir=TMPSim.DOCKER_TMP_ROOT_DIR)

        policy = TMPPolicy(policy_filepath)
        return policy

    def is_state_valid(self, state):

        raise NotImplementedError

    def get_dirs(self, dir_suffix):

        host_dir = "%s/%s" % (self.base_dir, dir_suffix)
        docker_dir = "%s/%s" % (self.docker_results_dir, dir_suffix)

        return host_dir, docker_dir

    def get_docker_tmp_exec_control_cmd(self,
                                        output_dir,
                                        problem_filepath,
                                        log_filepath,
                                        timelimit_in_sec,
                                        action_name,
                                        policy_filename="policy.gv"):

        tmp_cmd = "%s/%s" % (TMPSim.DOCKER_TMP_ROOT_DIR,
                             TMPSim.DOCKER_CONTROL_EXEC_SCRIPT)

        tmp_cmd += " %s" % (self.domain_name)
        tmp_cmd += " %s" % (output_dir)
        tmp_cmd += " %s" % (problem_filepath.name)
        tmp_cmd += " %s" % (log_filepath.name)
        tmp_cmd += " %s" % (timelimit_in_sec)
        tmp_cmd += " %s" % (action_name)
        tmp_cmd += " %s" % (policy_filename)
        tmp_cmd += " %s" % (self.env_path)

        return tmp_cmd

    def _convert_pddlgym_action_to_control(self, action):

        control = action.predicate.name

        for arg in action.variables:

            control += ".%s" % (arg.name)

        return control

    def execute_control(self, action, state,
                        timelimit_in_sec=240,
                        output_dir_suffix=None):

        assert isinstance(state, State) \
               and isinstance(state.goal, TMPState)

        if output_dir_suffix is None:
            output_dir_suffix = ""

        host_output_dir, docker_output_dir = \
            self.get_dirs(output_dir_suffix)

        policy_filepath = "%s/%s" % (host_output_dir,
                                     TMPSim.TMP_REFINED_TREE_PY3_NAME)
        log_filepath = "%s/%s" % (host_output_dir, "output.log")

        FileUtils.initialize_directory(host_output_dir, clean=False)
        FileUtils.remove_file(policy_filepath)
        FileUtils.remove_file(log_filepath)

        host_problem_filepath, host_ll_filepath = self.write_problem(
            host_output_dir,
            state,
            goal=state)

        control = self._convert_pddlgym_action_to_control(action)
        tmp_cmd = self.get_docker_tmp_exec_control_cmd(docker_output_dir,
                                                       host_problem_filepath,
                                                       pathlib.Path(log_filepath),
                                                       timelimit_in_sec,
                                                       control)

        print("*********", self.container.name)
        print(tmp_cmd)
        _ = self.container.exec_run(
            tmp_cmd,
            stderr=False,
            stdout=True,
            workdir=TMPSim.DOCKER_TMP_ROOT_DIR)

        policy = TMPPolicy(policy_filepath)
        if policy.is_valid():

            transitions, _, _ = policy.generate_transitions(None, total_steps=1)
            _, _, _, idxs, _ = transitions[-1]
            next_state = policy.get_state(idxs[1])
            next_state = TMPState.convert_to_abstract(next_state,
                                                      self.domain)
            next_state = self.convert_to_gym_state(next_state)
            return next_state, True
        else:
            return state, False

    def convert_to_gym_state(self, state):

        assert isinstance(state, TMPState)
        return TMPState.convert_to_gym_state(state,
                                             self.problem.objects)
