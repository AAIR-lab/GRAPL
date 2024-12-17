'''
Created on Jan 3, 2023

@author: rkaria
'''
import copy
import pathlib
import sys
sys.path.append("%s/../" % (pathlib.Path(__file__).parent))

import config
import os
import subprocess
import networkx as nx
import re
import gym
from utils import learning_utils
import logging

from pddlgym.structs import State
import collections

class PRPPolicyNotFoundException(Exception):

    pass

class PRPPolicy:
    
    PRP_STATE_SEPARATOR = "\\n"
    INIT_NODE_IDX = "1"
    GOAL_NODE_IDX = "2"
    
    ACTION_SEPARATOR = "_"
    
    def __init__(self, G):
        
        self.G = G
        self.has_path_to_goal = self.G is not None \
            and PRPPolicy.GOAL_NODE_IDX in self.G.nodes \
            and len(self.G.in_edges(PRPPolicy.GOAL_NODE_IDX)) > 0
            
        self.pddlgym_compatible = False
        
    def is_goal_reachable(self):
        
        return self.has_path_to_goal
        
    def transform_to_pddlgym(self, pddlgym_problem):
        
        assert not self.pddlgym_compatible
        
        # Node 1 represents the initial state (not expressed)
        # Node 2 represents "a" goal state (expressed)
        # Any node with a label "X" is a deadend state (not expressed)
        # All other nodes have the label attribute set to the state
        for node in self.G.nodes:
            
            if "label" not in self.G.nodes[node]:
                
                assert node == PRPPolicy.GOAL_NODE_IDX
                continue
            
            label = self.G.nodes[node]["label"]
            
            if label in ["I", "G", "X"]:
            
                assert False
                continue
            else:
                
                label = label.replace('"', "")
                label = label.strip()
                
                # From pddlgym.parser.PDDLProblemParser#_parse_problem_initial_state()
                pddlgym_params = \
                    {obj.name: obj.var_type for obj in pddlgym_problem.objects}
                
                state = set()
                if label != "":
                    for predicate in label.split(PRPPolicy.PRP_STATE_SEPARATOR):
                        # Remove extra things added when solving the temporary planning problem
                        if predicate == "p_psi()":
                            continue
                        t_pred = copy.deepcopy((predicate))
                        predicate = predicate.split("(")[0].rstrip("_1") + "(" + "".join(t_pred.split("(")[1:])
                        predicate = predicate.split("(")[0].rstrip("_2") + "(" + "".join(t_pred.split("(")[1:])
                        literal = PRP.convert_predicate_from_prp_to_pddlgym(
                            predicate, pddlgym_problem, pddlgym_params)
                        
                        state.add(literal)
                    
                self.G.nodes[node]["label"] = frozenset(state)

        self.pddlgym_compatible = True

class PRP:
    
    ROOT_DIR = "%s/dependencies/prp/" % (config.PROJECT_ROOTDIR)
    SAIA_SCRIPT = "%s/src/saia_prp.sh" % (ROOT_DIR)
    
    PREDICATE_REGEX_STR = "(?P<predicate_name>(\w|\W)*)\\((?P<arg_list>(\w|\W)*)\\)"
    PREDICATE_REGEX = re.compile(PREDICATE_REGEX_STR)

    # We need the action regex to strip out trailing numbers in the prp
    # policy regex.
    #
    # As a consequence of this, domain files cannot have actions ending with
    # numbers.
    ACTION_REGEX = re.compile("(?P<action_name>(\w|\W)*[a-z|A-Z|\-])\d*")

    @staticmethod
    def get_pddlgym_object_map(problem):
        
        return { obj.name : obj for obj in problem.objects}
    
    @staticmethod
    def get_pddlgym_action_map(domain):
        
        return {action : domain.predicates[action] for action in domain.actions}

    @staticmethod
    def get_prp_action_name_and_args(action):

        action = action.split(PRPPolicy.ACTION_SEPARATOR)
        assert len(action) > 0

        return action[0], action[1:]

    @staticmethod
    def get_pddlgym_args_from_prp_args(args, obj_map):

        return [obj_map[arg] for arg in args]

    @staticmethod
    def remove_trailing_numbers_from_action(action_name):

        return PRP.ACTION_REGEX.match(action_name).group("action_name")


    @staticmethod
    def create_pddlgym_action_from_prp_action(action, obj_map,
                                              action_map,
                                              naming_map = {},
                                              args_func_map={}):
        
        name, args = PRP.get_prp_action_name_and_args(action)
        name = PRP.remove_trailing_numbers_from_action(name)
        name = naming_map.get(name, name)
        args = PRP.get_pddlgym_args_from_prp_args(args, obj_map)
        args = args_func_map.get(name, lambda x: x)(args)
        
        pddlgym_action = action_map[name](*args)
        return pddlgym_action

    @staticmethod
    def get_pddlgym_action(simulator, prp_action, is_separated=False):

        obj_map = PRP.get_pddlgym_object_map(simulator.get_problem())
        action_map = PRP.get_pddlgym_action_map(simulator.get_domain())

        if is_separated:

            prp_action = PRPPolicy.ACTION_SEPARATOR.join(prp_action.split(" "))

        pddlgym_action = PRP.create_pddlgym_action_from_prp_action(
            prp_action, obj_map, action_map)

        return pddlgym_action

    @staticmethod
    def generate_pddlgym_samples_using_policy(simulator, domain, problem, policy,
                                              initial_state=None,
                                              max_steps=40,
                                              naming_map = {},
                                              args_func_map={}):

        sim_state = simulator.save_state()

        if initial_state is not None:
            
            assert isinstance(initial_state, State)
            
            pddlgym_state = State(initial_state.literals,
                                  initial_state.objects,
                                  initial_state.goal)
            simulator.set_state(pddlgym_state)
        
        obs = simulator.get_state()
        obj_map = PRP.get_pddlgym_object_map(problem)
        action_map = PRP.get_pddlgym_action_map(domain)

        if obs.literals == policy.G.nodes[PRPPolicy.INIT_NODE_IDX]["label"]:
            
            node_idx = collections.deque([(obs, PRPPolicy.INIT_NODE_IDX)])
        else:

            node_idx = []

        step = 0
        samples = []
        execution_status = True
        visited = set()
        done = False
        while not done \
            and step < max_steps \
            and len(node_idx) > 0:

            # Get the node idx.
            obs, idx = node_idx.popleft()
            simulator.set_state(obs)
            visited.add(obs.literals)

            # Get the action corresponding to it.
            actions = set()
            possible_successors = []                    
            edges = policy.G.edges(idx, data=True)
            for edge in edges:
                
                possible_successors.append(edge[1])
                actions.add(edge[2]["label"])

            assert len(actions) <= 1
            if len(actions) == 0:

                # No action, default execution status is failure.
                samples.append((obs, None, None, False))
                continue
            
            action = actions.pop()
            prp_action = PRP.get_prp_action_name_and_args(action)
            action = PRP.create_pddlgym_action_from_prp_action(
                action, obj_map, action_map, naming_map, args_func_map)

            old_state = obs
            obs, _, _, _ = simulator.step(action, True)
            execution_status = simulator.get_step_execution_status()
            samples.append((old_state, prp_action, obs, execution_status))

            # Update the successor node indices.
            for successor in possible_successors:

                if successor == PRPPolicy.GOAL_NODE_IDX:
                    done = True
                    break

                state = policy.G.nodes[successor]["label"]
                if state == obs.literals \
                    and obs.literals not in visited:
                    node_idx.append((obs, successor))

            # Action executed, increment the horizon
            step += 1

        simulator.restore_state(*sim_state)
        return samples

    @staticmethod
    def convert_predicate_from_prp_to_pddlgym(
            predicate, problem, params,
            grounded_actions_in_state=config.SHOULD_FIND_GYM_ACTION_PREDS):
        
        regex_match = PRP.PREDICATE_REGEX.match(predicate)
        assert regex_match is not None
        name = regex_match.group("predicate_name")
        name = name.strip()
        
        arg_list = regex_match.group("arg_list")
        arg_list = arg_list.strip()
        
        if arg_list != "":
            pddlgym_predicate_str = "(%s %s)" % (name, arg_list)
        else:
            pddlgym_predicate_str = "(%s)" % (name)

        literal = problem._parse_into_literal(pddlgym_predicate_str, params)
        if literal.predicate.name in problem.action_names:
            
            assert grounded_actions_in_state 
            
        assert literal.predicate.name != "="
        
        return literal

    @staticmethod
    def solve(domain_file, problem_file, 
              output_dir=None,
              timelimit_in_sec=None,
              raise_exception=False):
        if output_dir and os.path.exists(output_dir+"sas_plan"):
            os.remove(output_dir+"sas_plan")
        problem_filedir = pathlib.Path(problem_file).parent
        if output_dir is None:
            
            output_dir = problem_filedir
            
        stdout_filehandle = open("%s/%s.log" % (output_dir,
            pathlib.Path(problem_file).name), "w")
        
        cmd_string = "%s %s %s %s" % (PRP.SAIA_SCRIPT,
                                      PRP.ROOT_DIR,
                                                domain_file,
                                                problem_file)
        
        # print(cmd_string)
        
        try:
            subprocess.run(cmd_string, shell=True, check=True,
                           cwd=output_dir,
                           stdout=stdout_filehandle,
                           stderr=subprocess.STDOUT,
                           timeout=timelimit_in_sec)
        except subprocess.CalledProcessError as e:
            
            # The log file should capture the error,
            # However, we can also choose to re-raise the exception.
            if raise_exception:

                raise(e)
            
            return None
            
        finally:
                
            stdout_filehandle.close()
            
        return PRP.parse_policy(PRP.get_policy_filepath(output_dir))
        
            
    @staticmethod
    def get_policy_filepath(output_dir, name="graph_state_based.dot"):
        
        return "%s/%s" % (output_dir, name)
            
    @staticmethod
    def parse_policy(policy_filepath):
        
        G =  nx.nx_agraph.read_dot(policy_filepath)
        policy = PRPPolicy(G)
        return policy

def _prp_benchmark_example():

    print("Example using prp's benchmarks")
    
    domain_file = "%s/fond-benchmarks/tireworld/domain.pddl" % (PRP.ROOT_DIR)
    problem_file = "%s/fond-benchmarks/tireworld/p02.pddl" % (PRP.ROOT_DIR)
    
    from utils import FileUtils
    
    FileUtils.initialize_directory(config.RESULT_DIR)
    print("Storing results in %s" % (config.RESULT_DIR))
    
    PRP.solve(domain_file, problem_file, output_dir=config.RESULT_DIR)

def test(self, name):
    """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
    if name.startswith("_"):
        raise AttributeError(f"accessing private attribute '{name}' is prohibited")
    return getattr(self.env, name)

def _pddlgym_example():

    print("Example using pddlgym benchmarks")
    
    domain_file = "%s/domain.pddl" % (config.RESULT_DIR)
    problem_file = "%s/problem.pddl" % (config.RESULT_DIR)
    
    from utils import FileUtils
    import pddlgym
    import gym
    import utils
    from model import Model


    DOMAIN_NAME = "Tireworld"
    PROBLEM_IDX = 0

    # FileUtils.initialize_directory(config.RESULT_DIR)
    print("Storing results in %s" % (config.RESULT_DIR))
    
    train_env = gym.make(
        "PDDLEnv{}-v0".format(DOMAIN_NAME),
        grounded_actions_in_state=config.SHOULD_FIND_GYM_ACTION_PREDS)
    train_env.fix_problem_index(PROBLEM_IDX)
    _ = train_env.reset()
    
    domain, problem= utils.extract_elements(train_env, PROBLEM_IDX)

    model = Model(domain)
    model.write(domain_file)    #
    problem.write(problem_file, fast_downward_order=True)
    # domain_str = agent_model.

    policy = PRP.solve(domain_file, problem_file, output_dir=config.RESULT_DIR)
    
    if policy.is_goal_reachable():
        
        print("Policy found")
        policy.transform_to_pddlgym(problem)
        samples = PRP.generate_pddlgym_samples_using_policy(
            train_env, domain, problem, policy,
            H = 30,
            naming_map = {},
            args_func_map = {})
    else:
        
        print("Policy not found.")

if __name__ == "__main__":

    _pddlgym_example()

