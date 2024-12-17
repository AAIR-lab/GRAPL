#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import sys

from pddlgym.core import PDDLEnv
import tqdm

sys.path.append("%s/../" % (pathlib.Path(__file__).parent))
import config

import os
import gym
import re
from model import Model
from planner.prp import PRP
import config
import random

from utils import learning_utils
import collections
from pddlgym.structs import LiteralConjunction
from pddlgym.structs import Literal
from pddlgym.structs import State

from model import Model
class Agent:
    def __init__(self, simulator):

        self.simulator = simulator
        self.model = Model(self.get_domain())

    def get_domain(self):

        return self.simulator.get_domain()

    def get_problem(self):

        return self.simulator.get_problem(0)
        
class BFSAgent(Agent):
    
    DEFAULT_FILTER_FUNC = lambda _s, _a, _s_dash, _depth: True
    DEFAULT_STORAGE_FUNC = lambda _s, _a, _s_dash, _depth: (_s, _a, _s_dash, _depth)
    
    def __init__(self, simulator,
                 params={},
                 base_dir=None,
                 dynamic_action_space=True,
                 monte_carlo_steps=float("-inf")):

        super(BFSAgent, self).__init__(simulator)
        self.monte_carlo_steps = monte_carlo_steps

        # TODO: Can store simulation logs there.
        assert base_dir is None
    
    def generate_state_samples(self, 
        pddlgym_state=None,
        max_depth=float("inf"),
        max_samples=float("inf"),
        filter_func = DEFAULT_FILTER_FUNC,
        storage_func = DEFAULT_STORAGE_FUNC,
        show_progress=False):

        sim_state = self.simulator.save_state()

        _ = self.simulator.reset()
        
        if pddlgym_state is not None:
            
            self.simulator.set_state(pddlgym_state)
        
        initial_state = self.simulator.get_state()
        fringe = collections.deque()
        visited = set()
        
        # previous_state, action_used, next_state, depth
        # First two terms are for debug.
        fringe.append((None, None, initial_state, 0))
        samples = []

        if show_progress:
            progress_bar = tqdm.tqdm(unit=" states")


        # A step in the environment is whenever we use a transition
        # to add an (s, a, s') tuple to the fringe.
        #
        # An SDM agent can be imagined to take so many "steps" in the
        # agent in RL mode in such a case and this would be the minimum
        # number of steps required (in the stochastic case).
        total_steps = 0

        while len(fringe) > 0 \
            and len(samples)  < max_samples:

            if show_progress:
                progress_bar.update(1)
            
            prev_state, action, state, depth = fringe.popleft()
            
            # Only add to samples if it passes the filter.
            if filter_func(prev_state, action, state, depth):
                samples.append(storage_func(prev_state, action, state, 
                                            depth))
            
            if state.literals in visited or depth >= max_depth:
                
                continue
            else:
                
                visited.add(state.literals)
                self.simulator.set_state(state)
                actions = self.simulator.get_applicable_actions()
                for action in actions:

                    self.simulator.set_state(state)

                    if self.simulator.domain.is_probabilistic \
                        and self.monte_carlo_steps == float("-inf"):
                        
                        transitions = self.simulator.get_all_possible_transitions(
                            action,
                            return_probs=False)
                        successors = [t[0] for t in transitions]
                    else:

                        successors = []
                        for _ in range(self.monte_carlo_steps):

                            self.simulator.set_state(state)
                            next_state, _, _, _ = self.simulator.step(action, True)
                            successors.append(next_state)

                            # Do not do MC if a failure status is returned.
                            # End the markove chain here.
                            if not self.simulator.get_step_execution_status():

                                break

                    total_steps += len(successors)
                    
                    fringe.extend([(state, action, x, depth + 1) for x in successors])

        if show_progress:
            progress_bar.close()

        self.simulator.restore_state(*sim_state)
        return samples, total_steps
    
    @staticmethod
    def example(gym_domain_name, problem_idx):
        
        print("Running BFS agent example")
        
        agent = BFSAgent(gym_domain_name, problem_idx, None)
        samples = agent.generate_state_samples()
        # print(samples)
        
        
class PRPAgent(Agent):
    
    def __init__(self, simulator,
                 params={},
                 base_dir=None):

        super(PRPAgent, self).__init__(simulator)

        import types
        import copy
        sim_problem = self.simulator.get_problem()
        self.problem = copy.deepcopy(sim_problem)

        self.problem.initial_state = State(sim_problem.initial_state,
                                           sim_problem.objects,
                                           sim_problem.goal)

        # TODO: Can store simulation logs there.
        assert base_dir is None
        
        self.horizon = params.get("horizon", 40)
        self.naming_map = params.get("naming_map", {})
        self.args_func_map = params.get("args_func_map")

    def get_problem(self):

        return self.problem

    def get_simulator(self):

        return self.simulator

    def get_state_where_action_is_applicable(self, action_name):
        
        filter_func = lambda s, a, s_dash, depth: a is not None \
            and a.predicate.name == action_name
        storage_func = lambda s, a, s_dash, depth: s.literals
        
        print("Finding for ", action_name)
        samples, total_steps = self.get_transitions(filter_func=filter_func,
                                    storage_func=storage_func,
                                    max_samples=1)
        
        return (None, total_steps) if len(samples) == 0 else (samples[0], total_steps)

    def initialize_applicable_action_cache(self, applicable_action_cache):

        class ActionFilter:

            def __init__(self, prp_agent, action_names):

                self.prp_agent = prp_agent
                self.action_names = action_names
                self.stored_actions = {}
            def filter(self, s, a, s_dash, depth):

                if a is not None \
                    and a.predicate.name not in self.stored_actions \
                    and self.prp_agent.get_execution_status(s, a):

                    self.stored_actions[a.predicate.name] = s
                    return True

            def storage(self, s, a, s_dash, depth):

                return s

        action_names = self.get_domain().actions
        action_filter = ActionFilter(self, action_names)

        _, total_steps = self.get_transitions(
            filter_func=action_filter.filter,
            storage_func=action_filter.storage,
            max_samples=len(action_names))

        assert len(action_filter.stored_actions) == len(action_names)
        for action_name in action_filter.stored_actions:

            applicable_action_cache[action_name] = \
                action_filter.stored_actions[action_name]

        return total_steps



    def get_transitions(self, filter_func=BFSAgent.DEFAULT_FILTER_FUNC,
                        storage_func=BFSAgent.DEFAULT_STORAGE_FUNC,
                        max_samples=float("inf"),
                        max_depth=float("inf"),
                        monte_carlo_steps=5):
        

        bfs_agent = BFSAgent(self.simulator,
                             dynamic_action_space=False,
                             monte_carlo_steps=monte_carlo_steps)
        
        samples = bfs_agent.generate_state_samples(
            max_samples=max_samples,
            max_depth=max_depth,
            filter_func=filter_func,
            storage_func=storage_func)
        
        return samples
            
    @staticmethod
    def example_get_transitions(domain_file, problem_file):

        agent = PRPAgent(domain_file, problem_file,
                         params={
                             "horizon": 100,
                             "naming_map": {},
                             "args_func_map": {}
                             })
        
        transitions = agent.get_transitions(max_samples=10)
        return transitions
        
    @staticmethod
    def example_get_state_where_action_is_applicable(domain_file,
                                                     problem_file,
                                                     action_name):
        
        agent = PRPAgent(domain_file, problem_file,
                         params={
                             "horizon": 100,
                             "naming_map": {},
                             "args_func_map": {}
                             })
        
        goal_state = agent.get_state_where_action_is_applicable(action_name)
        print(goal_state)

            
    def generate_samples(self, policy, initial_state=None,
                         sampling_count=config.SAMPLING_COUNT):


        policy.transform_to_pddlgym(self.problem)
        all_samples = []
        # problem_count = len(self.problems)
        for _i in range(sampling_count):
            
            samples = PRP.generate_pddlgym_samples_using_policy(
                self.simulator,
                self.get_domain(),
                self.get_problem(),
                policy,
                initial_state=initial_state,
                max_steps=250)
            all_samples.append(samples)

        return all_samples

    def get_execution_status(self, state, action, s_dash=None):

        if not isinstance(action, Literal):
            action_name = action[0]
            action_params = action[1]
            action_name = re.sub(r'\d', '', action_name)
            action = self.get_domain().predicates[action_name](*action_params)

        return self.simulator.is_action_applicable(action, state=state)

    @staticmethod
    def example_get_execution_status(domain_file, problem_file):

        agent = PRPAgent(domain_file, problem_file,
                         params={
                             "horizon": 100,
                             "naming_map": {},
                             "args_func_map": {}
                         })

        state = agent.env.get_state()
        applicable_actions = set(agent.env.get_actions(state=state, applicable_only=True))
        all_actions = set(agent.env.get_actions(state=state, applicable_only=False))

        # This example is not valid if every action is applicable.
        assert len(applicable_actions) != len(all_actions)

        for action_list, expect_status in [(all_actions - applicable_actions, False),
                                           (applicable_actions, True)]:
            for action in action_list:

                assert agent.get_execution_status(state, action) == expect_status

        pass

if __name__ == "__main__":
    
    # BFSAgent.example("Tireworld", 2)
    PRPAgent.example_get_execution_status("Tireworld", 0)
