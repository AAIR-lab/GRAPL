import pathlib
import sys
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
from utils import FileUtils
import collections
from pddlgym.structs import LiteralConjunction
from pddlgym.structs import Literal
from pddlgym.structs import State

from model import Model

from agent import Agent
from agent import PRPAgent
from tmp_sim import TMPSim

class CafeworldAgent(Agent):

    def __init__(self, gym_domain_name,
                 big_problem=False,
                 params={},
                 base_dir=None):

        self.gym_domain_name = gym_domain_name

        if big_problem:
            env = gym.make(
                "PDDLEnv{}Test-v0".format(gym_domain_name),
                dynamic_action_space=False,
                grounded_actions_in_state=config.SHOULD_FIND_GYM_ACTION_PREDS)
            env.fix_problem_index(0)
            self.env_path = "/root/git/TMP/test_domains/Cafeworld/Environments/env_big.dae"
            self.problem_path = "/root/git/TMP/test_domains/Cafeworld/Tasks/problem_big.pddl"
        else:

            env = gym.make(
                "PDDLEnv{}-v0".format(gym_domain_name),
                dynamic_action_space=False,
                grounded_actions_in_state=config.SHOULD_FIND_GYM_ACTION_PREDS)
            env.fix_problem_index(0)
            self.env_path = "/root/git/TMP/test_domains/Cafeworld/Environments/env_small.dae"
            self.problem_path = "/root/git/TMP/test_domains/Cafeworld/Tasks/problem_small.pddl"

        _ = env.reset()

        # Only need the pddlgym env to extract the actions and predicates
        # needed to do stuff with the code.
        domain, problem = learning_utils.extract_elements(
            env,
            0)
        self.actions = env.get_actions()

        super(CafeworldAgent, self).__init__(domain, problem)

        self.horizon = params.get("horizon", 40)
        self.naming_map = params.get("naming_map", {})
        self.args_func_map = params.get("args_func_map")

        self.base_dir = "%s/cafeworld_sim" % (base_dir)
        FileUtils.initialize_directory(self.base_dir, clean=True)
        self.tmp_sim = TMPSim("Cafeworld", self.base_dir,
                              domain, problem,
                              self.actions,
                              env_path=self.env_path,
                              problem_path=self.problem_path)

        self.problem.initial_state = self.tmp_sim.get_initial_state()

    def get_simulator(self):

        return self.tmp_sim

    def initialize_applicable_action_cache(self, applicable_action_cache,
                                           monte_carlo_steps=5):

        initial_state = self.tmp_sim.get_initial_state()

        fringe = [self.tmp_sim.initial_state]
        total_steps = 0
        visited = set()
        while len(fringe) > 0 \
                and len(applicable_action_cache) != len(self.domain.actions):

            state = fringe.pop()

            if state.literals in visited:
                continue

            visited.add(state.literals)
            self.tmp_sim.set_state(state)
            for action in self.actions:

                for _ in range(monte_carlo_steps):

                    next_state, execution_status = self.tmp_sim.execute_control(
                        action, state)

                    total_steps += 1
                    if execution_status \
                        and action.predicate.name not in applicable_action_cache:

                        applicable_action_cache[action.predicate.name] = \
                            state

                fringe.append(next_state)

        assert len(applicable_action_cache) == len(self.domain.actions)
        return total_steps

    def generate_samples(self, policy, initial_state=None,
                         sampling_count=config.SAMPLING_COUNT):

        policy.transform_to_pddlgym(self.problem)

        all_samples = []
        for _ in range(sampling_count):
            samples = PRP.generate_pddlgym_samples_using_policy(
                self.tmp_sim,
                self.domain,
                self.problem,
                policy,
                initial_state=initial_state,
                H=40,
                naming_map=self.naming_map,
                args_func_map=self.args_func_map)

            all_samples.append(samples)

        return all_samples

