import itertools
import pathlib
import sys
import tempfile

sys.path.append("%s/../" % (pathlib.Path(__file__).parent))
import config

from pddlgym.core import PDDLEnv
from utils import learning_utils
from utils.time_thread import TimeThread

from model import Model
import argparse
import os

from pddlgym import structs
import random
import numpy as np
import itertools
import math
from pddlgym.parser import Operator, PDDLDomain, PDDLProblemParser
from pddlgym.spaces import LiteralSpace
import types
import collections

from generator.action_generator import ActionGenerator
from generator.object_generator import ObjectGenerator
from generator.types_generator import TypesGenerator
from generator.predicate_generator import PredicateGenerator
from generator.init_state_generator import InitStateGenerator

from planner.laostar import LAOStar, LAOStarPolicy
from tqdm import tqdm

class RandomTaskGenerator:

    def __init__(self,
                 types_r=(1, 1),
                 predicates_r=(3, 6),
                 predicate_arity_r=(1, 2),
                 num_actions_r=(5, 5),
                 action_extra_param_r=(0, 0),
                 action_precons_r=(1, 2),
                 num_effects_r=(2, 2),
                 action_effects_r=(1, 2),
                 action_effect_bin_size=0.1,
                 num_objects_r=(4,4)):

        self.DEBUG_DOMAIN = "%s/tireworld/domain.pddl" % (
            config.BENCHMARKS_DIR)

        self.DEBUG_PROBLEM= "%s/tireworld/training_problem.pddl" % (
            config.BENCHMARKS_DIR)

        self.debug_env = PDDLEnv(self.DEBUG_DOMAIN, self.DEBUG_PROBLEM,
                           operators_as_actions=True)

        self.debug_domain, self.debug_problem = learning_utils.extract_elements(
            self.debug_env)
        self.debug_model = Model(self.debug_domain)

        self.types_r = types_r
        self.predicates_r = predicates_r
        self.predicate_arity_r = predicate_arity_r

        self.num_actions_r = num_actions_r
        self.action_extra_param_r = action_extra_param_r
        self.action_precons_r = action_precons_r
        self.action_num_effects_r = num_effects_r
        self.action_effects_r = action_effects_r
        self.action_effect_bin_size = action_effect_bin_size

        self.num_objects_r = num_objects_r

    def generate_types(self):

        num_types = random.randint(*self.types_r)
        return TypesGenerator(num_types).generate_types()

    def generate_predicates(self, types):

        predicate_generator = PredicateGenerator(types,
                                  self.predicates_r,
                                  self.predicate_arity_r)

        return predicate_generator.generate_predicates()

    def generate_domain_name(self, task_name):

        return "domain-%s" % (task_name)

    def generate_task_name(self, task_name):

        return "task-%s" % (task_name)

    def generate_actions(self, types, predicates):
        # actions_r, action_extra_param_r,
        # action_precondition_r,
        # action_effects_r, action_effect_bin_size,
        # predicates, types):

        action_generator = ActionGenerator(self.num_actions_r,
                                           self.action_extra_param_r,
                                           self.action_precons_r,
                                           self.action_num_effects_r,
                                           self.action_effects_r,
                                           self.action_effect_bin_size,
                                           types,
                                           predicates)
        actions, is_stochastic_domain = action_generator.generate_actions()
        return actions, is_stochastic_domain

    def generate_domain(self, task_name):

        otypes, type_hierarchy, type_to_parent_types = self.generate_types()

        fluent_predicates = self.generate_predicates(otypes)
        non_fluent_predicates = {}
        assert len(non_fluent_predicates) == 0

        predicates = fluent_predicates | non_fluent_predicates

        actions, is_stochastic_domain = self.generate_actions(otypes, predicates)

        domain = PDDLDomain(domain_name=None,
                            types=otypes,
                            type_hierarchy=type_hierarchy,
                            predicates=predicates,
                            operators=actions,
                            constants=None,
                            operators_as_actions=True,
                            is_probabilistic=is_stochastic_domain)
        domain.initialize_action_predicates_from_operators()

        return domain

    def generate_objects(self, domain):

        num_objects = random.randint(*self.num_objects_r)
        return ObjectGenerator(num_objects, domain.types).generate_objects()

    def _is_applicable(self, state, action):

        return action.preconds.holds(state)

    def _can_make_applicable_in_state(self, state, action):

        for literal in action.preconds.literals:

            if literal.is_negative and literal.positive in state:

                return False

        return True

    def _make_applicable_in_state_action(self, state, action):

        assert self._can_make_applicable_in_state(state, action)

        for literal in action.preconds.literals:

            if not literal.is_negative:

                state.add(literal)

    def _apply(self, state, a, ground_actions_dict,
               uncovered_actions,
               covered_actions):

        action = ground_actions_dict[a]
        successors = []
        if action.preconds.holds(state):

            uncovered_actions.difference_update([a])
            covered_actions.add(a)

            if isinstance(action.effects, list):
                effects = action.effects
            else:
                effects = [action.effects]

            for effect in effects:

                if isinstance(effect,structs.ProbabilisticEffect):

                    successors += effect.apply_all(state)
                else:
                    successors.append(effect.apply(state))

        return successors

        pass

    def _perform_walk(self, init_state, ground_actions_dict,
                      all_actions, visited,
                      uncovered_actions,
                      action_depth,
                      covered_actions,
                      max_visited_states=10000):

        init_state = set(frozenset(init_state))
        fringe = collections.deque()

        fringe.append((init_state, 0))
        while len(fringe) > 0 \
                and len(uncovered_actions) > 0 \
                and len(visited) < max_visited_states:

            state, depth = fringe.popleft()
            for a in all_actions:

                successors = self._apply(state, a,
                                         ground_actions_dict,
                                         uncovered_actions,
                                         covered_actions)

                if len(successors) > 0:
                    action_depth[a.predicate.name] = \
                        min(action_depth[a.predicate.name], depth)

                for successor in successors:

                    if frozenset(successor) not in visited:

                        visited.add(frozenset(successor))
                        fringe.append((successor, depth+1))



    def generate_init_and_goal_state(self, domain, objects,
                            time_limit_in_sec=60):

        init_state_generator = InitStateGenerator()
        init_state = init_state_generator.generate_init_state(
            domain, objects, time_limit_in_sec=time_limit_in_sec)
        depth_goal_dict, action_depth = init_state_generator.generate_goal_state(
            domain, objects, init_state,
            time_limit_in_sec=time_limit_in_sec)

        # Remove horizon 0 from the goal dict. We never want
        # to generate a goal state that is equal to the
        # initial state.
        depth_goal_dict.pop(0)
        return init_state, depth_goal_dict, action_depth


        problem = PDDLProblemParser(None, domain.domain_name,
                                    domain.types,
                                    domain.predicates,
                                    domain.actions)
        problem.objects = objects
        problem.goal = structs.LiteralConjunction(goal_state)
        problem.initial_state = init_state

        model = Model(domain, clean=True)
        model.write("/tmp/domain.pddl", with_probabilities=True)
        problem.write("/tmp/problem.pddl",
                      fast_downward_order=True)

        env = PDDLEnv("/tmp/domain.pddl", "/tmp/problem.pddl",
                      operators_as_actions=True,
                      dynamic_action_space=True)

    def _compute_policy(self, domain, problem,
                        time_limit_in_sec=60,
                        keep_files=False):

        model = Model(domain, clean=True)
        domain_fh = tempfile.NamedTemporaryFile(mode="w",
                                                delete=False)
        problem_fh = tempfile.NamedTemporaryFile(mode="w",
                                                 delete=False)
        lao_fh = tempfile.NamedTemporaryFile(mode="w",
                                             delete=False)

        model.write(lao_fh, with_probabilities=True, close=False)
        model.write(domain_fh, with_probabilities=True, close=False)

        problem.write(lao_fh, fast_downward_order=True)
        problem.write(problem_fh, fast_downward_order=True)

        policy_file = "%s.policy.txt" % (lao_fh.name)
        # print("Policy file:", policy_file)

        domain_fh.close()
        problem_fh.close()
        lao_fh.close()

        simulator = PDDLEnv(domain_fh.name, problem_fh.name,
                            operators_as_actions=True)

        solver = LAOStar(simulator, model=model)
        policy = solver.solve(None,
            lao_fh.name,
            simulator.get_problem().problem_name,
            policy_file,
            time_limit_in_sec=time_limit_in_sec)

        if not keep_files:

            os.remove(domain_fh.name)
            os.remove(problem_fh.name)
            os.remove(lao_fh.name)

        if os.path.exists(policy_file):

            os.remove(policy_file)
            return policy
        else:
            return None

    def get_task_cost(self, domain, objects,
                             init_state, goal_node,
                             ignore_only_negated_fluents=True,
                             time_limit_in_sec=60,
                             keep_files=False):

        # print("Printing path to goal")
        # InitStateGenerator.print_transitions(goal_node)

        problem = PDDLProblemParser(None, domain.domain_name,
                                    domain.types,
                                    domain.predicates,
                                    domain.actions)
        problem.objects = objects
        goal_state = set(goal_node.state)

        domain.initialize_non_fluents()

        # Need negative conditions since otherwise it is a goal
        # formula.
        for literal in init_state - goal_node.state:

            if ignore_only_negated_fluents\
                and literal.predicate.name in domain.only_negated_fluents:
                continue
            else:
                goal_state.add(literal.negative)

        problem.goal = structs.LiteralConjunction(goal_state)
        problem.initial_state = init_state

        policy = self._compute_policy(domain, problem,
                                      time_limit_in_sec=15,
                                      keep_files=keep_files)

        if policy is not None:

            # LAOstar currently has a bug so it might
            # not guarantee a policy found.
            # See SHA
            # 1d08cf917f86d3cbed56c85654e1416d6864d302

            # The policy must have a goal since it comes
            # from our generated goal.
            # assert policy.has_path_to_goal()
            return policy.get_cost(LAOStarPolicy.INIT_NODE_IDX), goal_state, policy
        else:
            return LAOStarPolicy.DEAD_END_COST, goal_state, policy

    def generate_interesting_task(self, domain, objects,
                                   init_state, depth_goal_dict,
                                  max_cost=20,
                                  time_limit_in_sec=60):

        best_cost = LAOStarPolicy.DEAD_END_COST - max_cost
        best_problem = None

        # This removes horizon=0 from consideration.
        best_horizon = 0

        print("Trying to generate tasks with costs close to", max_cost)

        time_thread = TimeThread(time_limit_in_sec=time_limit_in_sec)
        time_thread.start()
        while time_thread.is_alive() and len(depth_goal_dict) > 0:

            horizons = sorted(depth_goal_dict.keys())
            for horizon in horizons:

                # Only look at problems with a longer horizon
                # than what we currently have.
                if horizon <= best_horizon:
                    del depth_goal_dict[horizon]
                    continue

                if not time_thread.is_alive():
                    break

                goal_nodes_list = depth_goal_dict[horizon]
                if len(goal_nodes_list) == 0:
                    del depth_goal_dict[horizon]
                    continue

                goal_node = goal_nodes_list.pop()
                # print("Checking a problem at horizon", horizon)
                cost, goal_state, policy = self.get_task_cost(domain, objects,
                                          init_state, goal_node)

                if cost < horizon:
                    cost = float("inf")

                cost_diff = abs(cost - max_cost)
                if cost_diff < best_cost and horizon >= best_horizon:

                    tqdm.write(
                        "Found a better problem at horizon %s with cost difference %s" % (
                          horizon, cost_diff))
                    best_problem = init_state, goal_state, horizon, cost, policy
                    best_cost = cost_diff
                    best_horizon = horizon

        time_thread.stop()
        time_thread.join()

        return (None, None, None, None, None) if best_problem is None \
            else best_problem

    def generate_problem(self, task_name, domain,
                         objects,
                         constraints=[],
                         seed_init_state=set(),
                         init_state_time_limit=60,
                         goal_check_time_limit=120,
                         max_cost=20,
                         max_depth=float("inf"),
                         enforce_all_actions=True):

        problem_name = self.generate_task_name(task_name)

        init_state_generator = InitStateGenerator()

        print("Generating initial states")
        init_state = init_state_generator.generate_init_state(
            domain, objects,
            seed_init_state=seed_init_state,
            constraints=constraints,
            time_limit_in_sec=init_state_time_limit)

        print("Generating goal states")
        depth_goal_dict, action_depth = init_state_generator.generate_goal_state(
            domain, objects, init_state,
            time_limit_in_sec=init_state_time_limit)

        all_represented = True
        for action, depth in action_depth.items():
            if depth > max_depth:
                all_represented = False
                break

        if not all_represented:

            return (None, None, None, None, None)

        init_state, goal_state, horizon, cost, policy = self.generate_interesting_task(
            domain, objects, init_state, depth_goal_dict,
            max_cost=max_cost,
            time_limit_in_sec=goal_check_time_limit)

        problem = PDDLProblemParser(None, domain.domain_name,
                                    domain.types,
                                    domain.predicates,
                                    domain.actions)
        problem.problem_name = problem_name
        problem.initial_state = init_state
        problem.goal = structs.LiteralConjunction(goal_state)
        problem.objects = objects

        if goal_state is None or len(goal_state) == 0:
            return (None, None, None, None, None)

        return problem, horizon, cost, policy, action_depth

    def generate_task(self, output_dir, task_name,
                      init_state_time_limit=15,
                      goal_check_time_limit=20,
                      max_cost=20):

        domain = self.generate_domain(task_name)
        domain.domain_name = self.generate_domain_name(task_name)

        objects = self.generate_objects(domain)

        problem = None
        while problem is None:

            problem, _, _, _, _ = self.generate_problem(
                task_name, domain, objects,
                init_state_time_limit=init_state_time_limit,
                goal_check_time_limit=goal_check_time_limit,
                max_cost=max_cost)

        self.write_task(output_dir, domain, problem)

    def write_task(self, output_dir, domain, problem):

        domain_file = "%s/%s.pddl" % (output_dir,
                                      domain.domain_name)
        problem_file = "%s/%s.pddl" % (output_dir,
                                       problem.problem_name)
        combined_file = LAOStar.get_combined_filename(problem_file)

        print("Writing domain file:", domain_file)
        model = Model(domain, clean=True)
        model = model.flatten(with_copy=True)
        model = model.optimize(with_copy=True)
        model.write(domain_file, with_probabilities=True)

        print("Writing problem file:", problem_file)
        problem.write(problem_file, fast_downward_order=True)

        print("Writing combined file:", combined_file)
        LAOStar.write_combined_file(combined_file, model, problem)

        return domain_file, problem_file, combined_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Random task generator")

    parser.add_argument("--base-dir", type=str,
                        default="/tmp/rtg",
                        help="The base directory to keep the files in")

    parser.add_argument("--min-actions", type=int,
                        default=1,
                        help="The mininum # of actions")

    import time
    seed = int(time.time())
    print("Using seed", seed)
    random.seed(seed)

    # Some good seeds
    # 1700183853, 1700266728, 1700277638

    # Unable to find solutions for
    # 1700261501, 1700266676

    rtg = RandomTaskGenerator()
    rtg.generate_task("/tmp", "t0")