import copy
import itertools
import pathlib
import sys
import tempfile

sys.path.append("%s/../" % (pathlib.Path(__file__).parent))
import config

import random
import argparse

from enum import Enum
from utils.file_utils import FileUtils
from pddlgym.parser import PDDLDomainParser
from pddlgym.parser import PDDLProblemParser
from model import Model
from generator.object_generator import ObjectGenerator
from generator.task_generator import RandomTaskGenerator
from generator.action_generator import ActionGenerator

from benchmarks.tireworld import constraints as tireworld_constraints
from pddlgym import structs

class PreconditionDrift(Enum):

    REMOVE = 0
    ADD = 1
    MODIFY = 2

class EffectDrift(Enum):

    REMOVE = 0
    ADD = 1
    MODIFY = 2
    NEW_EFFECT = 3
    DISTRIBUTION_CHANGE = 4

class DriftGenerator:

    PRECON_MAP = {

        PreconditionDrift.REMOVE: 0.33,
        PreconditionDrift.ADD: 0.33,
        PreconditionDrift.MODIFY: 0.34
    }

    EFFECT_MAP = {

        EffectDrift.REMOVE: 0.33,
        EffectDrift.ADD: 0.33,
        EffectDrift.MODIFY: 0.34,
        EffectDrift.NEW_EFFECT: 0.0,
        EffectDrift.DISTRIBUTION_CHANGE: 0.0,
    }

    def __init__(self,
                 num_objects_r=(4, 4),
                 action_drift_r=(1, 1),
                 precondition_drift_r=(0, 3),
                 effect_drift_r=(1, 3),
                 precon_map=PRECON_MAP,
                 effect_map=EFFECT_MAP):

        self.action_drift_r = action_drift_r
        self.precondition_drift_r = precondition_drift_r
        self.effect_drift_r = effect_drift_r

        self.precon_map = precon_map
        self.effect_map = effect_map

        self.num_objects_r = num_objects_r
        self.problem_generator = RandomTaskGenerator(
            num_objects_r=num_objects_r)

    def generate_drift_task(self, types, predicates, actions):

        pass

    def get_task_name(self, domain_idx, task_idx=0):

        return "d%s-t%u" % (domain_idx, task_idx)

    def get_domain(self, domain_file):

        domain = PDDLDomainParser(domain_file,
                                  operators_as_actions=True,
                                  expect_action_preds=False)

        model = Model(domain, clean=True)
        model = model.flatten(with_copy=True)
        model = model.optimize(with_copy=False)

        for action in domain.operators:

            domain.operators[action] = model.actions[action]
            domain.operators[action].enforce_unique_params = False

        return domain

    def get_problem(self, domain, problem_file):

        problem = PDDLProblemParser(problem_file, domain.domain_name,
                domain.types, domain.predicates, domain.actions,
                domain.constants)

        return problem

    def write_files(self, output_dir, idx, model):

        pass

    def get_constraint(self, domain_name):

        if "tireworld" in domain_name:

            return [tireworld_constraints.has_one_vehicle]
        else:

            return []

    def get_add_remove_literal_set(self, literals, candidate_literals):

        candidate_literals = set(copy.deepcopy(candidate_literals))

        for literal in literals:

            if literal.is_negative:
                candidate_literals.discard(literal.positive)
            elif literal.is_anti:
                candidate_literals.discard(literal.inverted_anti)
            else:
                candidate_literals.discard(literal)

        return candidate_literals

    def prepare_action_for_drift(self, action):

        assert action.is_optimized
        assert isinstance(action.effects, structs.ProbabilisticEffect)

        for i, effect in enumerate(action.effects.literals):

            if action.effects.probabilities[i] == 0:

                continue

            assert isinstance(effect, structs.LiteralConjunction)
            effect.literals += copy.deepcopy(
                action.effects.common_effects.literals)

        action.effects.common_effects = structs.LiteralConjunction([])

        action.effects.is_optimized = False
        action.is_optimized = False

    def make_effects_conform_with_preconditions(self, action):

        precond_literals = set(action.preconds.literals)
        for i, effect in enumerate(action.effects.literals):
            if action.effects.probabilities[i] == 0:
                continue
            for j, literal in enumerate(effect.literals):

                if literal.is_anti:

                    pos_lit = literal.inverted_anti
                    neg_lit = pos_lit.negative
                    if neg_lit in precond_literals:

                        effect.literals[j] = pos_lit
                else:
                    anti_lit = literal.inverted_anti
                    if literal in precond_literals:

                        effect.literals[j] = anti_lit

    def get_modifiable_effects(self, precon_literals, effect_literals):

        modifiable_literals = []
        precon_literals = set(precon_literals)
        for literal in effect_literals:

            if literal.is_anti \
                    and literal.inverted_anti not in precon_literals:

                modifiable_literals.append(literal)
            elif literal.negative not in precon_literals:

                modifiable_literals.append(literal)

        return modifiable_literals

    def get_good_effect_indices(self, action):

        indices = []
        for i in range(len(action.effects.literals)):

            if action.effects.probabilities[i] > 0:

                indices.append(i)

        return indices

    def ensure_one_precon_changed(self, action):

        assert not action.is_optimized
        preconds = set(action.preconds.literals)
        precond_list = list(preconds)

        for i, effect in enumerate(action.effects.literals):

            assert isinstance(effect, structs.LiteralConjunction)
            if action.effects.probabilities[i] == 0:
                assert len(effect.literals) == 1
                name = effect.literals[0].predicate.name.lower()
                assert name == "nochange"
                continue

            found = False
            for literal in effect.literals:

                if literal.is_anti \
                    and literal.inverted_anti in preconds:
                    found = True
                    break
                elif literal.negative in preconds:
                    found = True
                    break

            if not found:

                random_precond = random.choice(precond_list)
                if random_precond.is_negative:

                    effect.literals.append(random_precond.positive)
                else:
                    effect.literals.append(random_precond.inverted_anti)


    def perform_effect_drift(self, action, candidate_literals):

        num_changes = random.randint(*self.effect_drift_r)
        precon_literals = action.preconds.literals

        change_types = list(self.effect_map.keys())


        indices = self.get_good_effect_indices(action)

        total_changes = 0
        while total_changes < num_changes:

            total_changes += 1
            change_weights = list(self.effect_map.values())

            eff_idx = random.choice(indices)
            effect_literals = action.effects.literals[eff_idx].literals
            addable_literals = self.get_add_remove_literal_set(
                effect_literals,
                candidate_literals)
            modifiable_effects = self.get_modifiable_effects(
                precon_literals, effect_literals)

            if len(addable_literals) == 0:

                idx = change_types.index(EffectDrift.ADD)
                change_weights[idx] = 0

            if len(modifiable_effects) == 0:
                idx = change_types.index(EffectDrift.MODIFY)
                change_weights[idx] = 0

            if len(effect_literals) == 1:
                idx = change_types.index(EffectDrift.REMOVE)
                change_weights[idx] = 0

            change_type = random.choices(change_types,
                                         weights=change_weights)[0]

            if change_type == EffectDrift.ADD:

                literal = addable_literals.pop()

                if literal in precon_literals:

                    effect_literals.append(literal.inverted_anti)
                elif literal.negative in precon_literals:

                    effect_literals.append(literal.positive)
                if random.random() < 0.5:

                    effect_literals.append(literal.inverted_anti)
                else:
                    effect_literals.append(literal.positive)
            elif change_type == EffectDrift.REMOVE:

                idx = random.randint(0, len(effect_literals) - 1)
                effect_literals.pop(idx)
            elif change_type == EffectDrift.MODIFY:

                idx = random.randint(0, len(modifiable_effects) - 1)
                idx = effect_literals.index(modifiable_effects[idx])

                effect_literals[idx] = \
                    effect_literals[idx].inverted_anti
            else:

                print("Warning: Nothin to change.")

        self.make_effects_conform_with_preconditions(action)
        self.ensure_one_precon_changed(action)


    def perform_precon_drift(self, action, candidate_literals):

        num_changes = random.randint(*self.precondition_drift_r)

        literals = action.preconds.literals
        candidate_literals = self.get_add_remove_literal_set(
            literals,
            candidate_literals)
        changed_literals = []

        change_types = list(self.precon_map.keys())
        change_weights = list(self.precon_map.values())

        total_changes = 0
        while total_changes < num_changes \
            and len(literals) > 0:

            if len(candidate_literals) == 0:

                idx = change_types.index(PreconditionDrift.ADD)
                change_weights[idx] = 0

            if len(literals) == 1:

                idx = change_types.index(PreconditionDrift.REMOVE)
                change_weights[idx] = 0

            change_type = random.choices(change_types,
                                         weights=change_weights)[0]

            if change_type == PreconditionDrift.REMOVE:

                total_changes += 1
                idx = random.randint(0, len(literals) - 1)
                literals.pop(idx)
            elif change_type == PreconditionDrift.ADD:

                total_changes += 1
                new_lit = candidate_literals.pop()
                if random.random() < 0.5:

                    changed_literals.append(new_lit)
                else:
                    changed_literals.append(new_lit.negative)
            elif change_type == PreconditionDrift.MODIFY:

                total_changes += 1
                idx = random.randint(0, len(literals) - 1)

                if literals[idx].is_negative:
                    changed_literals.append(literals[idx].positive)
                else:
                    changed_literals.append(literals[idx].negative)

                literals.pop(idx)
            else:

                assert False

        action.preconds = structs.LiteralConjunction(
            literals + changed_literals)

        is_positive = False
        for literal in action.preconds.literals:
            if not literal.is_negative:
                is_positive = True
                break

        if not is_positive:
            idx = random.randint(0, len(action.preconds.literals) - 1)
            lit = action.preconds.literals[idx]
            action.preconds.literals[idx] = lit.positive

    def has_drifted(self, old_action, new_action):

        old_p = set(old_action.preconds.literals)
        new_p = set(new_action.preconds.literals)

        if old_p != new_p:
            return True

        assert len(old_action.effects.common_effects.literals) == 0
        assert len(new_action.effects.common_effects.literals) == 0

        if len(old_action.effects.literals) != len(new_action.effects.literals):
            return True
        else:

            for i in range(len(old_action.effects.literals)):

                old_e = set(old_action.effects.literals[i].literals)
                new_e = set(new_action.effects.literals[i].literals)
                if old_e != new_e:

                    return True

        return False

    def perform_drift(self, domain, num_actions):

        actions = list(domain.operators.keys())
        num_actions = min(num_actions, len(actions))

        actions = random.sample(actions,
                                k=num_actions)

        drifted_actions = []
        for action_name in actions:

            action = domain.operators[action_name]
            self.prepare_action_for_drift(action)

            old_action = copy.deepcopy(action)
            candidate_literals = ActionGenerator.generate_candidate_literals(
                action.params,
                domain.get_non_action_predicates())
            candidate_literals = copy.deepcopy(candidate_literals)
            self.perform_precon_drift(action, candidate_literals)
            self.perform_effect_drift(action, candidate_literals)

            action = action.flatten(with_copy=True)
            action = action.optimize(with_copy=False)
            domain.operators[action_name] = action

            new_action = copy.deepcopy(action)
            self.prepare_action_for_drift(new_action)

            if self.has_drifted(old_action, new_action):
                drifted_actions.append(action_name)

        domain.initialize_non_fluents()

        return drifted_actions


    def generate_drift_tasks(self, output_dir, domain_file, problem_file,
                             total_tasks,
                             min_horizon=5,
                             clean=False):

        FileUtils.initialize_directory(output_dir,
                                       clean=clean)

        task_info = []
        idx = 0
        while idx < total_tasks:

            domain = self.get_domain(domain_file)
            problem = self.get_problem(domain, problem_file)

            constraints = self.get_constraint(domain.domain_name)
            task_name = self.get_task_name(idx, 0)

            if idx == 0:
                problem.problem_name = self.problem_generator.generate_task_name(
                    task_name)
                objects = problem.objects

                for name in domain.actions:
                    action = domain.operators[name]
                    self.prepare_action_for_drift(action)
                    self.make_effects_conform_with_preconditions(action)
                    self.ensure_one_precon_changed(action)
                    action = action.flatten(with_copy=True)
                    action = action.optimize(with_copy=False)
                    domain.operators[name] = action

            # objects = self.problem_generator.generate_objects(domain)

            domain.domain_name = "domain-d%s" % (idx)

            drifted_actions = set()
            if idx > 0:

                num_actions = random.randint(*self.action_drift_r)
                drifted_actions = set(self.perform_drift(domain, num_actions))
                print("Drifted actions", drifted_actions)
                problem, horizon, cost, new_policy, action_depth = \
                    self.problem_generator.generate_problem(
                        task_name, domain, objects,
                        seed_init_state=problem.initial_state,
                        constraints=constraints,
                        init_state_time_limit=15,
                        goal_check_time_limit=20)
            else:
                horizon = "default"
                cost = "default"

            if idx > 0 and len(drifted_actions) == 0:
                continue

            if problem is not None:
                old_domain = self.get_domain(domain_file)
                old_domain.domain_name = problem.domain_name
                policy = self.problem_generator._compute_policy(
                    old_domain,
                    problem)
            else:
                assert new_policy is None

            if policy is None:
                policy_actions = set(old_domain.actions)
            else:
                policy_actions = policy.get_actions()


            if idx > 0 and problem is not None:
                new_actions = new_policy.get_actions()
                for action in new_actions:

                    if action_depth[action] > 2:
                        horizon = None
                        break


            print("Drifted actions", drifted_actions)
            print("Policy actions", policy_actions)

            if (idx == 0) or (horizon is not None \
                and horizon >= min_horizon \
                and len(drifted_actions.intersection(policy_actions)) > 0) \
                and cost < 50:

                problem.domain_name = domain.domain_name
                domain_file, problem_file, _ = self.problem_generator.write_task(
                    output_dir, domain, problem)

                with open(domain_file, "a") as fh:

                    fh.write("\n")
                    fh.write("; Drifted actions\n")
                    for drifted_action in drifted_actions:

                        fh.write(";%s\n" % (drifted_action))

                task_info.append((task_name, horizon, cost))

                total_extra_tasks = 2
                for tidx in range(1, total_extra_tasks + 1):
                    seed_init_state = problem.initial_state
                    goal = set(problem.goal.literals)
                    new_goal = goal
                    same_goal_tries = 0
                    problem = None
                    horizon = float("-inf")
                    cost = float("inf")
                    while problem is None or horizon is None or \
                        horizon < min_horizon or cost > 50 or \
                        (goal == new_goal and same_goal_tries < 5):
                        problem, horizon, cost, policy, action_depth = \
                            self.problem_generator.generate_problem(
                                task_name, domain, objects,
                                seed_init_state=seed_init_state,
                                constraints=constraints,
                                init_state_time_limit=10,
                                goal_check_time_limit=10)

                        if problem is not None:
                            new_goal = set(problem.goal.literals)
                            same_goal_tries += int(new_goal == goal)

                    problem.domain_name = domain.domain_name
                    task_name = self.get_task_name(idx, tidx)
                    problem.problem_name = self.problem_generator.generate_task_name(
                        task_name)
                    domain_file, problem_file, _ = self.problem_generator.write_task(
                        output_dir, domain, problem)
                    task_info.append((task_name, horizon, cost))

                idx += 1

        for task_name, horizon, cost in task_info:

            print("Name: %s, Horizon: %s, Cost: %s" % (task_name,
                                                       horizon,
                                                       cost))


        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Drift Generator")

    parser.add_argument("--base-dir", type=str,
                        default="/tmp/drift-tasks",
                        help="The base directory to keep the files in")

    parser.add_argument("--domain-file", type=str,
                        required=True,
                        help="The base task domain file.")

    parser.add_argument("--problem-file", type=str,
                        required=True,
                        help="The base task problem file.")

    parser.add_argument("--clean", action="store_true",
                        default=False,
                        help="Clean the directory")

    parser.add_argument("--total-tasks", type=int,
                        default=5,
                        help="The total tasks to generate")

    import time
    seed = int(time.time())
    print("Using seed", seed)
    random.seed(seed)

    args = parser.parse_args()

    drift_generator = DriftGenerator()
    drift_generator.generate_drift_tasks(args.base_dir,
                                         args.domain_file,
                                         args.problem_file,
                                         args.total_tasks,
                                         clean=args.clean)

    # Some good seeds
    # 1700183853, 1700266728, 1700277638

    # Unable to find solutions for
    # 1700261501, 1700266676

    # rtg = RandomTaskGenerator()
    # rtg.generate_task("/tmp", "t0")