from pddlgym.parser import Operator, PDDLProblemParser, parse_plan_step
from pddlgym.structs import LiteralConjunction, Predicate, State
from ndrs import NOISE_OUTCOME

from collections import defaultdict
import sys
import os
import numpy as np
import re
import subprocess
import time
import abc
import random


class PlannerTimeoutException(Exception):
    pass


class NoPlanFoundException(Exception):
    pass



def find_policy(planner_name, ndr_operators, action_space, observation_space):
    if planner_name == "ff_replan":
        return find_ff_replan_policy(ndr_operators, action_space, observation_space)
    raise Exception("Unknown planner `{}`".format(planner_name))

def find_ff_replan_policy(ndr_operators, action_space, observation_space):
    # First pull out the most likely effect per NDR to form
    # deterministic operators
    deterministic_operators = []
    for ndr_list in ndr_operators.values():
        for i, ndr in enumerate(ndr_list):
            op_name = "{}{}".format(ndr.action.predicate.name, i)
            probs, effs = ndr.effect_probs, ndr.effects
            max_idx = np.argmax(probs)
            max_effects = LiteralConjunction(sorted(effs[max_idx]))
            if len(max_effects.literals) == 0 or NOISE_OUTCOME in max_effects.literals:
                continue
            preconds = LiteralConjunction(sorted(ndr.preconditions) + [ndr.action])
            params = sorted({ v for lit in preconds.literals for v in lit.variables })
            operator = Operator(op_name, params, preconds, max_effects)
            deterministic_operators.append(operator)

    domain_name = "mydomain"
    planner = FastForwardPlanner(deterministic_operators, domain_name, action_space, observation_space)

    # Only replan if outcome is not expected
    expected_next_state = None
    plan = []

    def get_next_expected_state(state, action):
        return ndr_operators[action.predicate].predict_max(state, action)

    # Given a state, replan and execute first section in plan
    def policy(obs):
        nonlocal expected_next_state
        nonlocal plan

        state = obs.literals
        goal = obs.goal
        objects = obs.objects

        if False: #len(plan) > 0:
            expected_next_state = get_next_expected_state(state, plan[0])
            return plan.pop(0)

        # Add possible actions to state
        full_state = set(state)
        full_state.update(action_space.all_ground_literals(obs))

        # Create problem file
        fname = '/tmp/problem.pddl'
        PDDLProblemParser.create_pddl_file(fname, objects, full_state, "myproblem", domain_name, goal)
        # Get plan
        print("goal:", goal)
        try:
            plan = planner.get_plan(fname, use_cache=False)
            print("plan:", plan)
        except NoPlanFoundException:
            # Default to random
            print("no plan found")
            return action_space.sample(obs)
        # Updated expected next state
        expected_next_state = get_next_expected_state(state, plan[0])
        return plan.pop(0)

    return policy


class Planner:
    def __init__(self, learned_operators, domain_name, action_space, observation_space):
        self._learned_operators = learned_operators
        self.domain_name = domain_name
        self._action_space = action_space
        self._predicates = {p.name : p \
            for p in observation_space.predicates + action_space.predicates}
        self._types = {str(t) : t for p in self._predicates.values() for t in p.var_types}
        self._actions = {p.name for p in action_space.predicates}
        self._problem_files = {}

    @abc.abstractmethod
    def get_plan(self, raw_problem_fname):
        pass

    def _create_domain_file(self):
        dom_str = self._create_domain_file_header()
        dom_str += self._create_domain_file_types()
        dom_str += self._create_domain_file_predicates()

        for operator in sorted(self._learned_operators, key=lambda o:o.name):
            dom_str += self._create_domain_file_operator(operator)
        dom_str += '\n)'

        return self._create_domain_file_from_str(dom_str)

    def _create_domain_file_header(self):
        return """(define (domain {})\n\t(:requirements :strips :typing)\n""".format(
            self.domain_name.lower())

    def _create_domain_file_types(self):
        return "\t(:types " + " ".join(self._types.values()) + ")\n"

    def _create_domain_file_predicates(self):
        preds_pddl = []
        for pred in self._predicates.values():
            var_part = []
            for i, var_type in enumerate(pred.var_types):
                var_part.append("?arg{} - {}".format(i, var_type))
            preds_pddl.append("\t\t({} {})".format(pred.name, " ".join(var_part)))
        preds_pddl.append("\t(Different ?arg0 ?arg1)")
        return """\t(:predicates\n{}\n\t)""".format("\n".join(preds_pddl))

    def _create_domain_file_operator(self, operator):
        param_strs = [str(param).replace(":", " - ") for param in operator.params]
        dom_str = "\n\n\t(:action {}".format(operator.name)
        dom_str += "\n\t\t:parameters ({})".format(" ".join(param_strs))
        preconds_pddl_str = self._create_preconds_pddl_str(operator.preconds)
        dom_str += "\n\t\t:precondition (and {})".format(preconds_pddl_str)
        indented_effs = operator.effects.pddl_str().replace("\n", "\n\t\t")
        dom_str += "\n\t\t:effect {}".format(indented_effs)
        dom_str += "\n\t)"
        return dom_str

    def _create_preconds_pddl_str(self, preconds):
        all_params = set()
        precond_strs = []
        for term in preconds.literals:
            params = set(map(str, term.variables))
            if term.negated_as_failure:
                # Negative term. The variables to universally
                # quantify over are those which we have not
                # encountered yet in this clause.
                universally_quantified_vars = list(sorted(
                    params-all_params))
                precond = ""
                for var in universally_quantified_vars:
                    precond += "(forall ({}) ".format(
                        var.replace(":", " - "))
                precond += "(or "
                for var in universally_quantified_vars:
                    var_cleaned = var[:var.find(":")]
                    for param in list(sorted(all_params)):
                        param_cleaned = param[:param.find(":")]
                        precond += "(not (Different {} {})) ".format(
                            param_cleaned, var_cleaned)
                precond += "(not {}))".format(term.positive.pddl_str())
                for var in universally_quantified_vars:
                    precond += ")"
                precond_strs.append(precond)
            else:
                # Positive term.
                all_params.update(params)
                precond_strs.append(term.pddl_str())

        all_params = list(sorted(all_params))
        for param1 in all_params:
            param1_cleaned = param1[:param1.find(":")]
            for param2 in all_params:
                if param1 >= param2:
                    continue
                param2_cleaned = param2[:param2.find(":")]
                precond_strs.append("(Different {} {})".format(
                    param1_cleaned, param2_cleaned))

        return "\n\t\t\t".join(precond_strs)

    def _create_domain_file_from_str(self, dom_str):
        filename = "/tmp/learned_dom_{}_{}.pddl".format(
            self.domain_name, random.randint(0, 9999999))
        with open(filename, 'w') as f:
            f.write(dom_str)
        return filename

    def _create_problem_file(self, raw_problem_fname, use_cache=True):
        if (not use_cache) or (raw_problem_fname not in self._problem_files):
            problem_fname = os.path.split(raw_problem_fname)[-1]
            problem_fname = problem_fname.split('.pddl')[0]
            problem_fname += '_{}_with_diffs_{}.pddl'.format(
                self.domain_name, random.randint(0, 9999999))
            problem_fname = os.path.join('/tmp', problem_fname)

            # Parse raw problem
            problem_parser = PDDLProblemParser(raw_problem_fname,
                self.domain_name.lower(), self._types, self._predicates, self._actions)
            new_initial_state = set(problem_parser.initial_state)

            # Add actions
            action_lits = set(self._action_space.all_ground_literals(
                State(new_initial_state, problem_parser.objects, problem_parser.goal),
                valid_only=False))
            new_initial_state |= action_lits

            # Add 'Different' pairs for each pair of objects
            Different = Predicate('Different', 2)

            for obj1 in problem_parser.objects:
                for obj2 in problem_parser.objects:
                    if obj1 == obj2:
                        continue
                    # if obj1.var_type != obj2.var_type:
                    #     continue
                    diff_lit = Different(obj1, obj2)
                    new_initial_state.add(diff_lit)

            # Write out new temporary problem file
            problem_parser.initial_state = frozenset(new_initial_state)
            problem_parser.write(problem_fname)

            # Add to cache
            self._problem_files[raw_problem_fname] = problem_fname

        return self._problem_files[raw_problem_fname]


class FastForwardPlanner(Planner):
    FF_PATH = os.environ['FF_PATH']
    planner_timeout = 10

    def get_plan(self, raw_problem_fname, use_cache=True):
        # If there are no operators yet, we're not going to be able to find a plan
        if not self._learned_operators:
            raise NoPlanFoundException()
        domain_fname = self._create_domain_file()
        problem_fname = self._create_problem_file(raw_problem_fname, use_cache=use_cache)
        cmd_str = self._get_cmd_str(domain_fname, problem_fname)
        start_time = time.time()
        output = subprocess.getoutput(cmd_str)
        end_time = time.time()
        os.remove(domain_fname)
        if not use_cache:
            os.remove(problem_fname)
        if end_time - start_time > 0.9*self.planner_timeout:
            raise PlannerTimeoutException()
        plan = self._output_to_plan(output)
        return self._plan_to_actions(plan)

    def _get_cmd_str(self, domain_fname, problem_fname):
        timeout = "gtimeout" if sys.platform == "darwin" else "timeout"
        return "{} {} {} -o {} -f {}".format(
            timeout, self.planner_timeout, self.FF_PATH,
            domain_fname, problem_fname)

    @staticmethod
    def _output_to_plan(output):
        if not output.strip() or \
           "goal can be simplified to FALSE" in output or \
            "unsolvable" in output or "increase MAX_VARS" in output:
            raise NoPlanFoundException()
        plan = re.findall(r"\d+?: (.+)", output.lower())
        if not plan and "found legal" not in output and \
           "The empty plan solves it" not in output:
            raise Exception("Plan not found with FF! Error: {}".format(output))
        return plan

    def _plan_to_actions(self, plan):
        operators = self._learned_operators
        action_predicates = self._action_space.predicates
        objects = self._action_space._objects

        actions = []
        for plan_step in plan:
            if plan_step == "reach-goal":
                continue
            action = parse_plan_step(plan_step, operators, action_predicates, objects)
            actions.append(action)
        return actions
