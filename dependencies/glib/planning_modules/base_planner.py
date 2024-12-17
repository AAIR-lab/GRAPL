from pddlgym.structs import Predicate, Exists, State
from pddlgym.parser import PDDLProblemParser

import random
import abc
import os


class PlannerTimeoutException(Exception):
    pass


class NoPlanFoundException(Exception):
    pass


class Planner:
    def __init__(self, learned_operators, domain_name, action_space, observation_space):
        self._learned_operators = learned_operators
        self.domain_name = domain_name
        self._action_space = action_space
        self._observation_space = observation_space
        self._predicates = {p.name : p \
            for p in observation_space.predicates + action_space.predicates}
        self._types = {str(t) : t for p in self._predicates.values() for t in p.var_types}
        self._problem_files = {}

    @abc.abstractmethod
    def get_policy(self, raw_problem_fname):
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
            action_names = []  # purposely empty b/c we WANT action literals in there
            problem_parser = PDDLProblemParser(
                raw_problem_fname, self.domain_name.lower(), self._types,
                self._predicates, action_names)

            # Add action literals (in case they're not already present in the initial state)
            # which will be true when the original domain uses operators_as_actions
            init_state = State(problem_parser.initial_state, problem_parser.objects, None)
            act_lits = self._action_space.all_ground_literals(init_state, valid_only=False)
            problem_parser.initial_state = frozenset(act_lits | problem_parser.initial_state)

            # Add 'Different' pairs for each pair of objects
            Different = Predicate('Different', 2)
            init_state = set(problem_parser.initial_state)
            for obj1 in problem_parser.objects:
                for obj2 in problem_parser.objects:
                    if obj1 == obj2:
                        continue
                    # if obj1.var_type != obj2.var_type:
                    #     continue
                    diff_lit = Different(obj1, obj2)
                    init_state.add(diff_lit)
            problem_parser.initial_state = frozenset(init_state)
            # Also add 'different' pairs for goal if it's existential
            if isinstance(problem_parser.goal, Exists):
                diffs = []
                for var1 in problem_parser.goal.variables:
                    for var2 in problem_parser.goal.variables:
                        if var1 == var2:
                            continue
                        diffs.append(Different(var1, var2))
                problem_parser.goal = Exists(
                    problem_parser.goal.variables,
                    type(problem_parser.goal.body)(
                        problem_parser.goal.body.literals+diffs))

            # If no objects, write a dummy one to make FF not crash.
            if not problem_parser.objects:
                problem_parser.objects.append("DUMMY")

            # Write out new temporary problem file
            problem_parser.write(problem_fname)

            # Add to cache
            self._problem_files[raw_problem_fname] = (problem_fname, problem_parser.objects)

        return self._problem_files[raw_problem_fname]
