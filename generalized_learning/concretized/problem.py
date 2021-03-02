

import pathlib

from concretized.action import NewAction
from concretized.state import State
from pddl import conditions
from translate import instantiate
from translate import normalize
from translate import pddl_parser
import translate
from util import constants
from util import file


class Problem:

    @staticmethod
    def _extract_task(domain_file, problem_file, directory):

        # Extract the domain specific args.
        domain_filepath = "%s/%s" % (directory, domain_file)
        domain_pddl = pddl_parser.pddl_file.parse_pddl_file(
            "domain", domain_filepath)
        domain_name, \
            domain_requirements, \
            types, \
            type_dict, \
            constants, \
            predicates, \
            predicate_dict, \
            functions, \
            actions, \
            axioms = pddl_parser.parsing_functions.parse_domain_pddl(
                domain_pddl)

        # Heavily influenced from pddl_parser.parsing_functions.parse_task().
        # There are a couple changes like removing all equality related
        # attributes.
        problem_filepath = "%s/%s" % (directory, problem_file)
        task_pddl = pddl_parser.pddl_file.parse_pddl_file(
            "task", problem_filepath)
        task_name, \
            task_domain_name, \
            task_requirements, \
            objects, \
            init, \
            goal, \
            use_metric = pddl_parser.parsing_functions.parse_task_pddl(
                task_pddl, type_dict, predicate_dict)

        assert domain_name == task_domain_name
        requirements = translate.pddl.Requirements(sorted(set(
            domain_requirements.requirements +
            task_requirements.requirements)))
        objects = constants + objects
        pddl_parser.parsing_functions.check_for_duplicates(
            [o.name for o in objects],
            errmsg="error: duplicate object %r",
            finalmsg="please check :constants and :objects definitions")
        init += [translate.pddl.Atom("=", (obj.name, obj.name))
                 for obj in objects]

        task = translate.pddl.Task(domain_name,
                                   task_name,
                                   requirements,
                                   types,
                                   objects,
                                   predicates,
                                   functions,
                                   init,
                                   goal,
                                   actions,
                                   axioms,
                                   use_metric)

        return task

    def __init__(self, domain_file, problem_file, directory="./",
                 unsupported_predicates=set(["="])):

        self._domain_file = domain_file
        self._problem_file = problem_file
        self._directory = directory
        self._task = Problem._extract_task(
            domain_file, problem_file, directory)

        problem_filepath = pathlib.Path(self._directory, self._problem_file)
        self._properties = file.read_properties(problem_filepath)
        if self._properties is None:

            self._properties = {}

        # Append the properties we get from the task.
        self._properties["name"] = self._task.task_name
        self._properties["filepath"] = str(problem_filepath)
        self._properties["domain"] = self._task.domain_name

        # TODO: Add an assert to make sure that the goal is just a conjunction
        # operator.
        if len(self._task.goal.parts) == 0:

            self._goal_literals = [self._task.goal]
        else:

            self._goal_literals = self._task.goal.parts
        assert len(self._goal_literals) > 0

        # Normalize the task.
        normalize.normalize(self._task)

        # Instantiate the task to get the set of possible actions.
        relaxed_reachable, atoms, actions, axioms, reachable_action_params = \
            instantiate.explore(self._task)

        self._relaxed_reachable = relaxed_reachable
        if not relaxed_reachable:

            self._relaxed_reachable = relaxed_reachable
            pass
#             raise Exception("Problem=%s does not admit a relaxed solution." % (
#                 self.get_problem_filepath()))

        self._atoms = atoms
        self._actions = actions
        self._axioms = axioms
        self._reachable_action_params = reachable_action_params
        self._is_relaxed_reachable = relaxed_reachable
        self._action_map = self._extract_action_map(self._actions)

        self._initial_state, self._auxiliary_constant_atoms = \
            self._extract_initial_state(self._task, unsupported_predicates,
                                        self._goal_literals)

        self._initial_state = self.tag_goals(self._initial_state)

    def is_relaxed_reachable(self):

        return self._is_relaxed_reachable

    def get_properties(self):

        return self._properties

    def get_problem_filepath(self):

        return "%s/%s" % (self._directory, self._problem_file)

    def get_domain_filepath(self):

        return "%s/%s" % (self._directory, self._domain_file)

    def _extract_initial_state(self, task, unsupported_predicates,
                               goal_literals):

        # Represents an atom set that will never change and will always
        # be present in all states.
        auxiliary_constant_atoms = set()

        atom_set = set()
        for atom in task.init:

            if not atom.negated \
                    and atom.predicate not in unsupported_predicates:

                atom_set.add(atom)

        # Add in types.
        if constants.TAG_TYPES:
            for task_object in task.objects:

                if task_object.type_name is not None:

                    typed_atom = conditions.Atom(
                        "type_%s" % (task_object.type_name),
                        (task_object.name, ))
                    atom_set.add(typed_atom)
                    auxiliary_constant_atoms.add(typed_atom)

        # Create the goal conditions.
        for goal_literal in goal_literals:

            assert not goal_literal.negated
            goal_predicate_name = "goal_%s" % (goal_literal.predicate)
            goal_atom = conditions.Atom(goal_predicate_name, goal_literal.args)

            if len(goal_literal.args) <= 1 and constants.TAG_UNARY_GOALS:

                atom_set.add(goal_atom)
                auxiliary_constant_atoms.add(goal_atom)
            elif len(goal_literal.args) == 2 and constants.TAG_BINARY_GOALS:

                atom_set.add(goal_atom)
                auxiliary_constant_atoms.add(goal_atom)

                goal_predicate_name = "bin_goal_%s" % (goal_literal.predicate)
                goal_atom = conditions.Atom(goal_predicate_name,
                                            (goal_literal.args[0], ))
                atom_set.add(goal_atom)
                auxiliary_constant_atoms.add(goal_atom)

                required_predicate_name = "req_goal_%s" % (
                    goal_literal.predicate)
                req_atom = conditions.Atom(required_predicate_name,
                                           (goal_literal.args[1], ))
                atom_set.add(req_atom)
                auxiliary_constant_atoms.add(req_atom)

        return State(atom_set), auxiliary_constant_atoms

    def tag_goals(self, state):

        if isinstance(state, set):

            state_atom_set = state
        else:

            state_atom_set = set(state.get_atom_set())

        removed_bin_goals = set()
        removed_req_goals = set()

        for goal_literal in self._goal_literals:

            assert not goal_literal.negated
            goal_predicate_name = "done_goal_%s" % (goal_literal.predicate)
            goal_atom = conditions.Atom(goal_predicate_name, goal_literal.args)

            if len(goal_literal.args) <= 1 and not constants.TAG_UNARY_GOALS:

                continue
            elif len(goal_literal.args) == 2 and not constants.TAG_BINARY_GOALS:

                continue

            if goal_literal in state_atom_set:

                state_atom_set.add(goal_atom)
            else:

                try:

                    state_atom_set.remove(goal_atom)
                except KeyError:

                    pass

            if len(goal_literal.args) == 2:

                goal_predicate_name = "done_bin_goal_%s" % (
                    goal_literal.predicate)
                goal_atom = conditions.Atom(goal_predicate_name,
                                            (goal_literal.args[0], ))

                if goal_atom not in removed_bin_goals:

                    if goal_literal in state:

                        state_atom_set.add(goal_atom)
                    else:

                        removed_bin_goals.add(goal_atom)

                        try:

                            state_atom_set.remove(goal_atom)
                        except KeyError:

                            pass

                required_predicate_name = "done_req_goal_%s" % (
                    goal_literal.predicate)
                req_atom = conditions.Atom(required_predicate_name,
                                           (goal_literal.args[1], ))

                if req_atom not in removed_req_goals:

                    if goal_literal in state:

                        state_atom_set.add(req_atom)
                    else:

                        removed_bin_goals.add(req_atom)

                        try:

                            state_atom_set.remove(req_atom)
                        except KeyError:

                            pass

        return State(state_atom_set)

    def _extract_effect_atoms(self, effects):

        effect_set = set()

        for effect in effects:

            assert effect[0] == []
            effect_set.add(effect[1])

        return effect_set

    def get_applicable_actions(self, state):

        applicable_action_list = []

        for action_name in self._action_map:

            for action in self._action_map[action_name]:

                if action.is_applicable(state):

                    applicable_action_list.append(action)

        return applicable_action_list

    def _extract_action_map(self, actions):

        action_map = {}

        for action in actions:

            name = action.name

            try:

                action_set = action_map[name]

                # Disjunctive actions can have more than one instantiation for
                # the same action.
                #
                # Currently, we do not support the same.
                raise Exception(
                    "Multiple definitions for the same action not supported!")
            except KeyError:

                action_set = set()
                action_map[name] = action_set

            # Get the effect sets.
            add_effect_set = self._extract_effect_atoms(action.add_effects)
            del_effect_set = self._extract_effect_atoms(action.del_effects)

            # Get the preconditions.
            pos_precon_set = set()
            neg_precon_set = set()

            for precon in action.precondition:

                assert precon.parts == []

                if precon.negated:

                    neg_precon_set.add(precon.negate())
                else:

                    pos_precon_set.add(precon)

            new_action = NewAction(action.name, action.cost, pos_precon_set,
                                   neg_precon_set, add_effect_set,
                                   del_effect_set, self)
            action_set.add(new_action)

        return action_map

    def get_typed_objects(self):

        return self._task.objects

    def get_domain(self):

        return self._abstract_domain

    def get_action(self, action_name):

        action_set = self._action_map[action_name]
        assert len(action_set) == 1

        return next(iter(action_set))

    def get_initial_state(self):

        return self._initial_state

    def is_goal_satisfied(self, state):

        for goal_literal in self._goal_literals:

            if not goal_literal.negated and goal_literal not in state \
                    or goal_literal.negated and goal_literal in state:

                return False

        return True

    def encode_pyperplan_initial_state(self):

        # Re-encode pyperplans init state with our atoms.
        # This gives pyperplans initial state our instantiation from fast
        # downward which contains more information.
        atom_set = set()

        for atom in self.get_initial_state().get_atom_set():

            new_atom = "(" + atom.predicate

            for i in range(len(atom.args)):

                new_atom += " %s" % (atom.args[i])

            new_atom += ")"
            atom_set.add(new_atom)

        return atom_set

    def encode_pyperplan_state(self, state):

        atom_set = set()
        for atom in state:

            atom = atom.replace("(", "")
            atom = atom.replace(")", "")
            atom = atom.split(" ")
            name = atom[0]
            args = tuple(atom[1:])

            atom_set.add(conditions.Atom(name, args))

        state = self.tag_goals(atom_set)
        return state
