

import pathlib
import time

from generalized_learning import simulator
from generalized_learning.concretized.action import NewAction
from generalized_learning.concretized.action import StochasticAction
from generalized_learning.concretized.state import State
from generalized_learning.util import rddl
import instantiate
import normalize
from pddl import conditions
import pddl
import translate
from util import constants
from util import file


def create_problem(domain_file, problem_file, directory,
                   simulator_type="generic"):

    problem_filepath = pathlib.Path(directory, problem_file)
    properties = file.read_properties(problem_filepath)

    if properties is None or "rddl" not in properties:

        problem = Problem(domain_file, problem_file, directory, simulator_type)
    else:

        assert properties["rddl"] == True
        problem = RDDLProblem(domain_file, problem_file, directory,
                              simulator_type)

    return problem


class Problem:

    DONE_ATOM = conditions.Atom("horizon_exceeded", ())

    @staticmethod
    def create_timestep_atom(timestep):

        return conditions.Atom("timestep", (timestep,))

    @staticmethod
    def _extract_task(domain_file, problem_file, directory):

        # Extract the domain specific args.
        domain_filepath = "%s/%s" % (directory, domain_file)
        domain_pddl = pddl.pddl_file.parse_pddl_file(
            "domain", domain_filepath)
        domain_name, \
            domain_requirements, \
            types, \
            constants, \
            predicates, \
            functions, \
            actions, \
            axioms = pddl.tasks.parse_domain(
                domain_pddl)

        # Heavily influenced from pddl_parser.parsing_functions.parse_task().
        # There are a couple changes like removing all equality related
        # attributes.
        problem_filepath = "%s/%s" % (directory, problem_file)
        task_pddl = pddl.pddl_file.parse_pddl_file(
            "task", problem_filepath)
        task_name, \
            task_domain_name, \
            task_requirements, \
            objects, \
            init, \
            goal, \
            use_metric = pddl.tasks.parse_task(
                task_pddl)

        assert domain_name == task_domain_name
        requirements = translate.pddl.Requirements(sorted(set(
            domain_requirements.requirements +
            task_requirements.requirements)))
        objects = constants + objects
        pddl.tasks.check_for_duplicates(
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
                 simulator_type="generic"):

        self._domain_file = domain_file
        self._problem_file = problem_file
        self._directory = directory
        self._task = Problem._extract_task(
            domain_file, problem_file, directory)

        self._domain_actions, self._max_domain_action_params = \
            self.get_domain_action_info(self._task)

        problem_filepath = pathlib.Path(self._directory, self._problem_file)
        self._properties = file.read_properties(problem_filepath)
        if self._properties is None:

            self._properties = {}

        # Append the properties we get from the task.
        self._properties["name"] = self._task.task_name
        self._properties["filepath"] = str(problem_filepath)
        self._properties["domain"] = self._task.domain_name

        if len(self._task.goal.parts) == 0:

            if isinstance(self._task.goal, pddl.conditions.Conjunction):

                assert self._properties["rddl"] == True

                # Allow for empty (:goal (and)) conditions for RDDL instances.
                self._goal_literals = []
            else:

                # For PDDL goals of the form (:goal (predicate ...))
                assert not isinstance(self._task.goal,
                                      pddl.conditions.Conjunction)
                self._goal_literals = [self._task.goal]
        else:

            self._goal_literals = self._task.goal.parts

        # Normalize the task.
        normalize.normalize(self._task)

        # Instantiate the task to get the set of possible actions.
        relaxed_reachable, atoms, actions, axioms, reachable_action_params = \
            instantiate.explore(self._task)

        self._ssipp_map = {}
        self._relaxed_reachable = relaxed_reachable
        self._atoms = atoms
        self._actions = actions
        self._axioms = axioms
        self._reachable_action_params = reachable_action_params
        self._is_relaxed_reachable = relaxed_reachable
        self._action_map, self._stochastic_action_map, self._stochastic_actions_dict = \
            self._extract_action_map(self._actions)

        self._initial_state = self._extract_initial_state(
            self._task)

        self._sim = simulator.get_simulator(simulator_type)

    def add_done_atom(self, state):

        state_atom_set = set(state.get_atom_set())
        state_atom_set.add(Problem.DONE_ATOM)
        return State(state_atom_set)

    def add_timestep_atom(self, state, timestep):

        state_atom_set = set(state.get_atom_set())

        previous_timestep_atom = Problem.create_timestep_atom(timestep - 1)
        timestep_atom = Problem.create_timestep_atom(timestep)

        state_atom_set.discard(previous_timestep_atom)
        state_atom_set.add(timestep_atom)

        return State(state_atom_set)

    def redo_action_maps(self, actions):

        # For SSiPP
        self._action_map, self._stochastic_action_map, self._stochastic_actions_dict = \
            self._extract_action_map(actions, [])

    def has_goals(self):

        return len(self._goal_literals) > 0

    def is_relaxed_reachable(self):

        return self._is_relaxed_reachable

    def is_trivial(self):

        return self.is_goal_satisfied(self.get_initial_state())

    def requires_planning(self):

        return self.is_relaxed_reachable() and not self.is_trivial()

    def get_properties(self):

        return self._properties

    def get_problem_filepath(self):

        return "%s/%s" % (self._directory, self._problem_file)

    def get_domain_filepath(self):

        return "%s/%s" % (self._directory, self._domain_file)

    def get_domain_action_info(self, task):

        domain_actions = set()
        max_action_params = float("-inf")
        for i in range(len(task.actions)):

            domain_actions.add(task.actions[i].name)
            max_action_params = max(max_action_params,
                                    task.actions[i].num_external_parameters)

        return domain_actions, max_action_params

    def get_num_actions_in_domain(self):

        return len(self._domain_actions)

    def get_actions_in_domain(self):

        return self._domain_actions

    def get_max_action_params_in_domain(self):

        return self._max_domain_action_params

    def _extract_initial_state(self, task):

        atom_set = set()
        for atom in task.init:

            assert not atom.negated
            atom_set.add(atom)

        return State(atom_set)

    def _extract_condition_effect_dict(self, effects):

        condition_effect_dict = {}

        for condition, effect in effects:

            assert len(condition) <= 1
            if condition == []:

                condition = None
            else:

                condition = condition[0]
                assert isinstance(condition, (translate.pddl.Atom,
                                              translate.pddl.NegatedAtom))

                # The handling for this is already present in NewAction, but
                # if you trip it, its best if you check if NewAction works as
                # intended.
                #
                # This assert can be removed after that.
                assert not condition.negated

            assert isinstance(effect, (translate.pddl.Atom,
                                       translate.pddl.NegatedAtom))
            condition_set = condition_effect_dict.setdefault(condition, set())
            condition_set.add(effect)

        return condition_effect_dict

    def get_applicable_grounded_actions(self, state):

        applicable_action_list = []

        for action_name in self._action_map:

            for action in self._action_map[action_name]:

                if action.is_applicable(state):

                    applicable_action_list.append(action)

        return applicable_action_list

    def get_grounded_actions(self):

        grounded_action_list = []

        for action_name in self._action_map:

            action = next(iter(self._action_map[action_name]))
            grounded_action_list.append(action)

        return grounded_action_list

    def get_applicable_actions(self, state):

        applicable_action_list = []

        for action_name in self._stochastic_actions_dict:

            stochastic_action = self._stochastic_actions_dict[action_name]

            if stochastic_action.is_applicable(state):
                applicable_action_list.append(stochastic_action)

        return applicable_action_list

    def apply_action(self, action, state):

        return self._sim.apply_action(self, state, action)

    def get_successors(self, action, state):

        return action.get_successors(state)

    def get_stochastic_actions_dict(self):

        return self._stochastic_actions_dict

    def get_stochastic_action(self, action_name):

        try:

            return self._stochastic_actions_dict[action_name]
        except KeyError:

            return None

    def sort_stochastic_actions(self):

        for action_name in self._stochastic_actions_dict:

            stochastic_action = self._stochastic_actions_dict[action_name]

            actions = stochastic_action.get_actions()
            actions.sort(key=lambda y: y[1], reverse=False)

    def _extract_action_map(self, actions):

        stochastic_actions_dict = {}
        stochastic_action_map = {}
        action_map = {}
        for action in actions:

            name = action.name
            name = name.replace(" )", ")")

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

            # Get the condition-effect dictionaries.
            add_ce_dict = self._extract_condition_effect_dict(
                action.add_effects)
            del_ce_dict = self._extract_condition_effect_dict(
                action.del_effects)

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
                                   neg_precon_set, add_ce_dict,
                                   del_ce_dict, self)
            action_set.add(new_action)

            self._ssipp_map[new_action] = action

            stochastic_name, name, params = StochasticAction.extract_name(
                new_action)
            stochastic_action_set = stochastic_action_map.setdefault(
                name, set())

            stochastic_action = stochastic_actions_dict.setdefault(
                stochastic_name,
                StochasticAction(stochastic_name, params, name))

            stochastic_action_set.add(stochastic_action)
            stochastic_action.add_action(new_action)

        return action_map, stochastic_action_map, stochastic_actions_dict

    def get_typed_objects(self):

        return self._task.objects

    def get_action(self, action_name):

        action_set = self._action_map[action_name]
        assert len(action_set) == 1

        return next(iter(action_set))

    def get_initial_state(self):

        return self._initial_state

    def get_goal_literals(self):

        return self._goal_literals

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

        return atom_set


class RDDLProblem(Problem):

    def __init__(self, domain_file, problem_file, directory="./",
                 simulator_type="rddl"):

        assert simulator_type == "rddl"

        super(RDDLProblem, self).__init__(domain_file, problem_file,
                                          directory, simulator_type)

        # We have to reset some variables that only exist in the PPDDL realm
        # and do not really matter in the RDDL realm.
        self._relaxed_reachable = True
        self._is_relaxed_reachable = True

        # Keep these None for now for testing. Later we can optimize them
        # so that the overhead of converting to StochasticAction() is
        # avoided.
        self._ssipp_map = None
        self._atoms = None
        self._actions = None
        self._axioms = None
        self._reachable_action_params = None
        self._action_map = None
        self._stochastic_action_map = None
        self._stochastic_actions_dict = None
        self._initial_state = None

        self._sim = rddl.get_simulator(self._task.domain_name,
                                       self._task.task_name,
                                       directory,
                                       seed=int(time.time()))
        rddl_non_fluents = self._sim.py_get_non_fluents()

        self._non_fluents = set()
        for rddl_non_fluent in rddl_non_fluents._alNonFluents:

            if rddl_non_fluent._oValue == True:

                predicate = rddl.convert_rddl_predicate(rddl_non_fluent)
                self._non_fluents.add(predicate)

        self._applicable_actions = None

    def is_trivial(self):

        return False

    def requires_planning(self):

        return True

    def get_initial_state(self):

        rddl_state = self._sim.py_get_initial_state()

        state = rddl.convert_rddl_state(rddl_state, self._non_fluents)
        return state

    def get_applicable_actions(self, _state):

        self._applicable_actions = {}
        rddl_actions = self._sim.py_get_applicable_actions()

        for action_key in rddl_actions:

            rddl_action_list = rddl_actions[action_key]

            action = rddl.get_rddl_action(rddl_action_list)

            assert action not in self._applicable_actions
            self._applicable_actions[action] = rddl_action_list

        return list(self._applicable_actions.keys())

    def apply_action(self, action, _state):

        assert self._applicable_actions is not None
        assert action in self._applicable_actions

        rddl_action_list = self._applicable_actions[action]
        result = self._sim.py_apply_action(rddl_action_list)

        rddl_next_state = result._next_state
        reward = float(result._reward)
        done = bool(result._done)

        next_state = rddl.convert_rddl_state(rddl_next_state,
                                             self._non_fluents)

        return next_state, reward, done

    def get_successors(self, action, state):

        raise NotImplementedError

    def has_goals(self):

        return self._sim.py_has_goals()

    def is_goal_satisfied(self, _state):

        return self._sim.py_is_goal_satisfied()
