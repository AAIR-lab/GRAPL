"""
action.py
==========
Contains data structures for handling concretized actions and _effect.
"""

import random
import re

from generalized_learning.concretized.state import State


class StochasticAction:

    PROBABILITY_REGEX = re.compile(
        "(\w|\W)*_WEIGHT_(?P<w1>\d+)_(?P<w2>\d+)(\w|\W)*")

    @staticmethod
    def extract_name(action):

        assert isinstance(action, NewAction)

        name = str(action)
        name = name.strip()
        name = name.split(" ")

        if "_DETDUP" in name[0]:
            action_name = name[0][: name[0].index("_DETDUP")]
        else:
            action_name = name[0]

        final_name = action_name
        for i in range(1, len(name)):
            final_name += " " + name[i]
        if len(name) == 1 and ")" not in final_name:
            final_name += ")"

        action_name = action_name.strip()
        action_name = action_name.replace("(", "")
        action_name = action_name.replace(")", "")

        params = name[1:]
        for i in range(len(params)):

            params[i] = params[i].strip()
            params[i] = params[i].replace("(", "")
            params[i] = params[i].replace(")", "")

        return final_name, action_name, params

    def __init__(self, name, params, determinized_base_name, cost=1):

        self._name = name
        self._params = params
        self._cost = cost
        self._actions = []
        self._pos_precon_set = None
        self._neg_precon_set = None
        self._determinized_base_name = determinized_base_name

    def __hash__(self):

        return hash(self._name)

    def __eq__(self, other):

        return isinstance(other, self.__class__) \
            and self._name == other._name

    def add_action(self, action):

        assert self._name == StochasticAction.extract_name(action)[0]

        if self._pos_precon_set is None:

            # Create an actual copy of the set for the action.
            self._pos_precon_set = set(action._pos_precon_set)
            self._neg_precon_set = set(action._neg_precon_set)
        else:

            assert self._pos_precon_set == action._pos_precon_set
            assert self._neg_precon_set == action._neg_precon_set

        name = str(action)
        name = name.strip()

        if "_DETDUP" in name:

            regex_match = StochasticAction.PROBABILITY_REGEX.match(name)
            assert regex_match is not None

            w1 = float(regex_match.group("w1"))
            w2 = float(regex_match.group("w2"))

            self._actions.append((action, w1 / w2))
        else:

            self._actions.append((action, 1.0))

        # Change the action's precon set to be a reference to save a bit of
        # memory.
        action._pos_precon_set = self._pos_precon_set
        action._neg_precon_set = self._neg_precon_set

    def finalize(self):

        # The FD parser actually already verifies if the sum of the
        # probabilities equals 1.
        #
        # Furthermore, it also goes aheads and adds NULL actions just in case
        # the probabilities do not sum up to 1.
        #
        # Because of the reasons above, we can just assert and verify that the
        # sum of the probabilities is 1 for a sanity check. Triggering this
        # assert might be because of use of a parser that is not FD provided.

        total_probability = 0.0
        for _, probability in self._actions:

            total_probability += probability

        assert total_probability == 1.0

    def is_applicable(self, state):
        state_atom_set = state.get_atom_set()

        if self._pos_precon_set.issubset(state_atom_set) \
                and len(self._neg_precon_set.intersection(state_atom_set)) == 0:

            return True
        else:

            return False

    def get_successors(self, state):

        successors = []

        for action, probability in self._actions:

            successors.append((action.apply(state), probability))

        return successors

    def apply(self, state):

        # The seed is already set at the start of the program.
        weight = random.random()

        # The stochastic actions probabilities are expected to be sorted when
        # calling get_actions().
        current_weight = 0.0
        for action, probability in self.get_actions():

            current_weight += probability

            if current_weight >= weight:

                assert action.is_applicable(state)
                next_state = action.apply(state)

                return next_state

        assert False
        return None

    def get_all_atoms(self):

        atom_set = set()

        for action, _ in self.get_actions():

            atom_set.update(action.get_all_atoms())

        return atom_set

    def get_cost(self):

        return self._cost

    def __str__(self):

        return self._name

    def get_actions(self):

        return self._actions

    def get_name(self):

        return self._determinized_base_name

    def get_param_list(self):

        return self._params

    def get_num_params(self):

        return len(self._params)


class NewAction:

    def __init__(self, name, cost, pos_precon_set, neg_precon_set,
                 add_ce_dict, del_ce_dict, problem=None):

        name = name.strip()
        name = name.replace("(", "")
        name = name.replace(")", "")
        name = name.split(" ")

        self._name = name[0]
        self._params = name[1:]
        for i in range(len(self._params)):

            self._params[i] = self._params[i].strip()

        self._cost = cost
        self._pos_precon_set = pos_precon_set
        self._neg_precon_set = neg_precon_set
        self._add_ce_dict = add_ce_dict
        self._del_ce_dict = del_ce_dict

        self._problem = problem

        self._cached_string = None
        self._cached_hash = None

    def is_applicable(self, state):

        if self._pos_precon_set <= state.get_atom_set() \
                and self._neg_precon_set & state.get_atom_set() == set():

            return True
        else:

            return False

    def _is_condition_satisfied(self, condition, state_atom_set):

        if condition is None:

            return True
        elif condition.negated:

            return condition not in state_atom_set
        else:

            return condition in state_atom_set

    def apply(self, state):

        state_atom_set = state.get_atom_set()
        new_atom_set = set(state_atom_set)

        for condition in self._add_ce_dict:

            if self._is_condition_satisfied(condition, state_atom_set):

                new_atom_set |= self._add_ce_dict[condition]

        for condition in self._del_ce_dict:

            if self._is_condition_satisfied(condition, state_atom_set):

                new_atom_set -= self._del_ce_dict[condition]

        return State(new_atom_set)

    def get_all_atoms(self):

        atom_set = set()
        atom_set.update(self._pos_precon_set)
        atom_set.update(self._neg_precon_set)

        for condition in self._add_ce_dict:

            if condition is not None:

                atom_set.add(condition)

            atom_set.update(self._add_ce_dict[condition])

        for condition in self._del_ce_dict:

            if condition is not None:

                atom_set.add(condition)

            atom_set.update(self._del_ce_dict[condition])

        return atom_set

    def get_cost(self):

        return self._cost

    def __str__(self):

        if self._cached_string is None:

            self._cached_string = "(%s" % (self._name)

            for param in self._params:

                self._cached_string += " %s" % (param)

            self._cached_string += ")"

        return self._cached_string

    def __hash__(self):

        if self._cached_hash is None:

            self._cached_hash = hash(str(self))

        return self._cached_hash

    def __eq__(self, other):

        return str(self) == str(other)

    def get_name(self):

        return self._name

    def get_param_list(self):

        return self._params

    def get_num_params(self):

        return len(self._params)
