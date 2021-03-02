"""
action.py
==========
Contains data structures for handling concretized actions and _effect.
"""


from concretized.state import State


class NewAction:

    def __init__(self, name, cost, pos_precon_set, neg_precon_set,
                 add_effect_set, del_effect_set, problem=None):

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
        self._add_effect_set = add_effect_set
        self._del_effect_set = del_effect_set

        self._problem = problem

    def is_applicable(self, state):

        if self._pos_precon_set <= state.get_atom_set() \
                and self._neg_precon_set & state.get_atom_set() == set():

            return True
        else:

            return False

    def apply(self, state):

        new_atom_set = set(state.get_atom_set())

        new_atom_set |= self._add_effect_set
        new_atom_set -= self._del_effect_set

        if self._problem is not None:

            return self._problem.tag_goals(new_atom_set)
        else:

            # Deprecated for now since this is only used in abstraction.
            # If non-abstraction based methods requires this, then we can
            # change this.
            assert False
            return State(new_atom_set)

    def get_cost(self):

        return self._cost

    def __str__(self):

        string = "(%s" % (self._name)

        for param in self._params:

            string += " %s" % (param)

        string += ")"
        return string

    def get_name(self):

        return self._name

    def get_param_list(self):

        return self._params

    def get_num_params(self):

        return len(self._params)
