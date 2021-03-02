"""
generalized_learning.concretized.state.py
===================================
Contains classes for representing states in PDDL problems.
"""


class State:

    def __init__(self, atom_set):

        self._atom_set = frozenset(atom_set)
        self._cached_hash = None

    def __str__(self):

        string = "["

        for atom in self._atom_set:

            atom_string = "(%s" % (atom.predicate)
            for arg in atom.args:

                atom_string += " %s" % (arg)
            atom_string += ") "

            string += atom_string

        string = string.strip()
        string += "]"
        return string

    def __eq__(self, other):

        if self is other:

            return True
        elif not isinstance(other, State):

            return False
        else:

            return self._cached_hash == other._cached_hash \
                and self._atom_set == other._atom_set

    def __hash__(self):

        if self._cached_hash is None:

            self._cached_hash = hash(self._atom_set)

        return self._cached_hash

    def get_atom_set(self):

        return self._atom_set

    def get_arity_atom_set_dict(self, arities=None):

        arity_atom_set_dict = {}

        for atom in self._atom_set:

            arity = len(atom.args)
            if arities is None or arity in arities:

                try:
                    arity_atom_set_dict[arity].add(atom)
                except KeyError:

                    arity_atom_set_dict[arity] = set()
                    arity_atom_set_dict[arity].add(atom)

        return arity_atom_set_dict

    def __contains__(self, atom):

        return atom in self._atom_set
