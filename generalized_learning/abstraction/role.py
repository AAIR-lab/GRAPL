"""
role.py
========
Represents an abstract role.
"""

from functools import total_ordering


@total_ordering
class AbstractRole:

    def __init__(self, unary_predicate_names):

        unary_predicate_set = set()
        unary_predicate_set = unary_predicate_set.union(unary_predicate_names)

        self._unary_predicates = frozenset(unary_predicate_set)
        self._cached_string = None

    def __lt__(self, other):

        if not isinstance(other, self.__class__):

            # Raise an error instead of just returning NotImplemented.
            raise NotImplementedError
        else:

            return str(self) < str(other)

    def __contains__(self, element):

        if isinstance(element, str):

            return element in self._unary_predicates
        else:

            # Raise an error instead of just returning NotImplemented.
            raise NotImplementedError

    def __eq__(self, other):

        return isinstance(other, self.__class__) \
            and self._unary_predicates == other._unary_predicates

    def __hash__(self):

        return hash(self._unary_predicates)

    def __str__(self):

        if self._cached_string is not None:

            return self._cached_string
        else:

            # Convert the string manually instead of just using str(sorted_unary_predicates).
            # Using the in-built str() causes un-intuitive sorting since some lists start with
            # double-quotes whereas others begin with single quotes.
            #
            # Also, we ditch square-brackets and use parentheses instead to enfore a shortlex order
            # when one string is a prefix of the other.
            #    [role1, role2, role3] < [role1, role2]
            #    (role1, role2, role3) < (role1, role2)
            sorted_unary_predicates = sorted(self._unary_predicates)
            string = "("
            for unary_predicate in sorted_unary_predicates:

                string += "%s " % (unary_predicate)

            string = string.strip()
            string = string.replace(" ", ", ")
            string += ")"

            self._cached_string = string
            return self._cached_string

    def get_unary_predicates(self):

        return self._unary_predicates
