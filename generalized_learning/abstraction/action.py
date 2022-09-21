'''
Created on Jun 7, 2021

@author: anonymous
'''


class AbstractAction:

    @staticmethod
    def create(abstract_state, action):

        name = action.get_name()
        params = []

        for param in action.get_param_list():

            params.append(abstract_state.get_role(param))

        return AbstractAction(name, params)

    def __init__(self, name, params):

        self._name = name
        self._params = params

        self._cached_hash = None

    def __hash__(self):

        if self._cached_hash is None:

            self._cached_hash = hash(self._name)
            for param in self._params:

                self._cached_hash += hash(param)

        return self._cached_hash

    def __eq__(self, other):

        return self._name == other._name \
            and self._params == other._params

    def __str__(self):

        string = "(%s" % (self._name)

        for param in self._params:

            string += " %s" % (str(param))

        string += ")"
        return string
