#!/usr/bin/env python3


class Node:

    def __init__(self, concrete_state, parent, action, depth, fscore=0.0):

        self._concrete_state = concrete_state
        self._parent = parent
        self._action = action
        self._depth = depth
        self._fscore = fscore
        self.h = 0

    def __eq__(self, other):

        if self is other:

            return True
        else:

            return self._parent is other._parent \
                and self._depth == other._depth \
                and self._concrete_state == other._concrete_state

    def get_h(self):

        return self.h

    def get_concrete_state(self):

        return self._concrete_state

    def get_parent(self):

        return self._parent

    def get_action(self):

        return self._action

    def get_depth(self):

        return self._depth

    def get_fscore(self):

        return self._fscore
