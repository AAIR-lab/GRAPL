'''
Created on Apr 1, 2020

@author: rkaria
'''


class Series:

    def __init__(self):

        self._x = {}

    def __len__(self):

        return len(self._x)

    def get_x(self):

        return self._x

    def add_point(self, x, y):

        try:

            _y = self._x[x]
        except KeyError:

            _y = []
            self._x[x] = _y

        _y.append(y)

    def get_y(self, x):

        return self._x[x]
