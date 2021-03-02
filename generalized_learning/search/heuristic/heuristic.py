'''
Created on Feb 12, 2020

@author: rkaria
'''

from abc import ABC
from abc import abstractmethod


class Heuristic(ABC):

    @abstractmethod
    def __init__(self, name, problem):

        self._name = name
        self._problem = problem

    @abstractmethod
    def expand(self, parent):

        raise NotImplementedError

    @abstractmethod
    def get_properties(self):

        raise NotImplementedError
