'''
Created on Jan 29, 2020

@author: rkaria
'''


class Result:

    def __init__(self):

        self._result_dict = {}

    def add(self, name, value):

        self._result_dict[name] = value

    def get(self, name):

        return name

    def get_all_results(self):

        return self._result_dict
