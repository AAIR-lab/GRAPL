'''
Created on Feb 14, 2020

@author: rkaria
'''

import itertools


def flatten(list_of_lists):

    # https://docs.python.org/2/library/itertools.html
    return itertools.chain.from_iterable(list_of_lists)
