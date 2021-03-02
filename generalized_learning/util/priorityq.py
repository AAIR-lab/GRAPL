'''
Created on Feb 25, 2020

@author: rkaria
'''

import heapq
import itertools

# Very much influenced by:-
# https://docs.python.org/3/library/heapq.html?highlight=heapq


class PriorityQ:

    _REMOVED_TOKEN = 0xDEADC0DE

    def __init__(self):

        self._pq = []
        self._entry_counter = itertools.count()
        self._entry_dict = {}

    def push(self, entry, priority):

        if entry in self._entry_dict:

            old_entry_list = self._entry_dict[entry]
            old_entry_list[-1] = PriorityQ._REMOVED_TOKEN

        entry_list = [priority, next(self._entry_counter), entry]
        self._entry_dict[entry] = entry_list

        heapq.heappush(self._pq, entry_list)

    def is_empty(self):

        return not self._pq

    def pop(self):

        while self._pq:

            _, _, entry = heapq.heappop(self._pq)
            if entry != PriorityQ._REMOVED_TOKEN:

                del self._entry_dict[entry]
                return entry

        return IndexError("index out of range")
