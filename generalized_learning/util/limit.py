'''
Created on Apr 7, 2020

@author: rkaria
'''

import resource
import signal


class TimeLimit:

    _BUFFER_PERIOD_IN_SEC = 10

    def __init__(self):

        raise NotImplementedError

    @staticmethod
    def get_time_limit_in_sec(time_limit):

        if time_limit == float("inf"):

            return float("inf")
        else:

            time_limit = str(time_limit)
            time_limit = time_limit.lower()

            if time_limit[-1] == "h":

                return float(time_limit[0:-1]) * 3600
            elif time_limit[-1] == "m":

                return float(time_limit[0:-1]) * 60
            elif time_limit[-1] == "s":

                return float(time_limit[0:-1])
            else:

                return float(time_limit)

    @staticmethod
    def timeout(signum, handler):

        raise Exception("Time out!")

    @staticmethod
    def set(time_limit, buffer_period=_BUFFER_PERIOD_IN_SEC):

        time_limit_in_sec = TimeLimit.get_time_limit_in_sec(time_limit)
        if time_limit_in_sec < float("inf"):

            resource.setrlimit(
                resource.RLIMIT_CPU,
                (time_limit_in_sec,
                 time_limit_in_sec + buffer_period))

            signal.signal(signal.SIGXCPU, TimeLimit.timeout)


class MemoryLimit:

    def __init__(self):

        raise NotImplementedError

    @staticmethod
    def get_memory_limit_in_b(memory_limit):

        if memory_limit == float("inf"):

            return float("inf")
        else:

            memory_limit = str(memory_limit)
            memory_limit = memory_limit.lower()

            if memory_limit[-1] == "g":

                return float(memory_limit[0:-1]) * 1024 * 1024 * 1024
            elif memory_limit[-1] == "m":

                return float(memory_limit[0:-1]) * 1024 * 1024
            elif memory_limit[-1] == "k":

                return float(memory_limit[0:-1]) * 1024
            elif memory_limit[-1] == "b":

                return float(memory_limit[0:-1]) * 1024
            else:

                return float(memory_limit)

    @staticmethod
    def memory_exceeded(signum, handler):

        raise Exception("Memory exceeded.")

    @staticmethod
    def set(memory_limit):

        memory_limit_in_b = MemoryLimit.get_memory_limit_in_b(memory_limit)
        if memory_limit_in_b < float("inf"):

            resource.setrlimit(
                resource.RLIMIT_AS, (memory_limit_in_b, memory_limit_in_b))

            signal.signal(signal.SIGSEGV, MemoryLimit.memory_exceeded)
