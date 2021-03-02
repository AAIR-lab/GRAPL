'''
Created on Feb 5, 2020

@author: rkaria
'''

import logging
import math
import multiprocessing

from util import executor


logger = logging.getLogger(__name__)


class TestExample:

    def __init__(self, a1, a2, a3):

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

    def generate_args(self, chunk_size):

        total_chunks = math.ceil(len(self.a1) / chunk_size)
        for chunk_no in range(total_chunks):

            start = chunk_no * chunk_size
            end = min(len(self.a1), start + chunk_size)

            # (range(x, y)) expands to just range(x, y)
            # Use (range(x, y), ) to force it to remain (range(x, y), )
            yield (range(start, end), )

    def generate(self, array_range):

        result = []
        for i in array_range:

            result.append(self.a1[i] + self.a2[i] + self.a3[i])
        return result

    def run(self, max_workers=multiprocessing.cpu_count(), chunk_size=20,
            use_mpi=False):

        results = executor.multicore_execute(self.generate, (),
                                             self.generate_args, max_workers,
                                             chunk_size, use_mpi)

        print(results)


def example_generate_multicore_array_args(chunk_size, array_1,
                                          array_2, array_3):

    assert chunk_size > 0
    assert len(array_1) == len(array_2) == len(array_3)

    # Use the ceiling value to ensure that if the chunk size does not
    # equally divide the work, then the last extra chunk will handle it.
    total_chunks = math.ceil(len(array_1) / chunk_size)

    logger.debug("Generating total_chunks=%u" % (total_chunks))

    for chunk_no in range(total_chunks):

        start = chunk_no * chunk_size
        end = min(len(array_1), start + chunk_size)

        yield (array_1[start:end], array_2[start:end], array_3[start:end])


def example_add_multicore_arrays(array_1, array_2, array_3):

    result = []

    for i in range(len(array_1)):

        result.append(array_1[i] + array_2[i] + array_3[i])

    return result


def example_multicore():

    a1 = range(100)
    a2 = range(100)
    a3 = range(100)

    results = executor.multicore_execute(example_add_multicore_arrays,
                                         (a1, a2, a3),
                                         example_generate_multicore_array_args,
                                         chunk_size=10)

    print(results)

    # Try a object oriented example.
    test_example = TestExample(a1, a2, a3)
    test_example.run()


def example_singlecore():

    a1 = range(100)
    a2 = range(100)
    a3 = range(100)

    results = executor.singlecore_execute(example_add_multicore_arrays,
                                          (a1, a2, a3))

    print(results)


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    example_multicore()
    example_singlecore()
