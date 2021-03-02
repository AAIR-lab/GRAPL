'''
Created on Feb 5, 2020

@author: rkaria
'''

import concurrent.futures
import logging
import multiprocessing
import tqdm

# Logger for this module.
logger = logging.getLogger(__name__)


def singlecore_execute(func, func_args_tuple):

    logger.info("Executing func=%s in single-core mode." % (func.__name__))
    return func(*func_args_tuple)


def multicore_execute(func, func_args_tuple, func_args_generator,
                      max_workers=multiprocessing.cpu_count(), chunk_size=1,
                      use_mpi=False):

    assert func is not None
    assert max_workers > 0

    logger.info("Executing func=%s in multi-core mode." % (func.__name__))
    if use_mpi:

        # TODO: Get the MPI executor. mpi4py.
        logger.debug("Using mpi-based executor")
        raise NotImplementedError
    else:

        logger.debug("Using process pool executor")
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers)

    logger.debug(
        "Utilizing max_workers=%u with a chunk_size=%s" % (max_workers,
                                                           chunk_size))

    logger.info("Submmitting chunks for func=%s" % (func.__name__))
    progress_bar = tqdm.tqdm(desc="Submitted", unit=" chunks")
    futures = []
    for arg_tuple in func_args_generator(chunk_size, *func_args_tuple):

        logger.debug("Submitting future for func=%s with args=%s" % (
            func.__name__, str(arg_tuple)))
        future = executor.submit(func, *arg_tuple)
        futures.append(future)
        progress_bar.update(1)

    progress_bar.close()

    total_chunks = len(futures)
    logger.info("Submitted total_chunks=%u of chunk_size=%s for func=%s"
                % (total_chunks, chunk_size, func.__name__))

    logger.info("Collecting results")
    progress_bar = tqdm.tqdm(total=total_chunks, desc="Completed",
                             unit=" chunks")

    # Use concurrent.futures.as_completed() to ensure that futures are
    # processed only as they are completed and we do not need to hog the
    # cpu for the same.
    results = []
    for future in concurrent.futures.as_completed(futures):

        results.extend(future.result())
        progress_bar.update(1)

    progress_bar.close()

    logger.debug("Total results collected=%u" % (len(results)))
    return results
