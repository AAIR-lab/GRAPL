
import linecache
import logging
import os
import resource
import time
import tracemalloc

from . import console


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def memory_usage():
    """ Return the memory usage in MB """
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(1024*1024)
    return mem


def print_memory_usage():
    logging.info("Total memory usage: {:.2f}MB".format(memory_usage()))
    logging.info('Max. memory usage: {:.2f}MB'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
