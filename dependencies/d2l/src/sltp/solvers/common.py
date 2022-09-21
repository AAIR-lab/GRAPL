import logging
import os
import signal
import subprocess
import sys
from threading import Timer

from ..solvers import library


class Solution:
    def __init__(self):
        self.cost = sys.maxsize
        self.assignment = dict()
        self.solved = False
        self.result = "UNPARSED"

    def parse_result(self, line):
        self.result = line
        self.solved = {"UNSATISFIABLE": False, "UNKNOWN": False, "OPTIMUM FOUND": True, "SATISFIABLE": True}[line]


def parse_maxsat_output(filename):
    solution = Solution()
    with open(filename, 'r') as f:
        # We keep only lines which are the variable assignment (v), the result code (s), or cost bounds (o),
        # the last of which should be the actual cost of the solution, if a solution was found.
        relevant = ((line[0], line[1:].strip()) for line in f if line and line[0] in ('v', 's', 'o'))
        for t, line in relevant:
            if t == 'o':
                solution.cost = min(solution.cost, int(line))
            elif t == 's':
                solution.parse_result(line)
            else:
                assert t == 'v'
                lits = (int(x) for x in line.split(' '))
                solution.assignment = {abs(x): x > 0 for x in lits}

        return solution


def choose_solver(solver):
    if solver == 'wpm3':
        return library.WPM3
    elif solver == 'maxino':
        return library.Maxino
    elif solver == 'openwbo':
        return library.Openwbo
    elif solver == 'openwbo-inc':
        return library.OpenwboInc
    raise RuntimeError('Unknown solver "{}"'.format(solver))


def solve(rundir, cnf_filename, solver='wpm3', timeout=None):
    solver = choose_solver(solver)(rundir)
    error, output = run_solver(solver, rundir, cnf_filename, 'maxsat_solver', timeout=timeout)

    if error:
        raise RuntimeError("There was an error running the MAXSAT solver. Check error logs")

    return parse_maxsat_output(output)


def run_solver(solver, rundir, input_filename, tag=None, stdout=None, timeout=None):
    assert tag or stdout
    error = False
    if stdout is None:
        stdout = os.path.join(rundir, f'{tag}_run.log')

    with open(stdout, 'w') as driver_log:
        with open(os.path.join(rundir, f'{tag}_run.err'), 'w') as driver_err:
            cmd = solver.command(input_filename)
            logging.info(f'Executing (timeout: {timeout} s.) "{cmd}" on directory "{rundir}"')

            proc = subprocess.Popen(cmd, cwd=rundir, stdout=driver_log, stderr=driver_err,
                                    bufsize=0)  # use unbuffered output
            run_with_controlling_timer(proc, timeout)

            # retcode = subprocess.call(cmd, cwd=rundir, stdout=driver_log, stderr=driver_err)
    if os.path.getsize(driver_err.name) != 0:
        error = True
    if os.path.getsize(driver_err.name) == 0:  # Delete error log if empty
        os.remove(driver_err.name)
    return error, driver_log.name


def run_with_controlling_timer(proc, timeout=None):
    """ Run the given process.
    If a timeout is specified, send a SIGTERM after that timeout, and a SIGKILL a bit later.
    Otherwise, just run the process normally. """
    def send_sigterm():
        logging.warning(f"Sending SIGTERM to subprocess with ID {proc.pid}")
        os.kill(proc.pid, signal.SIGTERM)

    timeout = 0 if timeout is None else timeout
    slack = 2
    sigterm_timer = Timer(timeout, send_sigterm)  # Timeout in seconds
    sigkill_timer = Timer(timeout + slack, proc.kill)  # Give a little slack for the process to shutdown

    try:
        if timeout:
            sigterm_timer.start()
            sigkill_timer.start()
        proc.communicate()
    finally:
        if timeout:
            sigterm_timer.cancel()
            sigkill_timer.cancel()
