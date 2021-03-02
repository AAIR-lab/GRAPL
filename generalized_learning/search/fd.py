
import logging
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time

from concretized.problem import Problem
from concretized.solution import Solution
from search.algorithm import Algorithm
from util import constants
from util import file


logger = logging.getLogger(__name__)


class FD(Algorithm):

    # The ff executable.
    _FD_EXECUTABLE = (pathlib.Path(__file__).absolute().parent /
                      "../../bin/fast-downward/fast-downward.py").resolve()

    # The default extension for the log file.
    SOLN_LOGFILE_EXT = ".fd.log"

    #: The source name  for FD.
    _NAME = "fd"

    #: The plan regex for FD.
    # TODO: Add a group for action-cost.
    _PLAN_REGEX = re.compile("(?P<action>(\w|\W)*) \\(\d+\\)($|\n)")

    _SOLN_FAILED_RETCODE = 12
    _SOLN_TIMEOUT_RETCODE = 23

    #: The nodes expanded regex for FD.
    _NODES_EXPANDED_REGEX = re.compile(
        "Expanded (?P<nodes_expanded>(\d+)) state(\w|\W)*")

    _LAMA_FIRST = "lama_first"
    _ASTAR_FF = "astar_ff"
    _ASTAR_LMCUT = "astar_lmcut"
    _EHC_FF = "ehc_ff"
    _EHC_LMCUT = "ehc_lmcut"

    _MODES = set([_LAMA_FIRST, _ASTAR_FF, _ASTAR_LMCUT, _EHC_FF, _EHC_LMCUT])
    _MODE_DICT = {

        _ASTAR_FF: "astar(ff())",
        _ASTAR_LMCUT: "astar(lmcut())",
        _EHC_FF: "ehc(ff())",
        _EHC_LMCUT: "ehc(lmcut())",
    }

    DEFAULT_MEMORY_LIMIT = "3g"

    def __init__(self, name, search_param_dict):

        super(FD, self).__init__(name, search_param_dict)
        self._mode = search_param_dict["mode"]

    @staticmethod
    def _parse_log(logfile_handle, problem, mode, timeout, failed):

        # Reset the logfile_handle back to offset 0.
        logfile_handle.seek(0)

        # First, iterate until we get to the plan.
        regex_matches, next_line = file.extract_contiguous(
            logfile_handle,
            FD._PLAN_REGEX,
            ["action"])

        # Extract the plan.
        action_list = []
        for regex_action in regex_matches:

            action = regex_action["action"]
            action = action.strip()
            action = action.lower()
            action = "(%s)" % (action)
            action_list.append(action)

        # Create the solution.
        solution = Solution(action_list, problem=problem)
        solution.set_algorithm(FD._NAME)

        if timeout is True or failed:

            # The file might be corrupt at this point so we don't really care
            # if the other fields were parsed incorrectly. This flag should be
            # enough to tell us not to trust the plan!
            solution.set_is_plan_found(False)
        else:

            # failed is set to True whenever FD cannot find a solution.
            assert problem.is_relaxed_reachable()
            assert len(action_list) > 0 \
                or problem.is_goal_satisfied(problem.get_initial_state())
            solution.set_is_plan_found(True)

            # Its possible that the file_handle is already at EOF, which is
            # usually the case when the initial state was the goal state.
            #
            # Reset the logfile_handle back to position 0.
            logfile_handle.seek(0)

            # Extract the nodes expanded.
            regex_matches, next_line = file.extract_contiguous(
                logfile_handle,
                FD._NODES_EXPANDED_REGEX,
                ["nodes_expanded"],
                next_line)

            assert len(regex_matches) == 1
            nodes_expanded = int(regex_matches[0]["nodes_expanded"])

            solution.set_nodes_expanded(nodes_expanded)

            assert mode in FD._MODES
            solution.set_algorithm_property("heuristic", {"name": mode})

        return solution

    def _get_fd_cmd(self, plan_filepath, sas_filepath,
                    domain_filepath, problem_filepath):

        memory_limit_string = "--overall-memory-limit %s" % (
            FD.DEFAULT_MEMORY_LIMIT)

        if self._time_limit_in_sec != float("inf"):

            timeout_string = "--overall-time-limit %u" % (
                int(self._time_limit_in_sec))
        else:

            timeout_string = ""

        fd_cmd = "%s %s %s --plan-file %s --sas-file %s" % (self._FD_EXECUTABLE,
                                                            memory_limit_string,
                                                            timeout_string,
                                                            plan_filepath,
                                                            sas_filepath)

        if self._mode == FD._LAMA_FIRST:

            fd_cmd += " --alias lama-first %s %s" % (domain_filepath,
                                                     problem_filepath)
        else:

            assert self._mode in FD._MODES
            fd_cmd += " %s %s --search \"%s\"" % (domain_filepath,
                                                  problem_filepath,
                                                  FD._MODE_DICT[self._mode])

        return fd_cmd

    def _terminate(self, completed_process):

        try:

            completed_process.terminate()
        except Exception:

            pass

    def _kill(self, completed_process):

        try:

            completed_process.kill()
        except Exception:

            pass

    def _search(self, domain_filepath, problem_filepath):

        plan_filepath = "%s/%s.plan" % (problem_filepath.parent,
                                        problem_filepath.name)

        sas_filepath = "%s/%s.sas" % (problem_filepath.parent,
                                      problem_filepath.name)

        fd_cmd = self._get_fd_cmd(plan_filepath, sas_filepath,
                                  domain_filepath, problem_filepath)

        temp_logfile_handle = tempfile.NamedTemporaryFile("w+", delete=False)

        logger.debug("Running fd_cmd=%s" % (
            fd_cmd))

        start_time = time.time()
        timeout = False
        failed = False

        try:

            completed_process = subprocess.Popen(
                fd_cmd,
                shell=True,
                stdout=temp_logfile_handle)

            completed_process.wait(timeout=self._time_limit_in_sec)

            # http://www.fast-downward.org/ExitCodes
            failed = completed_process.returncode >= 4
            timeout = completed_process.returncode == FD._SOLN_TIMEOUT_RETCODE
        except subprocess.TimeoutExpired:

            self._terminate(completed_process)
            self._kill(completed_process)
            timeout = True
            failed = True
            pass

        total_time = time.time() - start_time
        total_time = min(total_time, self._time_limit_in_sec)

        # Get the problem parsed out.
        assert domain_filepath.parent == problem_filepath.parent
        problem = Problem(domain_filepath.name, problem_filepath.name,
                          problem_filepath.parent)

        # Parse the solution.
        solution = self._parse_log(
            temp_logfile_handle, problem, self._mode,
            timeout,
            failed)

        solution.set_solution_property("time_taken", "%.2f" % (total_time))

        # Set the algorithm properties.
        algorithm_properties = {
            "time_limit": self.get_time_limit_in_sec(),
            "nodes_expanded_limit": self.get_nodes_expanded_limit(),
        }

        for key in algorithm_properties:

            solution.set_algorithm_property(key, algorithm_properties[key])

        # Finally, check if we need to save the logfiile.
        temp_logfile_handle.close()
        if self._save_logs:

            log_filename = "%s.%s.%s.%s" % (problem_filepath.name,
                                            self._mode, self._name,
                                            constants.LOG_FILE_EXT)
            log_filepath = pathlib.Path(problem_filepath.parent, log_filename)
            shutil.move(temp_logfile_handle.name, log_filepath)
        else:

            os.remove(temp_logfile_handle.name)

        # Remove intermediate files.
        try:

            os.remove(plan_filepath)
        except FileNotFoundError:

            pass

        try:

            os.remove(sas_filepath)
        except FileNotFoundError:

            pass

        # Save the solution to a compliant file.
        solution_filename = "%s.%s.%s" % (problem_filepath.name, self._name,
                                          constants.SOLUTION_FILE_EXT)
        solution_filepath = pathlib.Path(problem_filepath.parent,
                                         solution_filename)

        solution.set_name(self._name)
        solution.write(solution_filepath)

        # Return nothing since the solution has already been written to the
        # disk.
        #
        # Returning nothing ensures that the phase system does not rely upon
        # results from this phase.
        return []

    def search(self, domain_filepath, problem_filepath):

        if self._mode == "all":

            old_name = self._name
            for mode in FD._MODES:

                self._name = "%s_%s" % (old_name, mode)
                self._mode = mode

                self._search(domain_filepath, problem_filepath)
        else:

            return self._search(domain_filepath, problem_filepath)
