
import logging
import os
import pathlib
import re
import shutil
import signal
import subprocess
import tempfile
import time

from concretized.problem import Problem
from concretized.solution import Solution
from search.algorithm import Algorithm
from util import constants
from util import file


logger = logging.getLogger(__name__)


class Pyperplan(Algorithm):

    # The ff executable.
    _PYPERPLAN_EXECUTABLE = (pathlib.Path(__file__).absolute().parent /
                             "../../dependencies/pyperplan/src/pyperplan.py").resolve()

    # The default extension for the log file.
    SOLN_LOGFILE_EXT = ".pyperplan.log"

    #: The source name  for Pyperplan.
    _NAME = "pyperplan"

    #: The plan regex for Pyperplan.
    # TODO: Add a group for action-cost.
    _PLAN_REGEX = re.compile("(\w|\W)* (?P<action>\\((\w|\W)*\\))($|\n)")

    #: The nodes expanded regex for Pyperplan.
    _NODES_EXPANDED_REGEX = re.compile(
        "(\w|\W)* (?P<nodes_expanded>(\d+)) Nodes expanded(\w|\W)*")

    _ASTAR_FF = "astar_ff"
    _ASTAR_LMCUT = "astar_lmcut"
    _ASTAR_NNPLACT = "astar_nnplact"

    _GBF_FF = "gbf_ff"
    _GBF_LMCUT = "gbf_lmcut"
    _GBF_NNPLACT = "gbf_nnplact"

    _MODES = set([_ASTAR_FF, _ASTAR_LMCUT, _ASTAR_NNPLACT,
                  _GBF_FF, _GBF_LMCUT, _GBF_NNPLACT])
    _MODE_DICT = {

        _ASTAR_FF: "-s astar -H hff",
        _ASTAR_LMCUT: "-s astar -H lmcut",
        _ASTAR_NNPLACT: "-s astar -H nnplact",

        _GBF_FF: "-s gbf -H hff",
        _GBF_LMCUT: "-s gbf -H lmcut",
        _GBF_NNPLACT: "-s gbf -H nnplact",
    }

    DEFAULT_MEMORY_LIMIT = "3g"

    def __init__(self, name, search_param_dict, parent_dir):

        super(Pyperplan, self).__init__(name, search_param_dict)
        self._mode = search_param_dict["mode"]

        assert self._mode in Pyperplan._MODES
        if self._mode == Pyperplan._ASTAR_NNPLACT \
                or self._mode == Pyperplan._GBF_NNPLACT:

            self._model_dir = file.get_relative_path(
                search_param_dict["model_dir"],
                parent_dir)

            self._model_name = search_param_dict["model_name"]

    @staticmethod
    def _parse_log(logfile_handle, problem, mode, timeout, failed):

        # Reset the logfile_handle back to offset 0.
        logfile_handle.seek(0)

        # First, iterate until we get to the plan.
        regex_matches, next_line = file.extract_contiguous(
            logfile_handle,
            Pyperplan._PLAN_REGEX,
            ["action"])

        # Extract the plan.
        action_list = []
        for regex_action in regex_matches:

            action = regex_action["action"]
            action = action.strip()
            action = action.lower()
            action = "%s" % (action)
            action_list.append(action)

        if (len(action_list) == 0) \
                and (problem.is_relaxed_reachable()) \
                and (not problem.is_goal_satisfied(problem.get_initial_state())):

            failed = True

        # Create the solution.
        solution = Solution(action_list, problem=problem)
        solution.set_algorithm(Pyperplan._NAME)

        if timeout is True or failed or not problem.is_relaxed_reachable():

            # The file might be corrupt at this point so we don't really care
            # if the other fields were parsed incorrectly. This flag should be
            # enough to tell us not to trust the plan!
            solution.set_is_plan_found(False)
        else:

            # failed is set to True whenever Pyperplan cannot find a solution.
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
                Pyperplan._NODES_EXPANDED_REGEX,
                ["nodes_expanded"],
                next_line)

            assert len(regex_matches) == 1
            nodes_expanded = int(regex_matches[0]["nodes_expanded"])

            solution.set_nodes_expanded(nodes_expanded)

            assert mode in Pyperplan._MODES
            solution.set_algorithm_property("heuristic", {"name": mode})

        return solution

    def _get_pyperplan_cmd(self, domain_filepath, problem_filepath):

        pyperplan_cmd = "python3 %s %s " % (
            self._PYPERPLAN_EXECUTABLE,
            Pyperplan._MODE_DICT[self._mode])

        if self._mode == Pyperplan._ASTAR_NNPLACT \
                or self._mode == Pyperplan._GBF_NNPLACT:

            pyperplan_cmd += "--model-dir %s --model-name %s " % (
                self._model_dir,
                self._model_name)

        pyperplan_cmd += "%s %s" % (domain_filepath, problem_filepath)

        return pyperplan_cmd

    def _terminate(self, completed_process):

        try:

            completed_process.terminate()
        except Exception:

            pass

    def _kill(self, completed_process):

        try:

            completed_process.send_signal(signal.SIGKILL)
            completed_process.kill()
        except Exception:

            pass

    def _search(self, domain_filepath, problem_filepath):

        pyperplan_cmd = self._get_pyperplan_cmd(domain_filepath,
                                                problem_filepath)

        temp_logfile_handle = tempfile.NamedTemporaryFile("w+", delete=False)

        logger.debug("Running pyperplan_cmd=%s" % (
            pyperplan_cmd))

        pyperplan_cmd = pyperplan_cmd.split(" ")

        start_time = time.time()
        timeout = False
        failed = False

        try:

            completed_process = subprocess.Popen(
                pyperplan_cmd,
                stdout=temp_logfile_handle)

            completed_process.wait(timeout=self._time_limit_in_sec)

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
            for mode in Pyperplan._MODES:

                self._name = "%s_%s" % (old_name, mode)
                self._mode = mode

                self._search(domain_filepath, problem_filepath)
        else:

            return self._search(domain_filepath, problem_filepath)
