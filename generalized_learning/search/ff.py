
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


class FF(Algorithm):

    # The ff executable.
    _FF_EXECUTABLE = (pathlib.Path(__file__).absolute().parent /
                      "../../dependencies/binaries/ff_v2.3m").resolve()

    #: The source name  for FF.
    _NAME = "ff"

    #: The plan regex for FF.
    _PLAN_REGEX = re.compile("(\w|\W)* \d+: (?P<action>(\w|\W)*)")

    #: The nodes expanded regex for FF.
    _NODES_EXPANDED_REGEX = re.compile(
        "(\w|\W)* expanding (?P<nodes_expanded>(\d+)) states(\w|\W)*")

    def __init__(self, name, search_param_dict):

        super(FF, self).__init__(name, search_param_dict)

    @staticmethod
    def _parse_log(logfile_handle, problem, timeout, failed):

        # Reset the temp_file back to offset 0.
        logfile_handle.seek(0)

        # First, iterate until we get to the plan.
        regex_matches, next_line = file.extract_contiguous(
            logfile_handle,
            FF._PLAN_REGEX,
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
        solution.set_algorithm(FF._NAME)

        if timeout or failed:

            # The file might be corrupt at this point so we don't really care
            # if the other fields were parsed incorrectly. This flag should be
            # enough to tell us not to trust the plan!
            solution.set_is_plan_found(False)
        else:

            #             assert problem.is_relaxed_reachable()
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
                FF._NODES_EXPANDED_REGEX,
                ["nodes_expanded"],
                next_line)

            assert len(regex_matches) == 1
            nodes_expanded = int(regex_matches[0]["nodes_expanded"]) - 1
            solution.set_nodes_expanded(nodes_expanded)

        return solution

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

    def search(self, domain_filepath, problem_filepath):

        temp_logfile_handle = tempfile.NamedTemporaryFile(
            "w+",
            delete=False)

        ff_cmd = "%s -o %s -f %s" % (self._FF_EXECUTABLE,
                                     domain_filepath,
                                     problem_filepath)

        logger.debug("Running ff_cmd=%s" % (
            ff_cmd))
        ff_cmd = ff_cmd.split(" ")

        start_time = time.time()

        timeout = False
        failed = False

        try:

            completed_process = subprocess.Popen(
                ff_cmd,
                stdout=temp_logfile_handle)

            completed_process.wait(timeout=self._time_limit_in_sec)

            # If we did not timeout, then the command must have run
            # successfully.
            #
            # Currently, we do not support internal errors.
            if completed_process.returncode != 0:

                failed = True
        except subprocess.TimeoutExpired:

            self._terminate(completed_process)
            self._kill(completed_process)
            timeout = True
            pass

        total_time = time.time() - start_time
        total_time = min(total_time, self._time_limit_in_sec)

        # Get the problem parsed out.
        assert domain_filepath.parent == problem_filepath.parent
        problem = Problem(domain_filepath.name, problem_filepath.name,
                          problem_filepath.parent)

        # Parse the solution.
        solution = self._parse_log(
            temp_logfile_handle, problem, timeout, failed)

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

            log_filename = "%s.%s.%s" % (problem_filepath.name, self._name,
                                         constants.LOG_FILE_EXT)
            log_filepath = pathlib.Path(problem_filepath.parent, log_filename)
            shutil.move(temp_logfile_handle.name, log_filepath)
        else:

            try:

                os.remove(temp_logfile_handle.name)
            except FileNotFoundError:

                pass

        try:

            # ff also creates a <problem>.soln file.
            # delete that file.
            #
            # TODO: What if a solution could not be found and consequently this
            #     file does not exist?
            os.remove("%s/%s.%s" % (problem_filepath.parent,
                                    problem_filepath.name,
                                    constants.SOLUTION_FILE_EXT))
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
