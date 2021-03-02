#!/usr/bin/env python3

from abc import ABC
from abc import abstractmethod
import pathlib
import time

from concretized.solution import Solution
from util import constants


class Algorithm(ABC):

    _NAME = "invalid"

    DEFAULT_NODES_EXPANDED_LIMIT = float("inf")
    DEFAULT_TIME_LIMIT_IN_SEC = 600

    @abstractmethod
    def __init__(self, name, search_param_dict):

        self._name = name

        if "save_logs" in search_param_dict:

            self._save_logs = search_param_dict["save_logs"]
        else:

            self._save_logs = True

        self._time_limit_in_sec = Algorithm.DEFAULT_TIME_LIMIT_IN_SEC
        self._nodes_expanded_limit = Algorithm.DEFAULT_NODES_EXPANDED_LIMIT

        if "time_limit" in search_param_dict:

            try:
                self._time_limit_in_sec = float(
                    search_param_dict["time_limit"])
            except Exception:

                pass

        if "nodes_expanded_limit" in search_param_dict:

            try:
                self._nodes_expanded_limit = float(
                    search_param_dict["nodes_expanded_limit"])
            except Exception:

                pass

        self._reset()

    def _reset(self):

        self._total_nodes_expanded = 0

    @abstractmethod
    def search(self, domain_filepath, problem_filepath):

        raise NotImplementedError

    def get_solution_filepath(self, problem_filepath):

        return pathlib.Path("%s.%s.%s" % (problem_filepath,
                                          self._name,
                                          constants.SOLUTION_FILE_EXT))

    def get_time_limit_in_sec(self):

        return self._time_limit_in_sec

    def get_nodes_expanded_limit(self):

        return self._nodes_expanded_limit

    def get_total_nodes_expanded(self):

        return self._total_nodes_expanded

    def get_avg_branching_factor(self, plan_length):

        if plan_length > 0:

            avg_branching_factor = \
                (1.0 * self._total_nodes_expanded) / plan_length
        else:

            avg_branching_factor = 0.00

        return avg_branching_factor

    def _create_solution(self, problem, goal_node, start_time, heuristic):

        grounded_action_list = []

        if goal_node is not None:

            node = goal_node
            while node.get_parent() is not None:

                grounded_action_list.append(str(node.get_action()))
                node = node.get_parent()

            grounded_action_list.reverse()

#         print("############ SOLUTION ############")
#         print("Plan length:", len(grounded_action_list))
#         print(grounded_action_list)
#         print("Total expanded:", self.get_total_nodes_expanded())
#         print("##################################")

        solution = Solution(grounded_action_list, problem=problem)

        solution.set_name(self._name)
        solution.set_algorithm(self._NAME)

        if goal_node is not None:

            solution.set_is_plan_found(True)
        else:

            solution.set_is_plan_found(False)

        solution.set_nodes_expanded(self.get_total_nodes_expanded())

        solution.set_solution_property(
            "avg_branching_factor",
            self.get_avg_branching_factor(len(grounded_action_list)))

        solution.set_solution_property("time_taken",
                                       "%.2f" % (time.time() - start_time))

        # Set the algorithm properties.
        algorithm_properties = {
            "time_limit": self.get_time_limit_in_sec(),
            "nodes_expanded_limit": self.get_nodes_expanded_limit(),
            "heuristic": heuristic.get_properties()
        }

        for key in algorithm_properties:

            solution.set_algorithm_property(key, algorithm_properties[key])

        return solution
