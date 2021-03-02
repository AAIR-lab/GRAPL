import re

from util import constants
from util import file


class Solution:

    _ACTION_LIST_REGEX = re.compile("\((\w|\W)*\)")

    _PROPERTY_NODES_EXPANDED = "nodes_expanded"
    _PROPERTY_IS_PLAN_FOUND = "is_plan_found"
    _PROPERTY_PLAN_LENGTH = "plan_length"
    _PROPERTY_NAME = "name"

    @staticmethod
    def parse(solution_filepath):

        file_handle = open(solution_filepath, "r")

        properties = None
        action_list = []
        for line in file_handle:

            line = line.strip()

            if properties is None:

                properties = file.parse_properties(line)
                if properties is not None:

                    continue

            action_match = Solution._ACTION_LIST_REGEX.match(line)
            if action_match is not None:

                action_list.append(line)

        assert properties is not None
        solution = Solution(action_list, properties=properties)
        return solution

    def __init__(self, action_list, problem=None, properties=None):

        self._action_list = action_list

        self._result_dict = {}

        if properties is None:

            assert problem is not None

            self._properties = {

                "experiment": {
                    "name": constants.EXPERIMENT_NAME,
                    "id": constants.EXPERIMENT_ID,
                    "hostname": constants.HOSTNAME
                },

                "git": {
                    "sha": constants.GIT_SHA,
                    "branch": constants.GIT_BRANCH,
                    "is_dirty": constants.GIT_IS_DIRTY
                },

                "solution": {

                    Solution._PROPERTY_PLAN_LENGTH: len(self._action_list),
                    "tag_types": constants.TAG_TYPES,
                    "tag_binary_goals": constants.TAG_BINARY_GOALS,
                    "tag_unary_goals": constants.TAG_UNARY_GOALS,
                    "use_artificial_g": constants.USE_ARTIFICIAL_G,
                },

                "problem": problem.get_properties(),

                "algorithm": {}
            }
        else:

            assert problem is None
            self._properties = properties

    def get_properties(self):

        return self._properties

    def set_algorithm(self, algorithm):

        assert isinstance(algorithm, str)
        self.set_algorithm_property("name", algorithm)

    def set_name(self, name):

        assert isinstance(name, str)
        self.set_solution_property(Solution._PROPERTY_NAME, name)

    def set_nodes_expanded(self, nodes_expanded):

        assert isinstance(nodes_expanded, int)
        self.set_solution_property(Solution._PROPERTY_NODES_EXPANDED,
                                   nodes_expanded)

    def set_is_plan_found(self, is_plan_found):

        assert isinstance(is_plan_found, bool)
        self.set_solution_property(Solution._PROPERTY_IS_PLAN_FOUND,
                                   is_plan_found)

    def set_solution_property(self, key, value):

        self.set_property("solution", key, value)

    def override_solution_property(self, key, value):

        self.override_property("solution", key, value)

    def set_algorithm_property(self, key, value):

        self.set_property("algorithm", key, value)

    def override_property(self, label, key, value):

        assert label in self._properties
        assert key in self._properties[label]
        self._properties[label][key] = value

    def set_property(self, label, key, value):

        assert label in self._properties
        assert key not in self._properties[label]
        self._properties[label][key] = value

    def get_plan_length(self):

        return len(self._action_list)

    def get_action_list(self):

        return self._action_list

    def write(self, solution_filepath):

        file_handle = open(solution_filepath, "w")

        file.write_properties(file_handle, self._properties, ";")

        for grounded_action in self._action_list:

            file_handle.write("%s\n" % (grounded_action))

        file_handle.close()
