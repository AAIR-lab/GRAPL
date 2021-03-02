#
# This file is part of pyperplan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import pathlib

from abstraction.state import AbstractState
from concretized.problem import Problem
from heuristics.heuristic_base import Heuristic
from neural_net.nn import NN
from task import Operator, Task


class NNPlactHeuristic(Heuristic):
    """
    Implements a simple blind heuristic for convenience.
    It returns 0 if the goal was reached and 1 otherwise.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self._nnplact = True

    def initialize_model(self, task, model_dir, model_name):

        domain_filepath = pathlib.Path(task.domain_file)
        problem_filepath = pathlib.Path(task.problem_file)

        self._problem = Problem(domain_filepath.name,
                                problem_filepath.name,
                                problem_filepath.parent)

        from generalized_learning.search.heuristic.nn_plact import NNPLACT
        self._nnplact = NNPLACT(self._problem, model_dir, model_name)

    def encode_pyperplan_initial_state(self):

        return self._problem.encode_pyperplan_initial_state()

    def update_candidates(self, current_node, candidate_list):

        from generalized_learning.search.node import Node

        parent_state = self._problem.encode_pyperplan_state(current_node.state)
        parent_node = Node(parent_state, None, None, 0)
        parent_node.artificial_g = current_node.artificial_g

        candidates = []
        for candidate in candidate_list:

            candidate_state = self._problem.encode_pyperplan_state(
                candidate.state)
            candidate_node = Node(
                candidate_state,
                None,
                self._problem.get_action(candidate.action.name),
                0)

            candidates.append(candidate_node)

        self._nnplact.update_candidates(parent_node, candidates)

        for i in range(len(candidates)):

            candidate_list[i].artificial_g = candidates[i].artificial_g
            candidate_list[i].h = candidates[i].h + candidates[i].artificial_g

    def __call__(self, node):

        concretized_state = self._problem.encode_pyperplan_state(node.state)
        h = self._nnplact.compute_h(None, concretized_state, None)
        if node.parent is None:

            node.action_score = 0
        else:

            parent_state = self._problem.encode_pyperplan_state(
                node.parent.state)
            action_score = self._nnplact.compute_d(
                parent_state,
                None,
                self._problem.get_action(node.action.name))

            node.action_score = node.parent.action_score + action_score

        # Store the confidence value as the artificial g.
        # We will use to compute the f-score rather than the real_g.
        # The real_g is still used to determine if a node must be added to
        # the open list in A*.
        node.artificial_g = node.action_score
        node.h = h

        return h + node.artificial_g
