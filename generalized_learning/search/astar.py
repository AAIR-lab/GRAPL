'''
Created on Feb 13, 2020

@author: rkaria
'''

import time

import tqdm

from concretized.problem import Problem
from search.algorithm import Algorithm
from search.heuristic.nn_plact import NNPLACT
from util import constants
from util import file
from util.priorityq import PriorityQ

from .heuristic.bfs import BFS
from .heuristic.nn_rollout import NNRollout
from .node import Node


class AStar(Algorithm):

    _NAME = "astar"

    @staticmethod
    def get_heuristic(search_param_dict, problem, parent_dir):

        heuristic_type = search_param_dict["heuristic"]

        if "bfs" == heuristic_type:

            return BFS(problem)
        elif "nn_rollout" == heuristic_type:

            model_dir = file.get_relative_path(search_param_dict["model_dir"],
                                               parent_dir)

            return NNRollout(problem, model_dir,
                             search_param_dict["model_name"])
        elif "nn_plact" == heuristic_type:

            model_dir = file.get_relative_path(search_param_dict["model_dir"],
                                               parent_dir)

            return NNPLACT(problem, model_dir,
                           search_param_dict["model_name"])
        else:

            raise Exception("Unknown heuristic_type={}".format(heuristic_type))

    def __init__(self, name, search_param_dict, parent_dir):

        super(AStar, self).__init__(name, search_param_dict)

        self._parent_dir = parent_dir
        self._search_param_dict = search_param_dict

    def _get_g_score(self, g_scores, state):

        try:
            node = g_scores[state]
            return node.get_depth()
        except KeyError:

            return float("inf")

    def search(self, domain_filepath, problem_filepath):

        start_time = time.time()
        end_time = start_time + self._time_limit_in_sec

        self._reset()

        assert domain_filepath.parent == problem_filepath.parent
        problem = Problem(domain_filepath.name,
                          problem_filepath.name,
                          problem_filepath.parent)

        heuristic = AStar.get_heuristic(self._search_param_dict,
                                        problem,
                                        self._parent_dir)

        initial_state = problem.get_initial_state()
        root_node = Node(initial_state, None, None, 0, 0)
        root_node.artificial_g = 0.0
        root_node._h = 0

        state_node_dict = {}
        fringe = PriorityQ()

        fringe.push(initial_state, 0.0)
        state_node_dict[initial_state] = root_node

#         progress_bar = tqdm.tqdm(unit=" nodes")
        goal_node = None

        true_g = {}
        true_g[initial_state] = 0

        while not fringe.is_empty() \
                and time.time() < end_time \
                and self._total_nodes_expanded < self._nodes_expanded_limit:

            #             progress_bar.update(1)
            current_state = fringe.pop()
            current_node = state_node_dict[current_state]
            del state_node_dict[current_state]

            action = current_node.get_action()
#             print("picked:", action)

            if problem.is_goal_satisfied(current_state):

                goal_node = current_node
                break
            else:

                self._total_nodes_expanded += 1

                applicable_actions = problem.get_applicable_actions(
                    current_state)

                candidates = []

#                 print("***************************")
                for action in applicable_actions:

                    next_state = action.apply(current_state)

                    assert action.get_cost() == 1

                    # g_scores for the current state must always be available.
                    assert current_state in true_g
                    temp_g_score = true_g[current_state] + action.get_cost()

                    if temp_g_score < true_g.get(next_state, float("inf")):

                        # Better path found, update priority in fringe
                        # (invalidate the old value) and also update the
                        # node reference for this state in the open set.
                        h = heuristic.compute_h(current_state,
                                                next_state, action)


#                         print("a:", action, "g:", temp_g_score,
#                               "h:", h, "f:", f_score)

                        child_node = Node(next_state, current_node, action,
                                          temp_g_score,
                                          0)
                        state_node_dict[next_state] = child_node
                        true_g[next_state] = temp_g_score

                        candidates.append(child_node)

                if len(candidates) > 0:

                    heuristic.update_candidates(current_node, candidates)

                for child_node in candidates:
                    fringe.push(child_node.get_concrete_state(),
                                (child_node.get_fscore(), child_node.get_h()))

#         progress_bar.close()
        solution = self._create_solution(problem, goal_node, start_time,
                                         heuristic)

        solution.write(self.get_solution_filepath(problem_filepath))
        return []
