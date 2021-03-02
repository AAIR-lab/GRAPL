'''
Created on Feb 13, 2020

@author: rkaria
'''

import heapq

import tqdm

from concretized.problem import Problem
from search.algorithm import Algorithm
from search.heuristic.nn_plact import NNPLACT
from util import file

from .heuristic.bfs import BFS
from .heuristic.nn_rollout import NNRollout
from .node import Node


class InformedSearch(Algorithm):

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

        super(InformedSearch, self).__init__(name)

        self._parent_dir = parent_dir
        self._search_param_dict = search_param_dict

    def _push_to_fringe(self, fringe, node):

        fscore = node.get_fscore()

        heapq.heappush(fringe, (fscore, self._node_order, node))
        self._node_order += 1

    def _pop_from_fringe(self, fringe):

        _, _, node = heapq.heappop(fringe)
        return node

    def search(self, domain_filepath, problem_filepath):

        self._reset()

        assert domain_filepath.parent == problem_filepath.parent
        problem = Problem(domain_filepath.name,
                          problem_filepath.name,
                          problem_filepath.parent)

        heuristic = InformedSearch.get_heuristic(self._search_param_dict,
                                                 problem,
                                                 self._parent_dir)

        initial_state = problem.get_initial_state()
        root_node = Node(initial_state, None, None, 0, 0)

        fringe_list = []
        self._push_to_fringe(fringe_list, root_node)

        progress_bar = tqdm.tqdm(unit=" nodes")
        progress_bar.update(1)
        goal_node = None
        while len(fringe_list) > 0:

            node = self._pop_from_fringe(fringe_list)
            action = node.get_action()
            print(action)
            concrete_state = node.get_concrete_state()

            if problem.is_goal_satisfied(concrete_state):

                goal_node = node
                break
            else:

                self._total_nodes_expanded += 1

                new_children = heuristic.expand(node)
                for child in new_children:

                    self._push_to_fringe(fringe_list, child)
                    progress_bar.update(1)

        progress_bar.close()
        return self._create_solution(problem, goal_node)
