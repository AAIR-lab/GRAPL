import heapq

from abstraction.state import AbstractState
from neural_net.nn import NN
from search.heuristic.heuristic import Heuristic
from search.node import Node


class NNRollout(Heuristic):

    def __init__(self, problem, model_dir, model_name):

        super(NNRollout, self).__init__("nn_rollout", problem)
        self._visited = set()

        self._nn = NN.load(model_dir, model_name)

    def expand(self, parent):

        depth = parent.get_depth() + 1

        current_state = parent.get_concrete_state()
        applicable_actions = self._problem.get_applicable_actions(
            current_state)

        action_entry_count = 0
        action_score_heap = []

        abstract_state = AbstractState(self._problem, current_state)

        for action in applicable_actions:

            action_score = 1.0 - self._nn.get_action_score(self._problem,
                                                           abstract_state,
                                                           action)

            heapq.heappush(action_score_heap, (action_score,
                                               action_entry_count,
                                               action))

            action_entry_count += 1

        expanded_nodes = []
        while len(action_score_heap) > 0:

            action_score, _, action = heapq.heappop(action_score_heap)
            next_state = action.apply(current_state)

            if next_state not in self._visited:

                child = Node(next_state, parent, action,
                             depth, action_score)
                self._visited.add(next_state)
                expanded_nodes.append(child)

                # Rollout just allows a single new node to be added.
                break

        return expanded_nodes
