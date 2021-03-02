

from search.node import Node
from .heuristic import Heuristic


class BFS(Heuristic):

    def __init__(self, problem):

        super(BFS, self).__init__("bfs", problem)
        self._visited = set()

    def expand(self, parent):

        depth = parent.get_depth() + 1

        concrete_state = parent.get_concrete_state()
        applicable_actions = self._problem.get_applicable_actions(
            concrete_state)

        expanded_nodes = []
        for action in applicable_actions:

            next_state = action.apply(concrete_state)
            if next_state not in self._visited:

                # Use the depth of the child as the fscore. This ensures
                # a bfs traversal. (tie-breaking is resolved via entry counts)
                child = Node(next_state, parent, action, depth, depth)
                self._visited.add(next_state)
                expanded_nodes.append(child)

        return expanded_nodes
