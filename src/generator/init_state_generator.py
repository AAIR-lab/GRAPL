
from pddlgym.spaces import LiteralSpace
from pddlgym import structs
import collections
import time
import itertools
from utils.time_thread import TimeThread
import random

class Node:

    def __init__(self, state, action, parent, depth, node_id):

        self.state = frozenset(state)
        self.action = action
        self.parent = parent
        self.depth = depth
        self.node_id = node_id

    def __hash__(self):

        raise NotImplementedError
    
    @staticmethod
    def print_transitions(goal_node):

        transitions = []
        node = goal_node
        while node != None:

            transitions.append(node.state)
            transitions.append(node.node_id)
            transitions.append(node.action)

            node = node.parent

        for step in reversed(transitions):

            if isinstance(step, frozenset):
                print(structs.LiteralConjunction(step).pddl_str())
            elif isinstance(step, int):
                print("ID:", step)
            elif step is not None:
                print(step.pddl_str())

class InitStateGenerator:

    def __init__(self, show_progress=True,
                 progress_bar_position=0):

        self.show_progress = show_progress
        self.progress_bar_position = progress_bar_position
    
    @staticmethod
    def print_transitions(goal_node):
        
        Node.print_transitions(goal_node)

    def get_all_actions(self, domain, objects):

        action_space = LiteralSpace(
            [domain.predicates[a] for a in domain.actions],
            type_to_parent_types=domain.type_to_parent_types)

        dummy_state = structs.State(literals=set(),
                                    objects=objects,
                                    goal=set())
        action_space._update_objects_from_state(dummy_state)

        all_actions = action_space.all_ground_literals(dummy_state)
        return all_actions

    def def_get_action_dicts(self, domain, objects):

        ground_actions_dict = {}
        action_depth = {}

        all_actions = self.get_all_actions(domain, objects)
        for a in all_actions:

            action = domain.operators[a.predicate.name]
            action = action.ground(a.variables, with_copy=True)

            action_depth[a.predicate.name] = float("inf")
            ground_actions_dict[a] = action

        return ground_actions_dict, action_depth

    def _can_make_action_applicable(self, state, action,
                                    init_state_neg_literals,
                                    constraints=[]):

        pos_literals = set()
        for literal in action.preconds.literals:

            if literal.is_negative:
                if literal.positive in state:
                    return False
            elif literal in init_state_neg_literals:
                return False
            else:
                pos_literals.add(literal.positive)

        new_state = state.union(pos_literals)
        for constraint in constraints:
            if not constraint(new_state, action):
                return False

        return True

    def _make_action_applicable(self, state, action,
                                init_state_neg_literals):

        for literal in action.preconds.literals:
            if not literal.is_negative:
                state.add(literal)
            else:
                init_state_neg_literals.add(literal.positive)

    def _get_successors(self, ground_actions_dict, a, node, counter):

        action = ground_actions_dict[a]
        assert action.is_optimized
        assert isinstance(action.effects,
                          structs.ProbabilisticEffect)



        successors = []
        for next_state in action.effects.apply_all(node.state):
            child_node = Node(next_state, a, node, node.depth + 1,
                              next(counter))
            successors.append(child_node)

        return successors

    def _perform_walk(self, init_state, ground_actions_dict,
                      visited,
                      uncovered_actions,
                      action_depth,
                      covered_actions,
                      depth_state_dict,
                      init_state_neg_literals,
                      time_limit_in_sec=60,
                      skip_uncovered_check=False,
                      can_try_making_applicable=False,
                      try_applicable_chance=0.0,
                      constraints=[]):

        init_state = set(frozenset(init_state))
        fringe = collections.deque()

        counter = itertools.count()

        init_node = Node(init_state, None, None, 0,
                          next(counter))
        fringe.append(init_node)

        time_thread = TimeThread(position=self.progress_bar_position,
            disable=not self.show_progress,
            time_limit_in_sec=time_limit_in_sec)
        time_thread.start()

        new_pos_literals = set()
        while len(fringe) > 0 \
            and (skip_uncovered_check or len(uncovered_actions) > 0) \
            and time_thread.is_alive():

            node = fringe.popleft()

            if node.state in visited:
                continue

            visited.add(node.state)
            depth_list = depth_state_dict.setdefault(node.depth, [])
            depth_list.append(node)
            actions = list(ground_actions_dict.keys())
            random.shuffle(actions)
            for a in actions:

                action = ground_actions_dict[a]
                name = a.predicate.name
                min_depth = action_depth.setdefault(name, float("inf"))

                assert action.is_optimized
                if action.preconds.holds(node.state):

                    uncovered_actions.discard(a)
                    covered_actions.add(a)
                    action_depth[name] = min(min_depth, node.depth)

                    successors = self._get_successors(ground_actions_dict,
                                                      a, node,
                                                      counter)
                    fringe.extend(successors)
                elif can_try_making_applicable \
                    and random.random() < try_applicable_chance \
                    and self._can_make_action_applicable(
                        node.state, action,
                        init_state_neg_literals,
                        constraints=constraints):

                    self._make_action_applicable(new_pos_literals,
                                                 action,
                                                 init_state_neg_literals)
                    uncovered_actions.discard(a)
                    covered_actions.add(a)
                    action_depth[name] = min(min_depth, node.depth)

        time_thread.stop()
        time_thread.join()

        return new_pos_literals


    def generate_init_state(self, domain, objects,
                            seed_init_state=set(),
                            constraints=[],
                            time_limit_in_sec=60):

        ground_actions_dict, action_depth = \
            self.def_get_action_dicts(domain, objects)

        uncovered_actions = set(ground_actions_dict.keys())
        visited_states = set()
        covered_actions = set()

        domain.initialize_non_fluents()
        non_fluents = domain.non_fluents.union(
            domain.only_negated_fluents)

        init_state = set(seed_init_state)
        init_state_neg_literals = set(seed_init_state)
        while len(uncovered_actions) > 0:

            a = random.sample(uncovered_actions, k=1)[0]
            uncovered_actions.remove(a)

            # This action already executable from somewhere.
            if action_depth[a.predicate.name] != float("inf"):
                continue

            action = ground_actions_dict[a]
            assert action.is_optimized

            if action.preconds.holds(init_state):

                action_depth[a.predicate.name] = 0
            elif self._can_make_action_applicable(init_state,
                action,
                init_state_neg_literals,
                constraints=constraints):

                self._make_action_applicable(init_state, action,
                                             init_state_neg_literals)
                action_depth[a.predicate.name] = 0

                # New init state, refresh the visited list.
                visited_states = set()

            # Always remove one action.
            uncovered_actions.discard(a)

            pos_literals = self._perform_walk(
                init_state, ground_actions_dict,
                visited_states, uncovered_actions,
                action_depth, covered_actions,
                {},
                init_state_neg_literals,
                time_limit_in_sec=time_limit_in_sec,
                can_try_making_applicable=True,
                constraints=constraints)

            if len(pos_literals) > 0:

                visited_states = set()
                init_state.update(pos_literals)

        print("Init state size:", len(init_state))


        return init_state

    def generate_goal_state(self, domain, objects, init_state,
                       time_limit_in_sec=60):

        ground_actions_dict, action_depth = \
            self.def_get_action_dicts(domain, objects)

        visited_states = set()
        depth_state_dict = {}
        self._perform_walk(init_state, ground_actions_dict,
                           visited_states, set(),
                           action_depth, set(),
                           depth_state_dict, set(),
                           time_limit_in_sec=time_limit_in_sec,
                           skip_uncovered_check=True)

        for name, depth in action_depth.items():
            print("%s executable at depth %s" % (name, depth))

        return depth_state_dict, action_depth