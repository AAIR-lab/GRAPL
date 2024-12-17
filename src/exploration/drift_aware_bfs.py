import random
import collections

from exploration import exploration_utils

def try_unlearned_action(simulator,
                         taboo_state_set,
                         state,
                         action,
                         fringe,
                         monte_carlo_count=10):

    for _ in range(monte_carlo_count):

        simulator.set_state(state)
        next_state, _, _, _ = simulator.step(action, True)
        execution_status = simulator.get_step_execution_status()

        if execution_status:

            if state.literals not in taboo_state_set:
                return action, state, True
            else:
                fringe.append(next_state)
        else:
            assert state.literals == next_state.literals
            return None, None, None

    return None, None, None

def try_learned_action(simulator,
                       learned_model,
                         taboo_state_set,
                         state,
                         action,
                         fringe,
                         action_cache,
                         monte_carlo_count=10):

    for _ in range(monte_carlo_count):

        simulator.set_state(state)
        next_state, _, _, _ = simulator.step(action, True)
        execution_status = simulator.get_step_execution_status()

        precondition_conformant, effect_conformant = \
            learned_model.is_transition_conformant(state, action, next_state,
                                                   execution_status)
        if not precondition_conformant:

            state_literals = action_cache.pop(action.predicate.name, None)
            if state_literals is not None:
                taboo_state_set.add(state_literals)
            return False, False, False
        elif not execution_status:

            break
        elif not effect_conformant:

            return action, state, False
        else:
            fringe.append(next_state)

    return None, None, None


def explore(simulator,
                action_cache,
                learned_model,
                taboo_state_dict={}):

    if learned_model is not None:
        learned_model = learned_model.flatten(with_copy=True)
        learned_model = learned_model.optimize(with_copy=False)

    actions = simulator.get_actions()
    initial_state = simulator.get_initial_state()

    sim_state = simulator.save_state()
    fringe = collections.deque()
    fringe.append(initial_state)
    visited = set()

    while len(fringe) > 0:

        state = fringe.popleft()
        if state.literals in visited:
            continue

        simulator.set_state(state)
        visited.add(state.literals)
        state = simulator.get_state()

        learned_actions, unlearned_actions = \
            exploration_utils.get_action_lists(
                actions,
                state,
                action_cache,
                learned_model)

        for action in unlearned_actions:

            taboo_state_set = taboo_state_dict.setdefault(
                action.predicate.name, set())

            a, return_state, learn_all = try_unlearned_action(
                simulator,
                taboo_state_set,
                state,
                action,
                fringe)

            if a is not None:

                simulator.restore_state(*sim_state)
                return a, return_state, learn_all

        for action in learned_actions:

            taboo_state_set = taboo_state_dict.setdefault(
                action.predicate.name, set())

            a, return_state, learn_all = try_learned_action(
                simulator,
                learned_model,
                taboo_state_set,
                state,
                action,
                fringe,
                action_cache)

            if a == False:

                visited.clear()
                fringe.clear()
                fringe.append(initial_state)
                break
            elif a is not None:

                simulator.restore_state(*sim_state)
                return a, return_state, learn_all

    simulator.restore_state(*sim_state)
    return None, None, None