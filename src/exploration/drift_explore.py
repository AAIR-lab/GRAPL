import random
import collections

from exploration import exploration_utils

def run_step(simulator, actions, action_cache, learned_model,
             taboo_state_dict):

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
        next_state, _, _, _ = simulator.step(action, True)
        execution_status = simulator.get_step_execution_status()

        if execution_status:
            if state.literals not in taboo_state_set:
                return action, state, True
            else:
                return None, None, None
        else:
            assert state.literals == next_state.literals

    for action in learned_actions:

        taboo_state_set = taboo_state_dict.setdefault(
            action.predicate.name, set())
        next_state, _, _, _ = simulator.step(action, True)
        execution_status = simulator.get_step_execution_status()

        precondition_conformant, effect_conformant = \
            learned_model.is_transition_conformant(state, action, next_state,
                                                   execution_status)
        if not precondition_conformant:
            state_literals = action_cache.pop(action.predicate.name, None)
            if state_literals is not None:
                taboo_state_set.add(state_literals)

            if execution_status:
                return None, None, None
            else:
                assert state.literals == next_state.literals
        elif execution_status:

            if not effect_conformant:
                return action, state, False
            else:
                return None, None, None
        else:

            pass

    return False, None, None

def run_episode(simulator, actions, action_cache, learned_model,
                taboo_state_dict, horizon=40):

    _ = simulator.reset()
    actions = simulator.get_actions()
    for _ in range(horizon):

        a, state, learn_all = run_step(simulator, actions, action_cache,
                                      learned_model, taboo_state_dict)
        if a is False:
            break
        elif a is not None:

            return a, state, learn_all
        else:
            pass

    return None, None, None

def explore(simulator,
                action_cache,
                learned_model,
                taboo_state_dict={},
                total_tries=float("inf")):

    if learned_model is not None:
        learned_model = learned_model.flatten(with_copy=True)
        learned_model = learned_model.optimize(with_copy=False)

    actions = simulator.get_actions()

    sim_state = simulator.save_state()
    attempt = 0
    while attempt < total_tries:

        attempt += 1
        a, state, learn_all = run_episode(simulator, actions,
                                             action_cache, learned_model,
                                             taboo_state_dict)

        if a is not None:
            simulator.restore_state(*sim_state)
            return a, state, learn_all

    return None, None, None