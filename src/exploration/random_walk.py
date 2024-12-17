import random

from exploration import exploration_utils

def random_walk(simulator,
                action_cache,
                learned_model,
                horizon=40,
                num_tries=float("inf"),
                drift_mode=False,
                taboo_state_dict={}):

    total_steps = 0
    try_no = 0
    learned_model = learned_model.flatten(with_copy=True)
    learned_model = learned_model.optimize(with_copy=False)
    actions = simulator.get_actions()
    initial_state = simulator.get_initial_state()

    sim_state = simulator.save_state()

    while try_no < num_tries:

        simulator.set_state(initial_state)
        state = simulator.get_state()
        old_state = None
        h = 0

        while h < horizon:

            h += 1

            if old_state != state:

                learned_actions, unlearned_actions = \
                    exploration_utils.get_action_lists(
                        actions,
                        state,
                        action_cache,
                        learned_model)

            old_state = state

            # Try all unlearned actions first.
            while len(unlearned_actions) > 0:

                action = unlearned_actions.pop()
                taboo_state_set = taboo_state_dict.setdefault(
                    action.predicate.name, set())
                next_state, _, _, _ = simulator.step(action, True)
                total_steps += 1
                execution_status = simulator.get_step_execution_status()

                if execution_status:

                    if state.literals not in taboo_state_set:

                        action_cache[action.predicate.name] = state
                        simulator.restore_state(*sim_state)
                        return total_steps, action.predicate.name
                    else:
                        simulator.set_state(state)
                else:
                    assert state.literals == next_state.literals

            if len(learned_actions) == 0:

                h = horizon
            else:
                action = learned_actions.pop()
                state, _, _, _ = simulator.step(action, True)
                total_steps += 1
                execution_status = simulator.get_step_execution_status()

                if not execution_status and drift_mode:

                    taboo_state_set = taboo_state_dict.setdefault(
                        action.predicate.name, set())
                    taboo_state_set.add(
                        action_cache[action.predicate.name].literals)
                    action_cache.pop(action.predicate.name, None)
                    h = horizon
                else:
                    assert execution_status

    simulator.restore_state(*sim_state)
    return total_steps, None