from exploration import exploration_utils
import collections
import time

def try_actions(simulator, state, actions):

    simulator.set_state(state)

    total_steps = 0
    while len(actions) > 0:

        action = actions.pop()
        next_state, _, _, _ = simulator.step(action, True)
        total_steps += 1
        execution_status = simulator.get_step_execution_status()

        if execution_status:

            return total_steps, action
        else:

            assert state.literals == next_state.literals

    return total_steps, None

def populate_fringe(fringe, state, visited,
                    actions, action_cache, learned_model):

    if state.literals in visited:

        return

    visited.add(state.literals)
    learned_actions, unlearned_actions = \
        exploration_utils.get_action_lists(
            actions,
            state,
            action_cache,
            learned_model)

    fringe.append((state, unlearned_actions))
    for learned_action in learned_actions:
        fringe.append((state, learned_action))

def intelligent_bfs(simulator,
                action_cache,
                learned_model,
                max_steps=float("inf"),
                max_time=float("inf"),
                monte_carlo_steps=5):

    sim_state = simulator.save_state()

    actions = simulator.get_actions()
    fringe = collections.deque()
    visited = set()

    populate_fringe(fringe,
                    simulator.get_state(),
                    visited,
                    actions,
                    action_cache,
                    learned_model)

    populate_fringe(fringe,
                    simulator.get_initial_state(),
                    visited,
                    actions,
                    action_cache,
                    learned_model)

    total_steps = 0
    end_time = time.time() + max_time

    while len(fringe) > 0 \
        and total_steps < max_steps \
        and time.time() < end_time:

        state, action = fringe.popleft()
        if isinstance(action, list):
            steps, action = try_actions(simulator, state, action)
            total_steps += steps

            if action is not None:

                action_cache[action.predicate.name] = state

                simulator.restore_state(*sim_state)
                return total_steps, action.predicate.name
        else:

            for _ in range(monte_carlo_steps):
                simulator.set_state(state)
                next_state, _, _, _ = simulator.step(action, True)
                total_steps += 1
                assert simulator.get_step_execution_status()

                populate_fringe(fringe,
                                next_state,
                                visited,
                                actions,
                                action_cache,
                                learned_model)

    simulator.restore_state(*sim_state)
    return total_steps, None


