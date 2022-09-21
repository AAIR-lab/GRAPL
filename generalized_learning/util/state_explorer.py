'''
Created on Oct 20, 2020

@author: anonymous
'''

from collections import deque
import random

import tqdm

from abstraction.domain import AbstractDomain


def explore_state_space(problem, MAX_STATES_TO_EXPLORE=500000,
                        disable_progress_bar=False):

    progress_bar = tqdm.tqdm(unit=" states", disable=disable_progress_bar)
    visited = set()
    fringe = deque()

    fringe.append(problem.get_initial_state())
    visited.add(problem.get_initial_state())

    while len(fringe) > 0 and len(visited) < MAX_STATES_TO_EXPLORE:

        progress_bar.update(1)
        current_state = fringe.popleft()

        for action in problem.get_applicable_actions(current_state):

            successors = problem.get_successors(action, current_state)

            for next_state, _ in successors:

                if next_state not in visited:

                    visited.add(next_state)
                    fringe.append(next_state)

    progress_bar.close()
    return visited


def random_walk(problem, MAX_WALK_LENGTH=10000,
                disable_progress_bar=False):

    progress_bar = tqdm.tqdm(total=MAX_WALK_LENGTH,
                             unit=" states", disable=disable_progress_bar)
    visited = set()

    visited.add(problem.get_initial_state())

    current_state = problem.get_initial_state()
    visited.add(current_state)
    for _ in range(MAX_WALK_LENGTH):

        progress_bar.update(1)

        actions = problem.get_applicable_actions(current_state)
        action = random.choice(actions)

        current_state, _, _ = problem.apply_action(action, current_state)
        visited.add(current_state)

        if problem.is_goal_satisfied(current_state):

            current_state = problem.get_initial_state()

    progress_bar.close()
    return visited


def sample_transitions(problem, NUM_TRANSITIONS_TO_SAMPLE=500,
                       disable_progress_bar=False):

    progress_bar = tqdm.tqdm(total=NUM_TRANSITIONS_TO_SAMPLE,
                             unit=" transitions",
                             disable=disable_progress_bar)

    current_state = problem.get_initial_state()

    transitions = set()
    for _ in range(NUM_TRANSITIONS_TO_SAMPLE):

        progress_bar.update(1)

        actions = problem.get_applicable_actions(current_state)
        if len(actions) == 0:

            break

        action = random.choice(actions)

        next_state, _, _ = problem.apply_action(action, current_state)
        transitions.add((current_state, next_state))

        current_state = next_state

    progress_bar.close()
    return transitions
