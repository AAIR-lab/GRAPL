#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import os
import pickle
import random
import sys

from config import *
from lattice import State
from utils import get_plan
from . import gym_to_iaa_dict, problem_file_header

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
import sim.sim_agent as sa


class SimAgent(sa.SimAgent):

    def __init__(self, n, n_gen, render=False, load_random_from_file=False, save_random_states=False):

        custom_fd = domain_dir_gym + '/sokoban/problem' + str(99) + '.pddl'
        super().__init__("PDDLEnvSokoban-v0", "sokoban", custom_fd)
        self.random_states = []
        self.state_gen_counter = 0
        self.random_file_name = '../random_states/random_sokoban.pkl'

        if not load_random_from_file:
            self.generate_random_init_states(n, render)
            _ign = self.get_more_random_states(n_gen, save=save_random_states)
        else:
            with open(self.random_file_name, 'rb') as f:
                self.random_states = pickle.load(f)

    def set_initial_and_goal_state(self, init_state_, render=False, partial_check=False, already_checked=False):
        can_be_set = False
        if already_checked:
            can_be_set = True
        elif self.validate_state(init_state_, partial_check):
            can_be_set = True

        if can_be_set:
            init_state = init_state_.state
            object_file_text = ''
            init_state_text = ''
            for pred, args in init_state.items():
                if len(args) != 0:
                    try:
                        for i, arg in enumerate(args):
                            arg = list(arg)
                            init_state_text += '(' + pred + ' '
                            for j, a in enumerate(arg):
                                init_state_text += a + ' '
                                temp_object_file_text = '\t ' + a + ' - ' + self.predicates[str.lower(pred)][j] + '\n'
                                if temp_object_file_text not in object_file_text:
                                    object_file_text += temp_object_file_text
                            init_state_text += ')\n'
                    except Exception as e:
                        print("setting Initial state Exception!" + str(e))
            if VERBOSE:
                print("Writing to file:" + self.custom_fd)
            with open(self.custom_fd, 'w') as f:
                header = problem_file_header(self.pddl_domain_name, object_file_text, init_state_text)
                f.write(header)
                goal_text = ''
                for thing in init_state_.objects['thing']:
                    if 'stone' in thing:
                        goal_text += '(at-goal ' + thing + ')\n'
                goal_text = goal_text.replace('\n', ' ')
                f.write(goal_text)
                f.write(')))')

            if self.test_reset():
                self.load_custom_problemfile()
                self.set_action_mapping()
                if render:
                    _ = self.env.render()
                return True

        else:
            print("Invalid Initial state created!:")
        return False

    def get_more_random_states(self, max_states, save=False, random_walk=False):
        extra_states = []
        ret_state = []
        if len(self.random_states) != 0:
            num_per_random_init_state = max_states
            state_gen_index = 0
            for init_state_ in self.random_states:
                if state_gen_index < self.state_gen_counter:
                    state_gen_index += 1
                    continue

                self.state_gen_counter += 1
                init_state = copy.deepcopy(init_state_)
                i = 0
                success = self.set_initial_and_goal_state(init_state, False)
                current_state = copy.deepcopy(self.state)
                plan = get_plan(self.env.domain.domain_fname, self.custom_fd)
                if success:
                    if len(plan) == 0 or random_walk:
                        # do random walks if problem can't be solved
                        if success:
                            j = 0
                            while i <= num_per_random_init_state:
                                if j > 100:
                                    print("preempting")
                                    break
                                action = self.env.action_space.sample()
                                next_state, _, _, _ = self.env.step(action)
                                if next_state == current_state or next_state in extra_states:
                                    j += 1
                                else:
                                    j = 0
                                    current_state = copy.deepcopy(next_state)
                                    extra_states.append(current_state)
                                    if VERBOSE:
                                        print("added")
                                    i += 1
                    else:
                        for step in plan:
                            action = self.action_mapping[step.split('|')[-1]]
                            next_state, _, _, _ = self.env.step(action)
                            if next_state not in extra_states:
                                current_state = copy.deepcopy(next_state)
                                extra_states.append(current_state)
                                if VERBOSE:
                                    print("added")
                else:
                    print("This shouldn't be")

            for state in extra_states:
                temp_state = gym_to_iaa_dict(state)
                temp_state.state['move'] = [('dir-down',), ('dir-left',), ('dir-right',), ('dir-up',)]
                self.random_states.append(temp_state)
                ret_state.append(temp_state)

            if save:
                with open(self.random_file_name, "wb") as f:
                    pickle.dump(self.random_states, f)
            if VERBOSE:
                print("Done adding more states")
        else:
            print("Not enough original init states")

        return ret_state

    @staticmethod
    def validate_state(init_state_, partial_check=False):
        """
        conditions to check:
            cells which have stone cannot have player and vice-versa
            cells which have either player or stone cannot be clear
            move should have all 4
            correct values for move-dir

        Adding partial check to be checked from inside update_pal_tuple_ordering

        """

        all_keys = ['clear', 'move', 'move-dir', 'at', 'is-player', 'is-goal', 'is-nongoal', 'is-stone']
        all_keys_partial = list()
        if partial_check:
            all_keys_partial = [{'clear', 'move-dir', 'at', 'is-player', 'is-goal', 'move', 'is-stone'},
                                {'clear', 'move-dir', 'at', 'is-player', 'is-goal', 'is-stone'},
                                {'clear', 'move-dir', 'at', 'is-player', 'is-goal', 'is-nongoal', 'is-stone'},
                                {'clear', 'move', 'move-dir', 'at', 'is-player', 'is-goal', 'is-nongoal', 'is-stone'}]
        init_state = copy.deepcopy(init_state_.state)
        if not partial_check and len(set(all_keys).difference(set(init_state.keys()))) != 0:
            print("Invalid because not all predicates present in state")
            return False
        elif partial_check and (set(init_state.keys()) not in all_keys_partial):
            print("Invalid because not all predicates present in state")
            return False

        extra_clear = copy.deepcopy(init_state['clear'])
        for i in init_state['at']:
            extra_clear.append(tuple([i[-1]]))

        for i in init_state['at']:
            if tuple([i[-1]]) in init_state['clear']:
                print("Invalid because:" + str(tuple([i[-1]])) + ' is in clear')
                return False
            if 'stone' in tuple([i[0]]) and tuple([i[0]]) in init_state['is-goal']:
                if 'at-goal' not in init_state.keys():
                    print("Invalid because stone at goal and at-goal not present in state")
                    return False
                else:
                    present = False
                    for k, v in init_state['at-goal'].items():
                        if v[0] == i[0] and v[1] == i[1]:
                            present = True
                            break
                    if not present:
                        print("Invalid because stone at goal and not in at-goal")
                        return False

        for i in init_state['move-dir']:
            if tuple([i[0]]) not in extra_clear:
                print("Invalid because:" + str(tuple([i[0]])) + ' is not in extra clear')
                return False
            if tuple([i[1]]) not in extra_clear:
                print("Invalid because:" + str(tuple([i[1]])) + ' is not in extra clear')
                return False
        return True

    @staticmethod
    def action_can_be_applied(state, action):
        if type(state) == set:
            temp_state = gym_to_iaa_dict(state)
        else:
            temp_state = state
        var = action.split('|')
        if 'move' in action:
            from_loc = var[2]
            to_loc = var[3]
            dir_ = var[4]
            if tuple(['player-01', from_loc]) not in temp_state.state['at']:
                return False
            if tuple([to_loc]) not in temp_state.state['clear']:
                return False
            if tuple([from_loc, to_loc, dir_]) not in temp_state.state['move-dir']:
                return False
        else:
            player_loc = var[3]
            from_loc = var[4]
            to_loc = var[5]
            dir_ = var[6]
            stone = var[2]
            if tuple(['player-01', player_loc]) not in temp_state.state['at']:
                return False
            if tuple([stone, from_loc]) not in temp_state.state['at']:
                return False
            if tuple([to_loc]) not in temp_state.state['clear']:
                return False
            if tuple([from_loc, to_loc, dir_]) not in temp_state.state['move-dir']:
                return False
            if tuple([player_loc, from_loc, dir_]) not in temp_state.state['move-dir']:
                return False
            if 'nongoal' not in var[0]:
                if 'is-goal' not in temp_state.state.keys() or tuple([to_loc]) not in temp_state.state['is-goal']:
                    return False
            else:
                if 'is-goal' not in temp_state.state.keys() or tuple([to_loc]) in temp_state.state['is-goal']:
                    return False

        return True

    def set_initial_state(self, init_state_, render=False, partial_check=False, already_checked=False):
        can_be_set = False
        if already_checked:
            can_be_set = True
        elif self.validate_state(init_state_, partial_check):
            can_be_set = True

        if can_be_set:
            init_state = init_state_.state
            object_file_text = ''
            init_state_text = ''
            for pred, args in init_state.items():
                if len(args) != 0:
                    try:
                        for i, arg in enumerate(args):
                            arg = list(arg)
                            init_state_text += '(' + pred + ' '
                            for j, a in enumerate(arg):
                                init_state_text += a + ' '
                                temp_object_file_text = '\t ' + a + ' - ' + self.predicates[str.lower(pred)][j] + '\n'
                                if temp_object_file_text not in object_file_text:
                                    object_file_text += temp_object_file_text
                            init_state_text += ')\n'
                    except Exception as e:
                        print("setting Initial state Exception!" + str(e))
            if VERBOSE:
                print("Writing to file:" + self.custom_fd)
            with open(self.custom_fd, 'w') as f:
                f.write('(define (problem ' + self.pddl_domain_name + ') \n (:domain ' +
                        self.pddl_domain_name + ')\n (:objects \n')
                if len(object_file_text) != 0:
                    f.write(object_file_text)
                f.write(')\n (:init \n')
                f.write(init_state_text)
                f.write(')\n(:goal (and' + ' ')
                f.write(init_state_text)
                f.write(')))')

            if self.test_reset():
                self.load_custom_problemfile()
                self.set_action_mapping()
                if render:
                    _ = self.env.render()
                return True

        else:
            print("Invalid Initial state created!:")

        return False

    def get_pddl_details(self):
        details = dict()
        details['gym_domain_file'] = self.env.domain.domain_fname
        details['gym_problem_file'] = self.problem_file_loaded
        return details

    def set_action_mapping(self):
        self.action_mapping = {}
        for action in list(self.env.action_space.all_ground_literals()):
            self.action_mapping[action.pddl_str()[1:-1].split(' ')[-1]] = action

    def run_query(self, query, partial_check=False):

        already_checked = False
        if partial_check and len(query['plan']) == 1:
            if self.action_can_be_applied(query['init_state'], query['plan'][0]):
                already_checked = True

        success = self.set_initial_state(query['init_state'], False, partial_check, already_checked=already_checked)
        renders = []

        if success:
            if VERBOSE:
                print("Running plan!")
            state = self.state
            j = 0
            plan = query['plan']
            renders.append(self.env.render())
            for k, step in enumerate(plan):
                action = self.action_mapping[step.split('|')[-1]]
                if VERBOSE:
                    print("Executing action: " + str(action))
                if self.action_can_be_applied(state, step):
                    new_state, reward, done, info = self.env.step(action)
                    changes = set(state) - set(new_state)
                    if len(changes) != 0:
                        state = new_state
                        renders.append(self.env.render())
                        j += 1
                    else:
                        print("No change in state,!: " + str(action) + "Invalid Action")
                        return False, j, self.state_rep_convert(state), renders
                else:
                    print("Invalid Action!: " + str(action))
                    return False, j, self.state_rep_convert(state), renders
            if VERBOSE:
                print("Executed full plan!")
            return success, j, self.state_rep_convert(state), renders
        else:
            print("Error setting initial state!")
            return success, 0, query['init_state'], renders

    def generate_random_init_states(self, n, render):
        # make an lxb environment and add blocks surrounding it
        sample_init_state = {
            'at': set(),
            'clear': set(),
            'is-stone': set(),
            'is-goal': set(),
            'is-nongoal': set(),
            'is-player': [('player-01',)],
            'move': [('dir-down',), ('dir-left',), ('dir-right',), ('dir-up',)],
            'move-dir': set(),
            'at-goal': set()
        }

        self.random_states = []
        s = 0
        while s <= n:
            l = 4
            b = 4
            num_stones = 1
            num_goals = num_stones + 0
            num_blocked_cells = random.choice(range(int(3 * l * b / 10)))
            all_cells = list(itertools.product(range(1, l), range(1, b)))
            blocked_cells = random.sample(all_cells, num_blocked_cells)
            unblocked_cells = list(set(all_cells).difference(set(blocked_cells)))
            stone_positions = random.sample(unblocked_cells, num_stones)
            goal_positions = random.sample(unblocked_cells, num_goals)
            player_location = [random.choice(
                list(set(unblocked_cells).difference(set(stone_positions))))]
            init_state = copy.deepcopy(sample_init_state)
            init_state['at'].add(('player-01',
                                  'pos-' + "{0:0=2d}".format(player_location[0][0]) + '-' + "{0:0=2d}".format(
                                      player_location[0][1])))

            for i, pos in enumerate(stone_positions):
                stone = 'stone-' + "{0:0=2d}".format(i + 1)
                loc = 'pos-' + "{0:0=2d}".format(pos[0]) + '-' + "{0:0=2d}".format(pos[1])
                init_state['is-stone'].add(tuple([stone]))
                init_state['at'].add(tuple([stone, loc]))
                if pos in goal_positions:
                    init_state['at-goal'].add(tuple([stone]))

            for pos in all_cells:
                location = 'pos-' + "{0:0=2d}".format(pos[0]) + '-' + "{0:0=2d}".format(pos[1])
                location_left = 'pos-' + "{0:0=2d}".format(pos[0] - 1) + '-' + "{0:0=2d}".format(pos[1])
                location_right = 'pos-' + "{0:0=2d}".format(pos[0] + 1) + '-' + "{0:0=2d}".format(pos[1])
                location_up = 'pos-' + "{0:0=2d}".format(pos[0]) + '-' + "{0:0=2d}".format(pos[1] - 1)
                location_down = 'pos-' + "{0:0=2d}".format(pos[0]) + '-' + "{0:0=2d}".format(pos[1] + 1)

                left_cell = (pos[0] - 1, pos[1])
                right_cell = (pos[0] + 1, pos[1])
                up_cell = (pos[0], pos[1] - 1)
                down_cell = (pos[0], pos[1] + 1)

                if pos in goal_positions:
                    init_state['is-goal'].add(tuple([location]))
                else:
                    init_state['is-nongoal'].add(tuple([location]))
                if pos in unblocked_cells:
                    if pos not in player_location and pos not in stone_positions:
                        init_state['clear'].add(tuple([location]))
                    if left_cell in unblocked_cells or left_cell in stone_positions or left_cell in player_location:
                        init_state['move-dir'].add(tuple([location, location_left, 'dir-left']))
                    if right_cell in unblocked_cells or right_cell in stone_positions or right_cell in player_location:
                        init_state['move-dir'].add(tuple([location, location_right, 'dir-right']))
                    if up_cell in unblocked_cells or up_cell in stone_positions or up_cell in player_location:
                        init_state['move-dir'].add(tuple([location, location_up, 'dir-up']))
                    if down_cell in unblocked_cells or down_cell in stone_positions or down_cell in player_location:
                        init_state['move-dir'].add(tuple([location, location_down, 'dir-down']))

            for k in init_state.keys():
                init_state[k] = list(init_state[k])

            test_state = State(init_state, {})
            success = self.set_initial_state(test_state, render=render)
            if success:
                state_objects = dict()
                state_objects['direction'] = ['dir-down', 'dir-left', 'dir-right', 'dir-up']
                state_objects['thing'] = ['player-01']
                state_objects['location'] = []
                for k in range(num_stones):
                    state_objects['thing'].append('stone-' + "{0:0=2d}".format(k + 1))
                for pos in all_cells:
                    location = 'pos-' + "{0:0=2d}".format(pos[0]) + '-' + "{0:0=2d}".format(pos[1])
                    state_objects['location'].append(location)

                if len(init_state['at-goal']) == 0:
                    del init_state['at-goal']
                t = State(init_state, state_objects)
                if t not in self.random_states:
                    self.random_states.append(t)
                    s += 1
            else:
                print("incorrect random state!")

    def bootstrap_model(self):
        """
        We check accuracy with PDDLGym's model.
        Since PDDLGym adds spurious predicates for its internal use,
        we must add them manually. This is independent of AIA as these
        are extra changes PDDLGym makes to make domain work.
        """
        abs_preds_test = {'is-player|0': 0, 'is-stone|0': 0, 'is-player|1': 0,
                          'is-stone|1': 0, 'move|3': 0, 'move|5': 0}
        abs_actions_test = {'move': {'is-player|0': [Literal.POS, Literal.ABS],
                                     'is-stone|0': [Literal.ABS, Literal.ABS],
                                     'move|3': [Literal.POS, Literal.ABS]},
                            'push-to-nongoal': {'is-player|0': [Literal.POS, Literal.ABS],
                                                'is-player|1': [Literal.ABS, Literal.ABS],
                                                'is-stone|0': [Literal.ABS, Literal.ABS],
                                                'is-stone|1': [Literal.POS, Literal.ABS],
                                                'move|5': [Literal.POS, Literal.ABS]},
                            'push-to-goal': {'is-player|0': [Literal.POS, Literal.ABS],
                                             'is-player|1': [Literal.ABS, Literal.ABS],
                                             'is-stone|0': [Literal.ABS, Literal.ABS],
                                             'is-stone|1': [Literal.POS, Literal.ABS],
                                             'move|5': [Literal.POS, Literal.ABS]}}
        pal_tuples_fixed = [('move', 'is-player|0', Location.PRECOND),
                            ('move', 'is-player|0', Location.EFFECTS),
                            ('move', 'is-stone|0', Location.PRECOND),
                            ('move', 'is-stone|0', Location.EFFECTS),
                            ('move', 'move|3', Location.PRECOND),
                            ('move', 'move|3', Location.EFFECTS),
                            ('push-to-nongoal', 'is-player|0', Location.PRECOND),
                            ('push-to-nongoal', 'is-player|0', Location.EFFECTS),
                            ('push-to-nongoal', 'is-player|1', Location.PRECOND),
                            ('push-to-nongoal', 'is-player|1', Location.EFFECTS),
                            ('push-to-nongoal', 'is-stone|0', Location.PRECOND),
                            ('push-to-nongoal', 'is-stone|0', Location.EFFECTS),
                            ('push-to-nongoal', 'is-stone|1', Location.PRECOND),
                            ('push-to-nongoal', 'is-stone|1', Location.EFFECTS),
                            ('push-to-nongoal', 'move|5', Location.PRECOND),
                            ('push-to-nongoal', 'move|5', Location.EFFECTS),
                            ('push-to-goal', 'is-player|0', Location.PRECOND),
                            ('push-to-goal', 'is-player|0', Location.EFFECTS),
                            ('push-to-goal', 'is-player|1', Location.PRECOND),
                            ('push-to-goal', 'is-player|1', Location.EFFECTS),
                            ('push-to-goal', 'is-stone|0', Location.PRECOND),
                            ('push-to-goal', 'is-stone|0', Location.EFFECTS),
                            ('push-to-goal', 'is-stone|1', Location.PRECOND),
                            ('push-to-goal', 'is-stone|1', Location.EFFECTS),
                            ('push-to-goal', 'move|5', Location.PRECOND),
                            ('push-to-goal', 'move|5', Location.EFFECTS)]

        return abs_preds_test, abs_actions_test, pal_tuples_fixed
