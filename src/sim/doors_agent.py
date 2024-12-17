#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
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
        custom_fd = domain_dir_gym + '/doors/problem' + str(99) + '.pddl'
        super().__init__("PDDLEnvDoors-v0", "doors", custom_fd)
        self.random_states = []
        self.state_gen_counter = 0
        self.random_file_name = '../random_states/random_doors.pkl'
        if not load_random_from_file:
            self.generate_random_init_states(n, render)
            _ign = self.get_more_random_states(n_gen, save=save_random_states)
        else:
            with open(self.random_file_name, 'rb') as f:
                self.random_states = pickle.load(f)
        if VERBOSE:
            print("done")

    def set_initial_and_goal_state(self, init_state_, render=False, partial_check=False, already_checked=False):
        can_be_set = False
        if already_checked:
            can_be_set = True
        elif self.validate_state(init_state_, partial_check):
            can_be_set = True

        if can_be_set:
            object_file_text = ''
            init_state_text = ''
            init_state = copy.deepcopy(init_state_.state)
            objects = copy.deepcopy(init_state_.objects)
            init_state['MoveTo'] = []
            init_state['Pick'] = []
            for loc in objects['location']:
                init_state['MoveTo'].append(tuple([loc]))
            for key in objects['key']:
                init_state['Pick'].append(tuple([key]))

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
                for room in init_state_.objects['room']:
                    if tuple([room]) not in init_state_.state['unlocked']:
                        for locinroom in init_state_.state['locinroom']:
                            if list(locinroom)[1] == room:
                                goal_text += '(at ' + str(list(locinroom)[0]) + ')\n'
                                break
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
        initial_n = len(self.random_states)
        if len(self.random_states) != 0 and len(self.random_states) < max_states + initial_n:
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
                if VERBOSE:
                    print(plan)
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
                        print("Plan")
                        for step in plan:
                            step = step.split('|')
                            step = step[0] + ' ' + step[2]
                            action = self.action_mapping[step.strip(' ')]
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
                temp_state.state['moveto'] = []
                temp_state.state['pick'] = []
                for key in temp_state.objects['key']:
                    temp_state.state['pick'].append(tuple([key]))
                for loc in temp_state.objects['location']:
                    temp_state.state['moveto'].append(tuple([loc]))
                assert len(temp_state.state['at']) == 1, "Multiple AT!!"
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

    def generate_random_init_states(self, n, render):
        sample_init_state = {
            'at': [],
            'unlocked': [],
            'locinroom': [],
            'keyforroom': [],
            'keyat': [],
            'moveto': [],
            'pick': [],
        }
        i = 0
        max_keys = 3
        num_keys = random.choice(range(1, max_keys))
        num_rooms = num_keys + 2
        while i <= n:
            # locations = []
            length = 4
            width = 4
            keys = []
            rooms = []
            locations = []
            for l in range(length):
                for w in range(width):
                    locations.append('loc-' + str(l) + '-' + str(w))
            for k in range(num_keys):
                keys.append('key-' + str(k))
            for r in range(num_rooms):
                rooms.append('room-' + str(r))
            init_state = copy.deepcopy(sample_init_state)
            # place keys randomly
            init_state['unlocked'].append(tuple([rooms[0]]))
            key_locations = random.sample(locations, len(keys))

            for key, key_loc in zip(keys, key_locations):
                init_state['keyat'].append(tuple([key, key_loc]))
            # place rooms randomly
            room_locations = random.sample(locations, len(rooms))
            j = 0
            for loc in locations:
                if loc not in room_locations:
                    init_state['locinroom'].append(tuple([loc, 'room-0']))
                else:
                    init_state['locinroom'].append(tuple([loc, rooms[j]]))
                    j += 1

            # put player in random non-key place
            at_loc = random.choice(list(set(locations).difference(room_locations)))
            init_state['at'].append(tuple([at_loc]))
            # assign keys to rooms
            random.shuffle(keys)
            temp_rooms = rooms[1:]
            random.shuffle(temp_rooms)
            random.shuffle(locations)

            for key, room in list(zip(keys, temp_rooms)):
                init_state['keyforroom'].append(tuple([key, room]))

            state_objects = {
                'location': [],
                'key': [],
                'room': []
            }

            for loc in locations:
                state_objects['location'].append(loc)
                init_state['moveto'].append(tuple([loc]))
            for key in keys:
                state_objects['key'].append(key)
                init_state['pick'].append(tuple([key]))

            for room in rooms:
                state_objects['room'].append(room)
            test_state = State(init_state, state_objects)
            assert len(test_state.state['at']) == 1, "Multiple AT!!"
            success = self.set_initial_state(test_state, render=render)
            if success:
                print("Found")
                i += 1
                self.random_states.append(test_state)
            else:
                print("Failed")

    def set_initial_state(self, init_state_, render=False, partial_check=False, already_checked=False):
        can_be_set = False
        if already_checked:
            can_be_set = True
        elif self.validate_state(init_state_, partial_check):
            can_be_set = True

        if can_be_set:
            object_file_text = ''
            init_state_text = ''
            init_state = copy.deepcopy(init_state_.state)
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
                        print("setting Initial state Exception: ", e)

            if VERBOSE:
                print("Writing to file:" + self.custom_fd)
            with open(self.custom_fd, 'w') as f:
                f.write('(define (problem ' + self.pddl_domain_name +
                        ') \n (:domain ' + self.pddl_domain_name + ')\n (:objects \n')
                if len(object_file_text) != 0:
                    f.write(object_file_text)
                f.write(')\n (:init \n')
                f.write(init_state_text)
                f.write(')\n(:goal (and' + ' ')
                init_state_text = init_state_text.replace('\n', ' ')
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
            self.action_mapping[action.pddl_str()[1:-1]] = action

    @staticmethod
    def action_can_be_applied(state, action):
        if type(state) == set:
            temp_state = gym_to_iaa_dict(state)
        else:
            temp_state = state
        var = action.split('|')
        if ('at' in temp_state.state.keys() and len(temp_state.state['at']) > 1) or \
                ('keyforroom' in temp_state.state.keys() and len(temp_state.state['keyforroom']) > 1):
            return False
        if 'move' in action:
            start_loc = var[1]
            end_loc = var[2]
            room = var[3]
            if 'at' not in temp_state.state.keys() or \
                    tuple([start_loc]) not in temp_state.state['at']:
                return False
            if 'locinroom' not in temp_state.state.keys() or \
                    tuple([end_loc, room]) not in temp_state.state['locinroom']:
                return False
            if 'unlocked' not in temp_state.state.keys() or \
                    tuple([room]) not in temp_state.state['unlocked']:
                return False
        else:
            loc = var[1]
            key = var[2]
            room = var[3]
            if 'keyforroom' not in temp_state.state.keys() or \
                    tuple([key, room]) not in temp_state.state['keyforroom']:
                return False
            if 'keyat' not in temp_state.state.keys() or \
                    tuple([key, loc]) not in temp_state.state['keyat']:
                return False
            if 'at' not in temp_state.state.keys() or \
                    tuple([loc]) not in temp_state.state['at']:
                return False

        return True

    def run_query(self, query, partial_check=False):

        """
        already_checked = False
        if partial_check and len(query['plan'])==1:
            if self.action_can_be_applied(query['init_state'],query['plan'][0]):
                already_checked = True
        """
        success = self.set_initial_state(query['init_state'])
        renders = []

        if success:
            if VERBOSE:
                print("Running plan!")
            state = self.state
            j = 0
            plan = query['plan']
            renders.append(self.env.render())
            for k, step in enumerate(plan):
                step_ = step.split('|')
                step_ = step_[0] + ' ' + step_[2]
                action = self.action_mapping[step_.strip(' ')]
                if VERBOSE:
                    print("Executing action: " + str(action))
                if self.action_can_be_applied(state, step):
                    try:
                        new_state, reward, done, info = self.env.step(action)
                    except:
                        print("Rendering Error")
                    changes = set(state) - set(new_state)
                    if len(changes) != 0:
                        state = new_state
                        renders.append(self.env.render())
                        j += 1
                    else:
                        print("No change in state after executing action: " + str(action) + "! Invalid action")
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

    @staticmethod
    def validate_state(init_state_, partial_check=False):
        state = copy.deepcopy(init_state_.state)
        min_keys = ['at', 'unlocked', 'locinroom']
        diff_min = set(min_keys).difference(set(state.keys()))

        if len(diff_min) != 0:
            print("all keys not present")
            return False
        else:
            if tuple(['room-0']) not in state['unlocked']:
                print("room 0 locked")
                return False
        if 'locinroom' not in state.keys() or len(init_state_.objects['location']) != len(state['locinroom']):
            return False
        if 'moveto' not in state.keys() or len(init_state_.objects['location']) != len(state['moveto']):
            return False
        if 'pick' not in state.keys() or len(init_state_.objects['key']) != len(state['pick']):
            return False

        return True

    def bootstrap_model(self):
        """
        We check accuracy with PDDLGym's model.
        Since PDDLGym adds spurious predicates for its internal use,
        we must add them manually. This is independent of AIA as these
        are extra changes PDDLGym makes to make domain work.
        """
        abs_preds_test = {'moveto|0': 0, 'moveto|1': 0, 'pick|1': 0}
        abs_actions_test = {'moveto': {'moveto|0': [Literal.ABS, Literal.ABS],
                                       'moveto|1': [Literal.POS, Literal.ABS]},
                            'pick': {'moveto|0': [Literal.ABS, Literal.ABS],
                                     'pick|1': [Literal.POS, Literal.ABS]}}
        pal_tuples_fixed = [('moveto', 'moveto|0', Location.PRECOND),
                            ('moveto', 'moveto|0', Location.EFFECTS),
                            ('moveto', 'moveto|1', Location.PRECOND),
                            ('moveto', 'moveto|1', Location.EFFECTS),
                            ('pick', 'moveto|0', Location.PRECOND),
                            ('pick', 'moveto|0', Location.EFFECTS),
                            ('pick', 'pick|1', Location.PRECOND),
                            ('pick', 'pick|1', Location.EFFECTS)]

        return abs_preds_test, abs_actions_test, pal_tuples_fixed
