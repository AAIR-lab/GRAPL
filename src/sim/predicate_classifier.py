#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict

from config import *
from sim import *
from lattice import State
import cv2


def create_custom_query_init_state(init_state):
    w = init_state['width']
    h = init_state['height']
    walls = init_state['walls']
    stones = init_state['stones']
    goals = init_state['goals']
    player_loc = init_state['player']
    state = State({}, {})
    keys = ["at", "clear", "is-goal", "is-stone", "is-goal", "is-nongoal", "is-player"]
    for key in keys:
        state.state[key] = []

    stone_num = 0
    for x in range(1, w + 1):
        for y in range(1, h + 1):
            loc = 'pos-' + "{0:0=2d}".format(x) + '-' + "{0:0=2d}".format(y)
            state.objects[loc] = "location"
            if loc in stones:
                stone_num += 1
                stone = "stone-" + "{0:0=2d}".format(stone_num)
                state.objects[stone] = "thing"
                state.state['at'].append(tuple([stone, loc]))
                state.state['is-stone'].append(tuple([stone]))
            elif loc in player_loc:
                state.state['at'].append(tuple(["player-01", loc]))
            elif loc not in walls:
                state.state['clear'].append(tuple([loc]))
            if loc in goals:
                state.state['is-goal'].append(tuple([loc]))
            else:
                state.state['is-nongoal'].append(tuple([loc]))
            state.objects[loc] = "location"
    state.objects['player-01'] = "thing"
    state.state["is-player"] = [tuple(["player-01"])]
    return state


class SokobanState(object):
    def __init__(self):
        print("Creating Sokoban State")
        self.ref_colors = OrderedDict({
            "player":[155.04545454545456, 131.84185606060606, 130.67140151515153],
			"wall":[143.87890625, 123.443359375, 156.7744140625],
			"goal":[208.541015625, 128.2021484375, 146.419921875],
			"clear":[215.638671875, 124.810546875, 146.9892578125],
			"stone-01":[151.3031746031746, 140.01666666666668, 166.56428571428572],
			"goal-at-stone": [156.53015873015872, 101.7611111111111, 156.62777777777777]
        })

    @staticmethod
    def pos_strings_is_adjacent(pos1, pos2):
        pos1 = pos1.replace("pos-", "")
        pos2 = pos2.replace("pos-", "")
        pos1_x = int(pos1.split('-')[0])
        pos1_y = int(pos1.split('-')[1])
        pos2_x = int(pos2.split('-')[0])
        pos2_y = int(pos2.split('-')[1])
        if pos1_x == pos2_x:
            if abs(pos1_y - pos2_y) == 1:
                return True
            return False
        if pos1_y == pos2_y:
            if abs(pos1_x - pos2_x) == 1:
                return True
            return False
        return False

    @staticmethod
    def add_missing_literals(state):
        """
        input: state - add the literals that are missing in the iaa state
        output: gym compatible complete state, (dict format)
        literals to add: 'move-dir', 'move'

        """

        state.state['move'] = [('dir-down',), ('dir-left',), ('dir-right',), ('dir-up',)]

        for v in state.state['at']:
            if "stone" in v[0]:
                if tuple([v[1]]) in state.state['is-goal']:
                    if 'at-goal' not in state.state.keys():
                        state.state['at-goal'] = [tuple([v[0]])]
                    else:
                        state.state['at-goal'].append(tuple([v[0]]))

        return state

    def get_state_from_render(self, img, init_state):
        """
        get all initial info from init_state
        detect and filter contours from image
        iterate over each contour and create Cells
        for each cell, detect its type and store the results
        add the detected cell literals to state
        """

        cell_width = 35
        temp_state = {}
        for pred in init_state.state:
            if "at" not in pred and "clear" not in pred:
                temp_state[pred] = init_state.state[pred]

        temp_state['at'] = []
        temp_state['clear'] = []

        detected_state = State(temp_state, init_state.objects)
        filtered_contours = get_contours(img, range_min=1100, range_max=1500)
        if VERBOSE:
            print("Number of contours:" + str(len(filtered_contours)))
        px_coords = []
        py_coords = []
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        grid = []
        for c in filtered_contours:
            _m = cv2.moments(c)
            pc_x = int(_m["m10"] / _m["m00"])
            pc_y = int(_m["m01"] / _m["m00"])
            px_coords.append(pc_x)
            py_coords.append(pc_y)
            mean_color = calculate_color(img_lab,c)
            cell_temp = Cell(pc_x, pc_y, c, mean_color)
            grid.append(cell_temp)

        px_coords = list(sorted(px_coords))
        grid = sorted(grid, key=lambda x: x.pc_x)
        i = 0
        j = 1
        while i < len(px_coords) and j < len(px_coords):
            if abs(px_coords[i] - px_coords[j]) < cell_width:
                px_coords[j] = px_coords[i]
                grid[j].pc_x = px_coords[j]
                j += 1
                continue
            i = j
            j += 1

        py_coords = list(sorted(py_coords))
        grid = sorted(grid, key=lambda x: x.pc_y)
        i = 0
        j = 1
        while i < len(py_coords) and j < len(py_coords):
            if abs(py_coords[i] - py_coords[j]) < 35:
                py_coords[j] = py_coords[i]
                grid[j].pc_y = py_coords[j]
                j += 1
                continue
            i = j
            j += 1

        px_coords = sorted(list(set(px_coords)))
        py_coords = sorted(list(set(py_coords)))
        for i in range(len(grid)):
            grid[i].c_x = px_coords.index(grid[i].pc_x) + 1
            grid[i].c_y = py_coords.index(grid[i].pc_y) + 1
            cell_location = 'pos-' + "{0:0=2d}".format(grid[i].c_x) + '-' + "{0:0=2d}".format(grid[i].c_y)
            if VERBOSE:
                print(str(grid[i].c_x) + "," + str(grid[i].c_y))
            if self.ref_colors is not None:
                grid[i].assign_cell_type(self.ref_colors)
                if VERBOSE:
                    print("Assigned " + str(grid[i].c_x) + ',' + str(grid[i].c_y) + ' as ' + grid[i].cell_type)

                if grid[i].cell_type == "player":
                    detected_state.state['at'].append(tuple(["player-01", cell_location]))

                if grid[i].cell_type == "clear" or grid[i].cell_type == "goal":
                    detected_state.state['clear'].append(tuple([cell_location]))

                if grid[i].cell_type == "wall":
                    continue

                if "stone" in grid[i].cell_type:
                    detected_state.state['at'].append(tuple(["stone-01", cell_location]))

                if "goal-at-stone" in grid[i].cell_type:
                    if "at-goal" not in detected_state.state.keys():
                        detected_state.state['at-goal'] = [tuple([grid[i].cell_type])]
                    else:
                        detected_state.state['at-goal'].append(tuple([grid[i].cell_type]))

                if grid[i].cell_type == "goal":
                    continue
        if VERBOSE:
            print("Number of cells found:" + str(len(grid)))
        return detected_state


class DoorsState(object):
    def __init__(self):
        print("Creating Doors State")
        self.ref_colors = OrderedDict({
            'clear':[215.70501730103805, 124.88235294117648, 147.01643598615917],
			'key_only':[204.20244897959185, 126.95591836734694, 162.29795918367347],
			'locked_only':[128.69243697478993, 125.98403361344539, 138.87899159663866],
			'locked_with_key':[160.6420634920635, 127.61746031746031, 157.83412698412698],
			'player':[167.99101307189542, 131.4893790849673, 147.08169934640523],
			'player_with_key':[182.13632653061225, 129.55755102040817, 158.30204081632652]
        })

    @staticmethod
    def add_missing_literals(state):
        """
        input: state - add the literals that are missing in the iaa state
        output: gym compatible complete state, (list of string of literals)
        """
        return state

    @staticmethod
    def create_custom_query_init_state(init_state):
        return init_state

    def get_state_from_render(self, img, init_state):
        """
        get all initial info from init_state
        detect and filter contours from image
        iterate over each contour and create Cells
        for each cell, detect its type and store the results
        add the detected cell literals to state
        """
        cell_width = 35
        temp_state = {}
        for pred in init_state.state:
            if pred not in ["at", "unlocked"]:
                temp_state[pred] = init_state.state[pred]

        temp_state['at'] = []
        temp_state['unlocked'] = [('room-0',)]
        key_locations = {}
        key_for_room = {}
        for (key, loc) in temp_state['keyat']:
            key_locations[loc] = key
        for (key, room) in temp_state['keyforroom']:
            key_for_room[key] = room

        detected_state = State(temp_state, init_state.objects)
        filtered_contours = get_contours(img, range_min=1500, range_max=2000)
        px_coords = []
        py_coords = []
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        grid = []
        for c in filtered_contours:
            _m = cv2.moments(c)
            pc_x = int(_m["m10"] / _m["m00"])
            pc_y = int(_m["m01"] / _m["m00"])
            px_coords.append(pc_y)
            py_coords.append(pc_x)
            mean_color = calculate_color(img_lab,c)
            cell_temp = Cell(pc_y, pc_x, c, mean_color)
            grid.append(cell_temp)

        px_coords = list(sorted(px_coords))
        grid = sorted(grid, key=lambda x: x.pc_x)
        i = 0
        j = 1
        while i < len(px_coords) and j < len(px_coords):
            if abs(px_coords[i] - px_coords[j]) < cell_width:
                px_coords[j] = px_coords[i]
                grid[j].pc_x = px_coords[j]
                j += 1
                continue
            i = j
            j += 1

        py_coords = list(sorted(py_coords))
        grid = sorted(grid, key=lambda x: x.pc_y)
        i = 0
        j = 1
        while i < len(py_coords) and j < len(py_coords):
            if abs(py_coords[i] - py_coords[j]) < 35:
                py_coords[j] = py_coords[i]
                grid[j].pc_y = py_coords[j]
                j += 1
                continue
            i = j
            j += 1

        cell_type_dict = {}
        px_coords = sorted(list(set(px_coords)))
        py_coords = sorted(list(set(py_coords)))
        keysat = set()
        for i in range(len(grid)):
            grid[i].c_x = px_coords.index(grid[i].pc_x)
            grid[i].c_y = py_coords.index(grid[i].pc_y)
            cell_location = 'loc-' + str(grid[i].c_x) + '-' + str(grid[i].c_y)
            if self.ref_colors is not None:
                grid[i].assign_cell_type(self.ref_colors)
                cell_type_dict[str(grid[i].c_x) + ',' + str(grid[i].c_y)] = grid[i].cell_type

                if grid[i].cell_type == "player":
                    detected_state.state['at'].append(tuple([cell_location]))

                if grid[i].cell_type == "player_with_key":
                    detected_state.state['at'].append(tuple([cell_location]))
                    keysat.add(cell_location)
                if grid[i].cell_type in ["locked_with_key", "key_only"]:
                    keysat.add(cell_location)

        # remove keys at for keys that have been picked
        detected_state.state['keyat'] = set(detected_state.state['keyat'])
        if sorted(keysat) != sorted(list((key_locations.keys()))):
            # if key is not present in location, that room is unlocked and keyat is removed
            for loc in set(key_locations.keys()).difference(set(keysat)):
                detected_state.state['unlocked'].append(tuple([key_for_room[key_locations[loc]]]))
                detected_state.state['keyat'].remove(tuple([key_locations[loc], loc]))
        if 'keyat' in detected_state.state.keys() and len(detected_state.state['keyat']) > 0:
            detected_state.state['keyat'] = list(detected_state.state['keyat'])
        elif 'keyat' in detected_state.state.keys() and len(detected_state.state['keyat']) == 0:
            detected_state.state.pop('keyat')
        print("Number of cells found:" + str(len(grid)))
        return detected_state


class StateHelper(object):
    """
    This Helper creates an object of type <EnvType>State
    It helps with:
        ----Given IAA state (missing some literals), get the gym-compatible
            state (add the missing literals)
        ----Given the render of the gym state from the sim,
            get the IAA state (Object Detection)
    """

    def __init__(self, env):
        # img is RGB render of state, saved by PIL and read by cv2 from gym's render
        if env == 'sokoban':
            self.state = SokobanState()
        elif env == 'doors':
            self.state = DoorsState()

    def iaa_to_gym_state(self, state):
        """
        input state is a dict format
        output is gym compatible state in dict format: this can be
        set using the helpers in sim_agent

        """
        gym_dict_state = self.state.add_missing_literals(state)
        return gym_dict_state

    def img_to_iaa_state(self, img, init_state):
        """
        input render of the sim and query that was given,
        output IAA state of type State
        """
        return self.state.get_state_from_render(img, init_state)

    def create_custom_query_init_state(self, init_state):
        """
        create custom query using minimum
        state specifications
        """
        return self.state.create_custom_query_init_state(init_state)
