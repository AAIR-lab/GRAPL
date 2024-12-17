#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict

import cv2
import numpy as np
from scipy.spatial import distance as dist

from lattice import State


def gym_to_iaa_dict(state):
    """
    Converts gym state to iaa state (dict) format
    """

    ret_state = {}
    objects = {}
    for lit in list(state):
        pred_name = lit.predicate.name
        if pred_name not in ret_state.keys():
            ret_state[pred_name] = []
        vars_temp = []
        for v in lit.variables:
            vars_temp.append(v.name)
            if v.var_type in objects.keys():
                if v.name not in objects[v.var_type]:
                    objects[v.var_type].append(v.name)
            else:
                objects[v.var_type] = [v.name]

        ret_state[pred_name].append(tuple(vars_temp))
    return State(ret_state, objects)


def problem_file_header(sim_domain_name, object_file_text, init_state_text):
    header = '(define (problem '
    header += sim_domain_name + ') \n (:domain '
    header += sim_domain_name + ')\n (:objects \n'
    if len(object_file_text) != 0:
        header += object_file_text
    header += ')\n (:init \n'
    header += init_state_text
    header += ')\n(:goal (and' + ' '

    return header


def get_contours(img_in, thresh=250, maxval=255, range_min=1500, range_max=2000):
    # input BGR image,output contours within thr given range
    img = img_in.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY_INV)[1]
    contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for i, cnt in enumerate(contours[0]):
        if range_min < cv2.contourArea(cnt) < range_max:
            filtered_contours.append(cnt)
    return filtered_contours


def calculate_color(img, contour):
    # Given the image and contour,calculate mean color in image's color space
    mask = np.zeros(img.shape[:2], np.uint8)
    x, y, w, h = cv2.boundingRect(contour)
    rect = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
    mask = cv2.erode(mask, None, iterations=2)
    mean = cv2.mean(img, mask=mask)[:3]

    return mean


class Cell:
    def __init__(self, c_x, c_y, contour, mean_color):
        self.pc_x = c_x
        self.pc_y = c_y
        self.c_x = 0
        self.c_y = 0
        self.contour = contour
        self.mean_color = mean_color
        self.cell_type = None

    def assign_cell_type(self, ref_colors):
        self.cell_type = self.match_cell(self.mean_color, ref_colors)

    @staticmethod
    def match_cell(mean_color, ref_colors):
        # Given mean color, match with the closest one in dict
        assert type(ref_colors) == OrderedDict, "colors template not ordered"
        m = min(ref_colors.keys(), key=(lambda k: dist.euclidean(mean_color, ref_colors[k])))
        return m
