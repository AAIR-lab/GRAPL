#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import subprocess
import sys
from itertools import combinations

from config import *
from utils import FileUtils

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
import query.genericQuery as gc


class Query(gc.GenericQuery):
    """
    This class is for the plan executability query.
    Given two models, it finds a plan that will result in both the models
    leading to different states.

    Initial State has to be given as input.

    :param model1: an instance of class Model
    :type model1: object of class Model
    :param model2: an instance of class Model
    :type model2: object of class Model
    :param init_state: Initial state (list of predicates)
    :type init_state: list of strs
    """

    def __init__(self, model1, model2, init_state, next_pal_tuple, pal_tuple_dict):
        """
        This method creates a new instance of AR Query.
        """
        self.model1 = copy.deepcopy(model1)
        self.model2 = copy.deepcopy(model2)
        self.init_state = dict()
        self.plan = []
        self.pal = next_pal_tuple
        self.debug_text = ''
        self.pal_tuple_dict = pal_tuple_dict

        predicates = {}
        pred_names_set = set()

        for key, value in self.model1.predicates.items():
            pred_name = key.split("|")[0]
            new_key = key.replace(pred_name, pred_name + "_1")
            predicates[new_key] = value
            pred_names_set.add(pred_name)

        for key, value in self.model2.predicates.items():
            pred_name = key.split("|")[0]
            new_key = key.replace(pred_name, pred_name + "_2")
            predicates[new_key] = value
            pred_names_set.add(pred_name)

        for i in init_state:
            if i in pred_names_set:
                self.init_state[i] = init_state[i]

        if len(self.init_state) == 0:
            self.init_state['empty_init'] = [()]
            self.model1.predicates['empty_init'] = 0
            self.model2.predicates['empty_init'] = 0
            for action in self.model1.actions.keys():
                self.model1.actions[action]['empty_init'] = [Literal.POS, Literal.ABS]
            for action in self.model2.actions.keys():
                self.model1.actions[action]['empty_init'] = [Literal.POS, Literal.ABS]

    def call_planner(self, domain_file, problem_file, result_file):
        """
        This method calls the planner.
        The planner can be either FF Planner (ff) or Madagascar (mg).
        It needs to be set in config.py in the root directory.

        :param domain_file: domain file (operator file) for the planner
        :type domain_file: str
        :param problem_file: problem file (fact file) for the planner
        :type problem_file: str
        :param result_file: result file to store output of the planner
        :type result_file: str

        :rtype: None

        """
        if PLANNER == "FF":
            param = FF_PATH + "ff"
            param += " -o " + domain_file
            param += " -f " + problem_file
            param += " > " + result_file

        elif PLANNER == "FD":
            param = FD_PATH + "fast-downward.py "
            param += " --plan-file ../" + FD_SAS_FILE
            param += " --alias seq-sat-lama-2011"
            param += " " + domain_file
            param += "  " + problem_file
            # param += " --search \"astar(lmcut(), verbosity=silent)\""

        else:
            print("Error: No planner provided")
            exit()
        p = subprocess.Popen([param], shell=True)
        p.wait()

        if PLANNER == "FD":
            f = open("../" + FD_SAS_FILE + ".1", "r")
            _plan_found = True
            _plan = ""
            for x in f:
                if ("found legal plan as follows"):
                    _plan_found = True
                if ";" in x:
                    continue

                if "(" in x and ")" in x:
                    k = copy.deepcopy(x)
                    _plan += "|".join(k.lower().rstrip().split()) + ")\n"

                if "time spent" in x:
                    break
            f.close()

            f = open(result_file, "w")
            f.write(_plan)
            f.close()

    def add_unknown_pred_to_model(self, model_actions):
        temp_actions = copy.deepcopy(model_actions)
        for actionName, predicateDict_m1 in temp_actions.items():
            if (actionName, self.pal[1], Location.PRECOND) not in self.pal_tuple_dict.keys():
                # This predicate and action might be incompatible
                continue
            predicateDict_m1['unknown'] = [Literal.POS, Literal.POS]
            if self.pal_tuple_dict[(actionName, self.pal[1], Location.PRECOND)]:
                predicateDict_m1['unknown'][0] = Literal.ABS
            if self.pal_tuple_dict[(actionName, self.pal[1], Location.EFFECTS)]:
                predicateDict_m1['unknown'][1] = Literal.NEG

        # Remove unknown from current pal tuple's a,l
        if self.pal[2] == Location.PRECOND:
            temp_actions[self.pal[0]]['unknown'][int(self.pal[2]) - 1] = Literal.ABS
        elif self.pal[2] == Location.EFFECTS:
            temp_actions[self.pal[0]]['unknown'][int(self.pal[2]) - 1] = Literal.NEG

        return temp_actions

    def add_unknown_predicate(self):
        temp_actions_m1 = self.add_unknown_pred_to_model(self.model1.actions)
        temp_actions_m2 = self.add_unknown_pred_to_model(self.model2.actions)

        return temp_actions_m1, temp_actions_m2

    def write_query_to_file(self, fd, domain_name, objects, pred_type_mapping, action_parameters):
        """
        This method creates files.

        :param fd: file descriptor of the pddl file in which model will be written
        :type fd: file descriptor
        :param domain_name: domain name of the model
        :type domain_name: str

        :rtype: None

        """

        use_unknown = True

        self.write(fd, "(define (domain " + domain_name + ")\n")
        self.write(fd, "(:requirements :strips :typing :conditional-effects :equality :negative-preconditions)\n")

        ####### Typing #######
        self.write(fd, "(:types")
        for t in objects.keys():
            self.write(fd, " " + t)
        self.write(fd, ")\n")

        self.write(fd, "(:predicates ")
        count = 0
        preds_printed = []
        for key, value in self.model1.predicates.items():
            params = ""
            cnt = 0
            pred_name = key.split("|")[0]
            if pred_name != 'empty_init':
                for val in pred_type_mapping[pred_name]:
                    params = params + " ?" + val[0] + str(cnt) + " - " + val
                    cnt += 1

            if count > 0:
                self.write(fd, "\n")
                for k in range(len("(:predicates ")):
                    self.write(fd, " ")
            if pred_name not in preds_printed:
                preds_printed.append(pred_name)
                self.write(fd, "(" + pred_name + "_1 " + params + ")")
                self.write(fd, "(" + pred_name + "_2 " + params + ")")
                count += 1

        self.write(fd, "\n")
        if use_unknown:
            # ADD UNKNOWN
            for k in range(len("(:predicates ")):
                self.write(fd, " ")
            self.write(fd, "(unknown_1)")
            self.write(fd, "(unknown_2)\n")

        for k in range(len("(:predicates ")):
            self.write(fd, " ")
        self.write(fd, "(dummy_pred_1)")
        self.write(fd, "(dummy_pred_2)")
        self.write(fd, ")\n\n")

        # Needed to copy because we will add key unknown later.
        temp_actions_m1, temp_actions_m2 = self.add_unknown_predicate()
        for actionName, predicateDict_m1 in temp_actions_m1.items():
            head = "(:action " + actionName + "\n" + "  :parameters"
            self.write(fd, head)

            count = 0
            type_count = {}
            param_ordering = []
            for p in action_parameters[actionName]:
                if p not in type_count.keys():
                    type_count[p] = 1
                else:
                    type_count[p] = type_count[p] + 1
                param_ordering.append(p + str(type_count[p]))
            self.write(fd, " (")
            head = ""
            param_count = len(action_parameters[actionName])
            for i in range(param_count):
                if i > 0:
                    for k in range(len("  :parameters (")):
                        head += " "
                head += "?" + param_ordering[i] + " - " + action_parameters[actionName][i] + "\n"
            for k in range(len("  :parameters ")):
                head += " "
            head += ")\n"
            self.write(fd, head)

            count = -1

            ########### Write Precondition ###########
            self.write(fd, "  :precondition ")

            equality_needed = False
            if param_count > 1:
                equality_needed = True

            if equality_needed:
                # Ensure none of the parameters are equal to each other
                combs = combinations(list(range(0, param_count)), 2)
                self.write(fd, "(and ")
                for c in combs:
                    self.write(fd, "(not (= ")
                    for j in range(2):
                        i = c[j]
                        self.write(fd, "?" + param_ordering[i])
                        if (j == 0):
                            self.write(fd, " ")
                        else:
                            self.write(fd, ")) ")
                    self.write(fd, "\n")
                    for k in range(len("  :precondition (and ")):
                        self.write(fd, " ")

            # Write precondition of M1 and Precondition of M2 in OR
            # This ensures the models are distinguished if only one model
            # can execute this action
            self.write(fd, "(or \n")
            for k in range(len("  :precondition (and (or ")):
                self.write(fd, " ")

            # Write predicate 1
            head_m1 = ""
            not_head_m1 = ""

            for predicate, value in predicateDict_m1.items():

                pred_split = predicate.split("|")
                pred_name = pred_split[0]

                t_value = copy.deepcopy(value)

                if (t_value[0] == Literal.AN or t_value[0] == Literal.AP):
                    t_value[0] = Literal.ABS
                elif (t_value[0] == Literal.NP):
                    t_value[0] = Literal.POS

                if (t_value[0] != Literal.ABS) and not (
                        use_unknown and self.pal[0] == actionName and self.pal[1] == predicate):
                    param = "("
                    not_param = "("
                    if (t_value[0] == Literal.NEG):
                        param += "not ("
                    if (t_value[0] == Literal.POS):
                        not_param += "not ("
                    param += pred_name + "_1"
                    not_param += pred_name + "_1"

                    if len(pred_split) > 1:
                        pred_params = pred_split[1:]
                        for p in pred_params:
                            # print(p)
                            param += " ?" + param_ordering[int(p)]
                            not_param += " ?" + param_ordering[int(p)]

                    param += ")"
                    not_param += ")"
                    if (t_value[0] != Literal.ABS and t_value[0] != Literal.POS):
                        param += ")"
                    if (t_value[0] != Literal.ABS and t_value[0] != Literal.NEG):
                        not_param += ")"
                    for k in range(len("  :precondition (and (or (and ")):
                        head_m1 += " "
                    for k in range(len("  :precondition (and (or (and (or ")):
                        not_head_m1 += " "
                    head_m1 += param + "\n"
                    not_head_m1 += not_param + "\n"

                elif (use_unknown and self.pal[0] == actionName and self.pal[1] == predicate):
                    if t_value[0] != Literal.ABS:
                        # Add +,-,unkn in or
                        param = "(or ("
                        not_param = "(or ("

                        if (t_value[0] == Literal.NEG):
                            param += "not ("
                        if (t_value[0] == Literal.POS):
                            not_param += "not ("

                        param += pred_name + "_1"
                        not_param += pred_name + "_1"
                        if len(pred_split) > 1:
                            pred_params = pred_split[1:]
                            for p in pred_params:
                                # print(p)
                                param += " ?" + param_ordering[int(p)]
                                not_param += " ?" + param_ordering[int(p)]
                        param += ")"
                        not_param += ")"
                        for k in range(len("  :precondition (and (or (and ")):
                            head_m1 += " "
                        for k in range(len("  :precondition (and (or (and (or ")):
                            not_head_m1 += " "
                        if (t_value[0] != Literal.ABS and t_value[0] != Literal.POS):
                            param += ")"
                        if (t_value[0] != Literal.ABS and t_value[0] != Literal.NEG):
                            not_param += ")"
                        head_m1 += param + "\n"
                        not_head_m1 += not_param + "\n"
                        for k in range(len("  :precondition (and (or (and (or ")):
                            head_m1 += " "
                        for k in range(len("  :precondition (and (or (and (or (or ")):
                            not_head_m1 += " "
                        head_m1 += "(unknown_1)\n"
                        not_head_m1 += "(unknown_1)\n"

                        for k in range(len("  :precondition (and (or (and ")):
                            head_m1 += " "
                        for k in range(len("  :precondition (and (or (and (or ")):
                            not_head_m1 += " "
                        head_m1 += ")\n"
                        not_head_m1 += ")\n"

            self.write(fd, "(and \n" + head_m1)
            for k in range(len("  :precondition (and (or ")):
                self.write(fd, " ")
            self.write(fd, ")\n")

            head_m2 = ""
            not_head_m2 = ""
            predicateDict_m2 = temp_actions_m2[actionName]

            for predicate, value in predicateDict_m2.items():
                pred_split = predicate.split("|")
                pred_name = pred_split[0]

                t_value = copy.deepcopy(value)
                if (t_value[0] == Literal.AN or t_value[0] == Literal.AP):
                    t_value[0] = Literal.ABS
                elif (t_value[0] == Literal.NP):
                    t_value[0] = Literal.POS

                if (t_value[0] != Literal.ABS) and not (
                        use_unknown and self.pal[0] == actionName and self.pal[1] == predicate):
                    not_param = "("
                    param = "("
                    if (t_value[0] == Literal.NEG):
                        param += "not ("
                    if (t_value[0] == Literal.POS):
                        not_param += "not ("
                    param += pred_name + "_2"
                    not_param += pred_name + "_2"

                    if len(pred_split) > 1:
                        pred_params = pred_split[1:]
                        for p in pred_params:
                            # print(p)
                            param += " ?" + param_ordering[int(p)]
                            not_param += " ?" + param_ordering[int(p)]

                    param += ")"
                    not_param += ")"
                    if (t_value[0] != Literal.ABS and t_value[0] != Literal.POS):
                        param += ")"
                    if (t_value[0] != Literal.ABS and t_value[0] != Literal.NEG):
                        not_param += ")"
                    for k in range(len("  :precondition (and (or (and ")):
                        head_m2 += " "
                    for k in range(len("  :precondition (and (or (and (or ")):
                        not_head_m2 += " "
                    head_m2 += param + "\n"
                    not_head_m2 += not_param + "\n"

                elif (use_unknown and self.pal[0] == actionName and self.pal[1] == predicate):
                    if t_value[0] != Literal.ABS:
                        # Add +,-,unkn in or
                        param = "(or ("
                        not_param = "(or ("
                        if (t_value[0] == Literal.NEG):
                            param += "not ("
                        if (t_value[0] == Literal.POS):
                            not_param += "not ("
                        param += pred_name + "_2"
                        not_param += pred_name + "_2"
                        if len(pred_split) > 1:
                            pred_params = pred_split[1:]
                            for p in pred_params:
                                param += " ?" + param_ordering[int(p)]
                                not_param += " ?" + param_ordering[int(p)]

                        param += ")"
                        not_param += ")"
                        for k in range(len("  :precondition (and (or (and ")):
                            head_m2 += " "
                        for k in range(len("  :precondition (and (or (and (or ")):
                            not_head_m2 += " "
                        if (t_value[0] != Literal.ABS and t_value[0] != Literal.POS):
                            param += ")"
                        if (t_value[0] != Literal.ABS and t_value[0] != Literal.NEG):
                            not_param += ")"
                        head_m2 += param + "\n"
                        not_head_m2 += not_param + "\n"
                        for k in range(len("  :precondition (and (or (and (or ")):
                            head_m2 += " "
                        for k in range(len("  :precondition (and (or (and (or (or ")):
                            not_head_m2 += " "
                        head_m2 += "(unknown_2)\n"
                        not_head_m2 += "(unknown_2)\n"

                        for k in range(len("  :precondition (and (or (and ")):
                            head_m2 += " "
                        for k in range(len("  :precondition (and (or (and (or ")):
                            not_head_m2 += " "
                        head_m2 += ")\n"
                        not_head_m2 += ")\n"

            for k in range(len("  :precondition (and (or ")):
                self.write(fd, " ")
            self.write(fd, "(and \n" + head_m2)
            for k in range(len("  :precondition (and (or ")):
                self.write(fd, " ")
            self.write(fd, ")\n")

            if equality_needed:
                for k in range(len("  :precondition (and ")):
                    self.write(fd, " ")
                self.write(fd, ")\n")
            for k in range(len("  :precondition ")):
                self.write(fd, " ")
            self.write(fd, ")\n")

            count = 0
            self.write(fd, "  :effect (and")
            # When (prec(m1)) (eff(m1))
            self.write(fd, " (when (and\n")
            self.write(fd, head_m1 + head_m2)
            for k in range(len("  :effect (and (when ")):
                self.write(fd, " ")
            self.write(fd, ")\n")
            for k in range(len("  :effect (and (when ")):
                self.write(fd, " ")
            fd.write("(and \n")
            for predicate, value in predicateDict_m1.items():
                pred_split = predicate.split("|")
                pred_name = pred_split[0]

                t_value = copy.deepcopy(value)
                if (t_value[1] == Literal.AN or t_value[1] == Literal.AP):
                    t_value[1] = Literal.ABS
                elif (t_value[1] == Literal.NP):
                    t_value[1] = Literal.POS

                param = ""
                for k in range(len("  :precondition (and (or (and ")):
                    param += " "
                if (t_value[1] != Literal.ABS):
                    param += "("
                    if (t_value[1] == Literal.NEG):
                        param += "not ("
                    param += pred_name + "_1"

                    if len(pred_split) > 1:
                        pred_params = pred_split[1:]
                        for p in pred_params:
                            # print(p)
                            param += " ?" + param_ordering[int(p)]

                    param += ")"
                    if (t_value[1] != Literal.ABS and t_value[1] != Literal.POS):
                        param += ")"
                    self.write(fd, param + "\n")

            for predicate, value in predicateDict_m2.items():
                pred_split = predicate.split("|")
                pred_name = pred_split[0]

                t_value = copy.deepcopy(value)
                if (t_value[1] == Literal.AN or t_value[1] == Literal.AP):
                    t_value[1] = Literal.ABS
                elif (t_value[1] == Literal.NP):
                    t_value[1] = Literal.POS

                param = ""
                for k in range(len("  :precondition (and (or (and ")):
                    param += " "
                if (t_value[1] != Literal.ABS):
                    param += "("
                    if (t_value[1] == Literal.NEG):
                        param += "not ("
                    param += pred_name + "_2"

                    if len(pred_split) > 1:
                        pred_params = pred_split[1:]
                        for p in pred_params:
                            # print(p)
                            param += " ?" + param_ordering[int(p)]

                    param += ")"
                    if (t_value[1] != Literal.ABS and t_value[1] != Literal.POS):
                        param += ")"
                    self.write(fd, param + "\n")
            for k in range(len("  :precondition (and ")):
                self.write(fd, " ")
            self.write(fd, ")\n")
            for k in range(len("  :effect (and ")):
                self.write(fd, " ")
            self.write(fd, ")\n")

            for k in range(len("  :effect (and ")):
                self.write(fd, " ")
            self.write(fd, "(when ")
            # When (or (!(prec(m1))) (!(prec(m2)))) (create dummy diff)
            self.write(fd, "(or \n")
            for k in range(len("  :effect (and (when (or ")):
                self.write(fd, " ")
            self.write(fd, "(and \n" + head_m1)
            for k in range(len("  :effect (and (when (or (and ")):
                self.write(fd, " ")
            self.write(fd, "(or \n" + not_head_m2)
            for k in range(len("  :effect (and (when (or (and ")):
                self.write(fd, " ")
            self.write(fd, ")\n")
            for k in range(len("  :effect (and (when (or ")):
                self.write(fd, " ")
            self.write(fd, ")\n")
            for k in range(len("  :effect (and (when (or ")):
                self.write(fd, " ")
            self.write(fd, "(and \n" + head_m2)
            for k in range(len("  :effect (and (when (or (and ")):
                self.write(fd, " ")
            self.write(fd, "(or \n" + not_head_m1)
            for k in range(len("  :effect (and (when (or (and ")):
                self.write(fd, " ")
            self.write(fd, ")\n")
            for k in range(len("  :effect (and (when (or ")):
                self.write(fd, " ")
            self.write(fd, ")\n")
            for k in range(len("  :effect (and (when ")):
                self.write(fd, " ")
            self.write(fd, ")\n")
            for k in range(len("  :effect (and (when ")):
                self.write(fd, " ")
            self.write(fd, "(and \n")
            for k in range(len("  :effect (and (when (and ")):
                self.write(fd, " ")
            self.write(fd, "(dummy_pred_1)\n")
            for k in range(len("  :effect (and (when (and ")):
                self.write(fd, " ")
            self.write(fd, "(not(dummy_pred_2))\n")
            for k in range(len("  :effect (and (when ")):
                self.write(fd, " ")
            self.write(fd, ") \n")
            for k in range(len("  :effect (and ")):
                self.write(fd, " ")
            self.write(fd, ")\n")
            for k in range(len("  :effect ")):
                self.write(fd, " ")
            self.write(fd, ")\n")
            self.write(fd, ")\n\n")

        self.write(fd, ")\n")

    def write(self, fd, txt):
        self.debug_text += txt
        fd.write(txt)

    def get_plan_from_query(self,
                            init_state,
                            domain_name,
                            objects,
                            pred_type_mapping,
                            action_parameters):

        f = open(Q_DOMAIN_FILE, "w")
        self.write_query_to_file(f, domain_name, objects, pred_type_mapping, action_parameters)
        f.close()

        f = open(Q_PROBLEM_FILE, "w")
        FileUtils.writeProblemToFile(self, f, domain_name, domain_name + "-1", True, objects, pred_type_mapping)
        f.close()

        self.call_planner(Q_DOMAIN_FILE, Q_PROBLEM_FILE, Q_RESULT_FILE)
        self.plan = FileUtils.get_plan_from_file(Q_RESULT_FILE)
        planRaw = self.plan

        if len(planRaw) != 0:
            f = open(Q_PLAN_FILE, "w")
            FileUtils.writePlanToFile(self, f, init_state, domain_name, domain_name + "-1", objects)
            f.close()

        return planRaw
