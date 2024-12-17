#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from itertools import combinations

from config import *
import os
import shutil

class FileUtils(object):

    @staticmethod
    def initialize_directory(dirpath, clean=True):
    
        dirpath = os.path.abspath(dirpath)
    
        if os.path.exists(dirpath):
            
            assert os.path.isdir(dirpath)
            if clean:
                
                shutil.rmtree(dirpath)
                os.makedirs(dirpath)
        else:
            os.makedirs(dirpath)

    @staticmethod
    def remove_file(filepath):

        try:

            os.remove(filepath)
        except FileNotFoundError:

            pass

    @classmethod
    def get_plan_from_file(cls, result_file):
        """
        This method extracts the plan from the output of the planner.
        The planner can be either FF Planner (ff) or Madagascar (mg).
        It needs to be set in config.py in the root directory.

        :param result_file: result file where output of the planner is stored.
        :type result_file: str

        :return: Plan as a list of action names
        :rtype: list of str

        """
        import time
        plan = []

        if PLANNER == "FF":
            for line in open(result_file):
                if 'STEP' in line:
                    values = line.split()
                    if (values[2] != "REACH-GOAL"):
                        plan.append(("|".join(values[2:])).lower())

        elif PLANNER == "FD" or PLANNER == "PRP":
            for line in open(result_file):
                if ';' not in line:
                    if line == "\n":
                        continue
                    values = line.split("(")
                    values = values[1].split(")")

                    plan.append(values[0].rstrip().lower())
        return plan

    def writeProblemToFile(ar_query, fd, domain_name, problemName, useDummyPreds, objects, pred_type_mapping):
        """
        This method creates files.

        :param fd: file descriptor of the pddl file in which problem will be written
        :type fd: file descriptor
        :param domain_name: domain name of the model
        :type domain_name: str

        :rtype: None

        """
        import time

        init_state = ar_query.init_state
        model = ar_query.model1

        fd.write("(define (problem " + problemName + ")\n")
        ####### Domain #######
        fd.write("  (:domain " + domain_name + ")\n")

        ####### Objects #######
        fd.write("  (:objects ")

        k = 0
        for t, vals in objects.items():
            if len(vals) == 0:
                # This case happens with domains like logistics
                # Here physobj has no actual object
                continue
            if k > 0:
                for k in range(len("  (:objects ")):
                    fd.write(" ")
            for v in vals:
                fd.write(v + str(" "))
            fd.write(" - " + t + " ")
            k += 1
        fd.write(")\n")

        fd.write("  (:init ")
        count = 0
        for p, value in init_state.items():
            for vals in value:
                params = ""
                for j in range(len(vals)):
                    if j > 0:
                        params += " "
                    params += vals[j]

                if count > 0:
                    fd.write("\n")
                    for k in range(len("  (:init ")):
                        fd.write(" ")
                count += 1
                if (p in init_state):
                    fd.write("(" + p + "_1 " + params + ")")
                    if count > 0:
                        fd.write("\n")
                        for k in range(len("  (:init ")):
                            fd.write(" ")
                    fd.write("(" + p + "_2 " + params + ")")
                else:
                    fd.write("(not (" + p + "_1 " + params + "))")
                    if count > 0:
                        fd.write("\n")
                        for k in range(len("  (:init ")):
                            fd.write(" ")
                    fd.write("(not (" + p + "_2 " + params + "))")
        fd.write("\n")

        fd.write("  )\n")

        ####### Goal #######

        max_type_count = {}
        preds_name = [i.split("|", 1)[0] for i in model.predicates.keys()]
        preds_name = list(set(preds_name))
        # print(preds_name)

        ################
        # any_object = True when there is at least one object in the domain.
        # This helps in avoiding situations where:
        # :goal (exists (
        #               )
        # Hence if no object, set any_object to False to avoid printing exists()
        ################
        any_object = False

        for t in objects.keys():
            max_count = 0
            for k, p in pred_type_mapping.items():
                if k in preds_name:
                    max_count = max(max_count, p.count(t))
            max_type_count[t] = max_count
            if max_count > 0:
                any_object = True
        if any_object == True:
            fd.write("  (:goal (exists (\n")
        else:
            fd.write("  (:goal \n")
        for t in objects.keys():
            param = ""
            for k in range(len("  (:goal (exists ")):
                param += " "
            for i in range(max_type_count[t]):
                param += " ?" + str(t) + str(i + 1)
            param += " - " + str(t) + "\n"
            if max_type_count[t] != 0:
                fd.write(param)
        if any_object == True:
            for k in range(len("  (:goal (exists ")):
                fd.write(" ")
            fd.write(")\n")

        for k in range(len("  (:goal (exists ")):
            fd.write(" ")
        fd.write("(or\n")
        preds_covered = []
        for k_, val in model.predicates.items():
            key = k_.split("|")[0]
            if key in preds_covered:
                continue
            else:
                preds_covered.append(key)

            # print("key = "+str(key))
            for k in range(len("  (:goal (exists (or")):
                fd.write(" ")
            param = "(and "
            param += "(" + key + "_1 "

            type_count = {}
            if key != 'empty_init':
                for v in pred_type_mapping[key]:
                    if v not in type_count.keys():
                        type_count[v] = 1
                    else:
                        type_count[v] = type_count[v] + 1
                    param += " ?" + v + str(type_count[v])
            param += ")\n"
            fd.write(param)
            for k in range(len("  (:goal (exists (or (and")):
                fd.write(" ")

            param = ""
            param += "(not (" + key + "_2 "
            type_count = {}
            if key != 'empty_init':
                for v in pred_type_mapping[key]:
                    if v not in type_count.keys():
                        type_count[v] = 1
                    else:
                        type_count[v] = type_count[v] + 1
                    param += " ?" + v + str(type_count[v])
            param += "))\n"
            fd.write(param)
            for k in range(len("  (:goal (exists (or")):
                fd.write(" ")
            fd.write(")\n")

            for k in range(len("  (:goal (exists (or")):
                fd.write(" ")
            param = "(and "
            param += "(" + key + "_2 "
            type_count = {}
            if key != 'empty_init':
                for v in pred_type_mapping[key]:
                    if v not in type_count.keys():
                        type_count[v] = 1
                    else:
                        type_count[v] = type_count[v] + 1
                    param += " ?" + v + str(type_count[v])
            param += ")\n"
            fd.write(param)
            for k in range(len("  (:goal (exists (or (and")):
                fd.write(" ")

            param = "(not (" + key + "_1 "
            type_count = {}
            if key != 'empty_init':
                for v in pred_type_mapping[key]:
                    if v not in type_count.keys():
                        type_count[v] = 1
                    else:
                        type_count[v] = type_count[v] + 1
                    param += " ?" + v + str(type_count[v])
            param += "))\n"
            fd.write(param)
            for k in range(len("  (:goal (exists (or")):
                fd.write(" ")
            fd.write(")\n")

        if useDummyPreds == True:
            for k in range(len("  (:goal (exists (or")):
                fd.write(" ")
            fd.write("(and (dummy_pred_1)\n")
            for k in range(len("  (:goal (exists (or (and")):
                fd.write(" ")
            fd.write("(not (dummy_pred_2))\n")
            for k in range(len("  (:goal (exists (or")):
                fd.write(" ")
            fd.write(")\n")
            for k in range(len("  (:goal (exists (or")):
                fd.write(" ")

        if any_object == True:
            for k in range(len("   (:goal (exists ")):
                fd.write(" ")
            fd.write(")\n")
        for k in range(len("   (:goal ")):
            fd.write(" ")
        fd.write(")\n")
        fd.write("  )\n")
        fd.write(")\n")

    def writePlanToFile(ar_query, fd, init_state, domain_name, problemName, objects):
        """
        This method creates files.

        :param fd: file descriptor of the pddl file in which plan problem will be written
        :type fd: file descriptor
        :param domain_name: domain name of the model
        :type domain_name: str

        :rtype: None

        """
        plan = ar_query.plan
        fd.write("(define (problem " + problemName + ")\n")
        fd.write("(:domain " + domain_name + ")\n")
        fd.write("(:init ")
        count = 0
        for p in init_state:
            params = ""
            if count > 0:
                fd.write("\n")
                for k in range(len("(:init ")):
                    fd.write(" ")
            fd.write("(" + p + params + ")")
            count += 1
        fd.write(")\n\n")

        fd.write("(:plan\n")
        for k in plan:
            for j in range(len("(:plan")):
                fd.write(" ")
            param = ""
            param = " (" + k + ")\n"
            fd.write(param)
            param = ""
        fd.write(")\n")
        fd.write(")\n")

    def add_unknown_predicate(model1, model2, pal_tuple_dict, pal):
        temp_actions_m1 = copy.deepcopy(model1.actions)
        for actionName, predicateDict_m1 in temp_actions_m1.items():
            if (actionName, pal[1], Location.PRECOND) not in pal_tuple_dict.keys():
                # This predicate and action might be incompatible
                continue
            predicateDict_m1['unknown'] = [Literal.POS, Literal.POS]
            if pal_tuple_dict[(actionName, pal[1], Location.PRECOND)] == True:
                predicateDict_m1['unknown'][0] = Literal.ABS
            if pal_tuple_dict[(actionName, pal[1], Location.EFFECTS)] == True:
                predicateDict_m1['unknown'][1] = Literal.NEG

        # Remove unknown from current pal tuple's a,l
        if pal[2] == Location.PRECOND:
            temp_actions_m1[pal[0]]['unknown'][int(pal[2]) - 1] = Literal.ABS
        elif pal[2] == Location.EFFECTS:
            temp_actions_m1[pal[0]]['unknown'][int(pal[2]) - 1] = Literal.NEG

        temp_actions_m2 = copy.deepcopy(model2.actions)
        for actionName, predicateDict_m2 in temp_actions_m2.items():
            if (actionName, pal[1], Location.PRECOND) not in pal_tuple_dict.keys():
                # This predicate and action might be incompatible
                continue
            predicateDict_m2['unknown'] = [Literal.POS, Literal.POS]

            if pal_tuple_dict[(actionName, pal[1], Location.PRECOND)] == True:
                predicateDict_m2['unknown'][0] = Literal.ABS
            if pal_tuple_dict[(actionName, pal[1], Location.EFFECTS)] == True:
                predicateDict_m2['unknown'][1] = Literal.NEG

        if pal[2] == Location.PRECOND:
            temp_actions_m2[pal[0]]['unknown'][int(pal[2]) - 1] = Literal.ABS
        elif pal[2] == Location.EFFECTS:
            temp_actions_m2[pal[0]]['unknown'][int(pal[2]) - 1] = Literal.NEG

        return temp_actions_m1, temp_actions_m2

    def write(var, txt):
        var += txt
        return var

    @classmethod
    def write_domain_to_file(cls, fd, domain_name, objects, pred_type_mapping, action_parameters, model1, model2,
                             pal_tuple_dict, pal):
        """
        This method creates files.

        :param fd: file descriptor of the pddl file in which model will be written
        :type fd: file descriptor
        :param domain_name: domain name of the model
        :type domain_name: str

        :rtype: None

        """

        use_unknown = True

        fd.write("(define (domain " + domain_name + ")\n")
        fd.write("(:requirements :strips :typing :conditional-effects :equality :negative-preconditions)\n")

        ####### Typing #######
        fd.write("(:types")
        for t in objects.keys():
            fd.write(" " + t)
        fd.write(")\n")

        fd.write("(:predicates ")
        count = 0
        preds_printed = []
        for key, value in model1.predicates.items():
            params = ""
            cnt = 0
            pred_name = key.split("|")[0]
            if pred_name != 'empty_init':
                for val in pred_type_mapping[pred_name]:
                    params = params + " ?" + val[0] + str(cnt) + " - " + val
                    cnt += 1

            if count > 0:
                fd.write("\n")
                for k in range(len("(:predicates ")):
                    fd.write(" ")
            if pred_name not in preds_printed:
                preds_printed.append(pred_name)
                fd.write("(" + pred_name + "_1 " + params + ")")
                fd.write("(" + pred_name + "_2 " + params + ")")
                count += 1

        fd.write("\n")
        if use_unknown:
            # ADD UNKNOWN
            for k in range(len("(:predicates ")):
                fd.write(" ")
            fd.write("(unknown_1)")
            fd.write("(unknown_2)\n")

        for k in range(len("(:predicates ")):
            fd.write(" ")
        fd.write("(dummy_pred_1)")
        fd.write("(dummy_pred_2)")
        fd.write(")\n\n")

        # Needed to copy because we will add key unknown later.
        temp_actions_m1, temp_actions_m2 = FileUtils.add_unknown_predicate(model1, model2, pal_tuple_dict, pal)
        for actionName, predicateDict_m1 in temp_actions_m1.items():
            head = "(:action " + actionName + "\n" + "  :parameters"
            fd.write(head)

            count = 0
            type_count = {}
            param_ordering = []
            for p in action_parameters[actionName]:
                if p not in type_count.keys():
                    type_count[p] = 1
                else:
                    type_count[p] = type_count[p] + 1
                param_ordering.append(p + str(type_count[p]))
            fd.write(" (")
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
            fd.write(head)

            count = -1

            ########### Write Precondition ###########
            fd.write("  :precondition ")

            precond = ""
            precond_str = ""
            has_something = False
            equality_needed = False
            if param_count > 1:
                equality_needed = True

            if equality_needed:
                # Ensure none of the parameters are equal to each other
                combs = combinations(list(range(0, param_count)), 2)
                precond_str = FileUtils.write(precond_str, "(and ")
                for c in combs:
                    has_something = True
                    precond_str = FileUtils.write(precond_str, "(not (= ")
                    for j in range(2):
                        i = c[j]
                        precond_str = FileUtils.write(precond_str, "?" + param_ordering[i])
                        if (j == 0):
                            precond_str = FileUtils.write(precond_str, " ")
                        else:
                            precond_str = FileUtils.write(precond_str, ")) ")
                    precond_str = FileUtils.write(precond_str, "\n")
                    for k in range(len("  :precondition (and ")):
                        precond_str = FileUtils.write(precond_str, " ")

            # Write precondition of M1 and Precondition of M2 in OR
            # This ensures the models are distinguished if only one model
            # can execute this action
            # precond_str = FileUtils.write(precond_str, "(or \n")
            # for k in range(len("  :precondition (and (or ")):
            # 	precond_str = FileUtils.write(precond_str, " ")

            if has_something == True:
                precond += precond_str
            precond_str = ""

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
                        use_unknown and pal[0] == actionName and pal[1] == predicate):
                    param = "("
                    not_param = "("
                    # has_something = True
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
                # for k in range(len("  :precondition (and ")):
                #   head_m1 += " "
                #   not_head_m1 += " "
                elif (use_unknown and pal[0] == actionName and pal[1] == predicate):
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
                        use_unknown and pal[0] == actionName and pal[1] == predicate):
                    not_param = "("
                    param = "("
                    # has_something = True
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
                elif (use_unknown and pal[0] == actionName and pal[1] == predicate):
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
                                # print(p)
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

            precond += precond_str
            if (head_m1 + head_m2).strip() != "":
                fd.write(precond + "\n(or \n")
                for k in range(len("  :precondition (and (or ")):
                    fd.write(" ")
                if head_m1 != "":
                    fd.write("(and " + head_m1 + ")")
                if head_m2 != "":
                    fd.write("(and " + head_m2 + ")")
                fd.write(")\n")
            elif equality_needed == False:
                fd.write("()\n")

            if equality_needed == True:
                fd.write("(" + precond + ")\n")

            fd.write("  :effect (and")
            # When (prec(m1)) (eff(m1))
            fd.write(" (when ")
            if (head_m1 + head_m2).strip() != "":
                fd.write("(and\n")
                if head_m1 != "":
                    fd.write(head_m1)
                if head_m2 != "":
                    fd.write(head_m2)
                for k in range(len("  :effect (and (when ")):
                    fd.write(" ")
                fd.write(")\n")
            else:
                fd.write("()\n")

            for k in range(len("  :effect (and (when ")):
                fd.write(" ")
            param1 = ""
            for predicate, value in predicateDict_m1.items():
                pred_split = predicate.split("|")
                pred_name = pred_split[0]

                t_value = copy.deepcopy(value)
                if (t_value[1] == Literal.AN or t_value[1] == Literal.AP):
                    t_value[1] = Literal.ABS
                elif (t_value[1] == Literal.NP):
                    t_value[1] = Literal.POS

                for k in range(len("  :precondition (and (or (and ")):
                    param1 += " "
                if (t_value[1] != Literal.ABS):
                    param1 += "("
                    if (t_value[1] == Literal.NEG):
                        param1 += "not ("
                    param1 += pred_name + "_1"

                    if len(pred_split) > 1:
                        pred_param1s = pred_split[1:]
                        for p in pred_param1s:
                            # print(p)
                            param1 += " ?" + param_ordering[int(p)]

                    param1 += ")"
                    if (t_value[1] != Literal.ABS and t_value[1] != Literal.POS):
                        param1 += ")"

            param2 = ""
            for predicate, value in predicateDict_m2.items():
                pred_split = predicate.split("|")
                pred_name = pred_split[0]

                t_value = copy.deepcopy(value)
                if (t_value[1] == Literal.AN or t_value[1] == Literal.AP):
                    t_value[1] = Literal.ABS
                elif (t_value[1] == Literal.NP):
                    t_value[1] = Literal.POS

                for k in range(len("  :precondition (and (or (and ")):
                    param2 += " "
                if (t_value[1] != Literal.ABS):
                    param2 += "("
                    if (t_value[1] == Literal.NEG):
                        param2 += "not ("
                    param2 += pred_name + "_2"

                    if len(pred_split) > 1:
                        pred_param2s = pred_split[1:]
                        for p in pred_param2s:
                            # print(p)
                            param2 += " ?" + param_ordering[int(p)]

                    param2 += ")"
                    if (t_value[1] != Literal.ABS and t_value[1] != Literal.POS):
                        param2 += ")"

            if (param1 + param2).strip() != "":
                fd.write("(and\n")
                if param1 != "":
                    fd.write(param1)
                if param2 != "":
                    fd.write(param2)
                for k in range(len("  :effect (and (when ")):
                    fd.write(" ")
                fd.write(")\n")
            else:
                fd.write("()\n")

            fd.write(")\n")

            for k in range(len("  :effect (and ")):
                fd.write(" ")
            fd.write("(when ")
            # When (or (!(prec(m1))) (!(prec(m2)))) (create dummy diff)
            if (head_m1 + head_m2 + not_head_m1 + not_head_m2).strip() != "":
                fd.write("(or \n")
                for k in range(len("  :effect (and (when (or ")):
                    fd.write(" ")
                if (head_m1 + not_head_m2).strip() != "":
                    fd.write("(and \n" + head_m1)
                    for k in range(len("  :effect (and (when (or (and ")):
                        fd.write(" ")
                    if not_head_m2.strip() != "":
                        fd.write("(or \n" + not_head_m2)
                        for k in range(len("  :effect (and (when (or (and ")):
                            fd.write(" ")
                        fd.write(")\n")
                    for k in range(len("  :effect (and (when (or ")):
                        fd.write(" ")
                    fd.write(")\n")
                if (head_m2 + not_head_m1).strip() != "":
                    for k in range(len("  :effect (and (when (or ")):
                        fd.write(" ")
                    fd.write("(and \n" + head_m2)
                    for k in range(len("  :effect (and (when (or (and ")):
                        fd.write(" ")
                    if not_head_m1.strip() != "":
                        fd.write("(or \n" + not_head_m1)
                        for k in range(len("  :effect (and (when (or (and ")):
                            fd.write(" ")
                        fd.write(")\n")
                    for k in range(len("  :effect (and (when (or ")):
                        fd.write(" ")
                    fd.write(")\n")
                for k in range(len("  :effect (and (when ")):
                    fd.write(" ")
                fd.write(")\n")
            else:
                fd.write("()\n")

            for k in range(len("  :effect (and (when ")):
                fd.write(" ")
            fd.write("(and \n")
            for k in range(len("  :effect (and (when (and ")):
                fd.write(" ")
            fd.write("(dummy_pred_1)\n")
            for k in range(len("  :effect (and (when (and ")):
                fd.write(" ")
            fd.write("(not(dummy_pred_2))\n")
            for k in range(len("  :effect (and (when ")):
                fd.write(" ")
            fd.write(") \n")
            for k in range(len("  :effect (and ")):
                fd.write(" ")
            fd.write(")\n")
            for k in range(len("  :effect ")):
                fd.write(" ")
            fd.write(")\n")
            fd.write(")\n\n")

        fd.write(")\n")
