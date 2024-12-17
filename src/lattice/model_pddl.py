#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from collections import OrderedDict
from itertools import permutations, combinations, product
from pddlgym.structs import LiteralConjunction
from pddlgym.structs import Literal as Ltr

from config import *


class State:
    def __init__(self, state, objects):
        self.state = state
        self.objects = objects

    def __str__(self):
        return str(self.state) + str(self.objects)


class Model(object):
    """
    This class defines the AI planning model that we assume the model to have.
    Each model is defined in terms of predicates and actions.

    :param predicates: dictionary of predicates and their parameters
    :type predicates: dict of (str, int)
    :param actions: dictionary of actions (action, dict(predicate,[pre,eff]))
    :type actions: dict of (str, dict(str, list))

    For each predicate in an action, pre and eff are 0 or 1.

    0 means, that predicate appears as a negative literal,

    1 means, that predicate appears as a positive literal.

    """

    def __init__(self, predicates, actions):
        """
        This method creates a new instance of Model.

        """

        self.predicates = {}
        for p in predicates:
            self.predicates[p] = predicates[p]

        self.actions = {}
        for a in actions:
            self.actions[a] = actions[a]

        self.discarded = False

    def __eq__(self, other):
        return (self.predicates == other.predicates) and \
               (self.actions == other.actions)

    def __hash__(self):
        return hash(tuple(self.actions))

    def __ne__(self, other):
        return not self.__eq__(other)

    def print_predicates(self):
        """
        This method prints the details of the predicates of the model.

        .. note::

            | Output format will be:
            | -------------------------Predicates-------------------------
            | Predicate Name        | Number of Parameters
            | -------------------------------- -----------------------------------

        :rtype: None

        """
        print("\n\n------------------Predicates------------------\n")
        print("Predicate Name          | Number of Parameters")
        print("----------------------------------------------")
        space_to_leave = len("Predicate Name          ")
        for key, value in self.predicates.items():
            print(key, end="")
            length = len(str(key))
            for i in range(space_to_leave - length):
                print(" ", end="")
            print("| " + str(value))
        print("----------------------------------------------\n")

    def print_actions(self):
        """
        This method prints the details of the actions of the model.

        .. note::

            | Output format will be:
            | ---------------Actions---------------
            | Action Name        | Predicates
            | ------------------- ----------------------

        :rtype: None

        """
        print("--------------------Actions-------------------\n")
        print("Action Name           | Predicates")
        print("----------------------------------------------")
        space_to_leave = len("Predicate Name        ")
        for key, preds in self.actions.items():
            print(key, end="")
            length = len(str(key))
            for i in range(space_to_leave - length):
                print(" ", end="")
            print("| pre :", end="")
            print_comma = False
            for pred, value in preds.items():
                if not print_comma:
                    print_comma = True
                else:
                    print(",", end="")
                if value[0] == Literal.NEG:
                    print(" !" + str(pred), end="")
                elif value[0] == Literal.POS:
                    print(" " + str(pred), end="")
                else:
                    print_comma = False
            print("")
            for i in range(space_to_leave):
                print(" ", end="")
            print("| eff :", end="")
            print_comma = False
            for pred, value in preds.items():
                if not print_comma:
                    print_comma = True
                else:
                    print(",", end="")
                if value[1] == Literal.NEG:
                    print(" !" + str(pred), end="")
                elif value[1] == Literal.POS:
                    print(" " + str(pred), end="")
                else:
                    print_comma = False
            print("")
        print("----------------------------------------------\n\n")

    def print_model(self):
        """
        This method prints the details of the model.

        :rtype: None

        """
        self.print_predicates()
        self.print_actions()

    def update_actions(self, new_actions):
        """
        This method updates the mapping of actions to predicates.

        :param new_actions:
        :type new_actions:

        :return: true for success, false otherwise
        :rtype: bool

        """

        for a in new_actions:
            self.actions[a] = new_actions[a]

    def add_predicates(self, predicates, action_pred_dict, actions):
        """
        This method adds the predicates to the model's predicate list.

        :param predicates: list of predicates to be removed
        :type predicates: list of str
        :param action_pred_dict:
        :type action_pred_dict:
        :param actions:
        :type actions:

        :rtype: None

        """
        for p in predicates:
            self.predicates[p] = predicates[p]
            for a in actions:
                all_poss_preds = action_pred_dict[a]
                action_preds = list(filter(lambda x: p in x, all_poss_preds))
                if p in action_preds:
                    self.actions[a].update({p: [Literal.ABS, Literal.ABS]})

    def write_model_to_file(self, fd, domain_name, pred_type_mapping, action_parameters, objects=None):
        """
        This method creates files.

        :param fd: file descriptor of the pddl file in which model will be written
        :type fd: file descriptor
        :param domain_name: domain name of the model
        :type domain_name: str
        :param pred_type_mapping:
        :type pred_type_mapping:
        :param action_parameters:
        :type action_parameters:
        :param objects:
        :type objects:

        :rtype: None

        """
        if objects is None:
            objects = dict()
        fd.write("(define (domain " + domain_name + ")\n")
        fd.write("(:requirements :strips :typing :equality)\n")

        # Typing
        fd.write("(:types")
        for t in objects.keys():
            fd.write(" " + t)
        fd.write(")\n")

        # Predicates
        fd.write("(:predicates ")
        count = 0
        preds_printed = []
        for key, value in self.predicates.items():
            params = ""
            cnt = 0
            pred_name = key.split("|")[0]
            if pred_name in preds_printed:
                continue
            else:
                preds_printed.append(pred_name)

            if pred_name.split("_")[-1] in ["1", "2"]:
                actual_pred_name_splits = pred_name.split("_")[0:-1]
                actual_pred_name = '_'.join(actual_pred_name_splits)
            else:
                actual_pred_name = pred_name
            for val in pred_type_mapping[actual_pred_name]:
                params = params + " ?" + val[0] + str(cnt) + " - " + val
                cnt += 1

            if count > 0:
                fd.write("\n")
                for k in range(len("(:predicates ")):
                    fd.write(" ")
            fd.write("(" + pred_name + params + ")")
            count += 1
        fd.write(")\n\n")

        # Actions
        for actionName, predicateDict in self.actions.items():
            head = "(:action " + actionName + "\n" + "  :parameters"
            fd.write(head)
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

            fd.write("  :precondition (and")
            equality_needed = False
            if param_count > 1:
                equality_needed = True

            if equality_needed:
                combs = combinations(list(range(0, param_count)), 2)
                for c in combs:
                    fd.write("(not (= ")
                    for j in range(2):
                        i = c[j]
                        fd.write("?" + param_ordering[i])
                        if j == 0:
                            fd.write(" ")
                        else:
                            fd.write(")) ")

            for predicate, value in predicateDict.items():
                pred_split = predicate.split("|")
                pred_name = pred_split[0]

                t_value = copy.deepcopy(value)
                if t_value[0] != Literal.ABS:
                    param = " ("
                    if t_value[0] == Literal.NEG:
                        param += "not ("
                    elif t_value[0] == Literal.AN:
                        param += "0/- ("
                    elif t_value[0] == Literal.AP:
                        param += "0/+ ("
                    elif t_value[0] == Literal.NP:
                        param += "+/- ("
                    param += pred_name

                    if len(pred_split) > 1:
                        pred_params = pred_split[1:]
                        for p in pred_params:
                            print(p)
                            param += " ?" + param_ordering[int(p)]
                    param += ")"
                    if t_value[0] != Literal.ABS and t_value[0] != Literal.POS:
                        param += ")"
                    fd.write(param)
            fd.write(")\n")

            fd.write("  :effect (and")
            for predicate, value in predicateDict.items():
                pred_split = predicate.split("|")
                pred_name = pred_split[0]
                if value[1] != Literal.ABS:
                    param = " ("
                    if value[1] == Literal.NEG:
                        param += "not ("
                    param += pred_name

                    if len(pred_split) > 1:
                        pred_params = pred_split[1:]
                        for p in pred_params:
                            param += " ?" + param_ordering[int(p)]

                    param += ")"
                    if value[1] == Literal.NEG:
                        param += ")"
                    fd.write(param)
            fd.write("))\n\n")
        fd.write(")\n")


class Lattice(object):
    """
    This class defines the lattice where each node (class LatticeNode)
    is a collection of models (class Models).

    """

    def __init__(self):
        self.nodes = {}
        # Refinement = 0 means precondition refined first
        # Refinement = 1 means effects refined first
        self.refinement = Location.EFFECTS

    def add_node(self, node_id, node):
        self.nodes[node_id] = node


class LatticeNode(object):
    """
    This class defines the AI planning model that we have the model to have.
    Each model is defined in terms of predicates and actions.

    :param models:  list of models
    :type models: list of Model objects
    :param predicates: dictionary of predicates and their parameters
    :type predicates: dict of (str, int)

    """

    def __init__(self, lattice, models, predicates, action_pred_dict=None):
        """
        This method creates a new instance of Model.

        """

        if action_pred_dict is None:
            action_pred_dict = {}
        self.models = models
        self.predicates = list(predicates.keys())
        self.id = hash(tuple(sorted(self.predicates)))

        self.lattice = lattice
        self.lattice.add_node(self.id, self)
        self.action_pred_dict = action_pred_dict

    def add_models(self, models):
        temp_models = self.models
        for i in range(len(temp_models)):
            temp_models[i].discarded = False
        for m in models:
            discarded = m.discarded
            m.discarded = False
            if m in temp_models:
                temp_models.remove(m)
            m.discarded = discarded
            self.models.append(m)

    @staticmethod
    def act_pred_mapping(action_list, ref, modes=(Literal.ABS, Literal.NEG, Literal.POS)):
        pre_list = [[Literal.ABS]]
        eff_list = [[Literal.ABS]]

        if ref == Location.ALL or ref == Location.PRECOND:
            pre_list = [list(i) for i in product(modes)]
        if ref == Location.ALL or ref == Location.EFFECTS:
            eff_list = [list(i) for i in product(modes)]
        pre_eff_list = [list(i[0] + i[1]) for i in list(product(pre_list, eff_list))]

        # Stores which action will have the predicate with what precondition and effect
        action_mapping = []
        pred_list = list(product(pre_eff_list, repeat=len(action_list)))
        act_list = list(combinations(action_list, len(action_list)))

        # Mapping of actions to predicates' variations
        action_mapping.append(list(product(act_list, pred_list)))

        return list(action_mapping)

    @staticmethod
    def generate_preds_for_action(predicate, action, pred_type_mapping, action_parameters):
        for p in pred_type_mapping[predicate]:
            if p not in action_parameters[action]:
                return None

        need_multiple_mapping = False
        pred_type_count = {}
        for t in action_parameters[action]:
            if t not in pred_type_count.keys():
                pred_type_count[t] = action_parameters[action].count(t)
            else:
                continue

            if pred_type_count[t] > 1 and t in pred_type_mapping[predicate]:
                need_multiple_mapping = True

            if pred_type_mapping[predicate].count(t) > 1:
                need_multiple_mapping = True

        if not need_multiple_mapping:
            updated_predicate = str(predicate)
            for p in pred_type_mapping[predicate]:
                if p in action_parameters[action]:
                    updated_predicate += "|" + str(action_parameters[action].index(p))
                else:
                    print("Error")
                    exit(1)
            return [updated_predicate]
        else:
            type_combination_dict = OrderedDict()
            for t in pred_type_mapping[predicate]:
                if pred_type_count[t] > 1 and t not in type_combination_dict.keys():
                    type_count_in_predicate = pred_type_mapping[predicate].count(t)
                    # Locations of type t in action_parameters[action]'s paramteres
                    pred_locations = [i for i, x in enumerate(action_parameters[action]) if x == t]
                    type_combinations = permutations(pred_locations, type_count_in_predicate)
                    type_combination_dict[t] = list(type_combinations)

            final_combinations = list(product(*list(type_combination_dict.values())))
            updated_predicate_list = []
            for comb in final_combinations:
                pred_type_count_temp = {}
                updated_predicate = str(predicate)
                to_remove = []  # to store preds like on|0|0
                for p in pred_type_mapping[predicate]:
                    if p not in pred_type_count_temp.keys():
                        pred_type_count_temp[p] = 0
                    else:
                        pred_type_count_temp[p] = pred_type_count_temp[p] + 1

                    if p not in type_combination_dict.keys():
                        updated_predicate += "|" + str(action_parameters[action].index(p))
                        if pred_type_mapping[predicate].count(p) > 1:
                            to_remove.append(updated_predicate)
                    else:
                        index_to_search = list(type_combination_dict.keys()).index(p)
                        updated_predicate += "|" + str(comb[index_to_search][pred_type_count_temp[p]])

                updated_predicate_list.append(updated_predicate)
                for r in to_remove:
                    if r in updated_predicate_list:
                        updated_predicate_list.remove(r)

            return updated_predicate_list

    def get_specific_children(self, model, predicate, ref, action, modes):
        child_predicates = self.predicates
        child_predicates.append(predicate)

        child_id = hash(tuple(sorted(child_predicates)))

        child_models = []
        if str(predicate) not in model.predicates.keys():
            model.predicates[str(predicate)] = predicate
        action_mapping = self.act_pred_mapping([action], ref, modes)
        for i in action_mapping:
            for actionNames, mappings in i:
                new_child = copy.deepcopy(model)
                _update_actions = {}
                literal = predicate
                if ref == Location.PRECOND:
                    precond = copy.deepcopy(model.actions[actionNames[0]].preconds.literals)
                    new_child.mode = mappings[0][0]
                    if new_child.mode == Literal.POS:
                        precond.append(literal)
                        new_child.actions[actionNames[0]].preconds = LiteralConjunction(list(precond))
                    elif new_child.mode == Literal.NEG:
                        precond.append(literal.inverted_anti)
                        new_child.actions[actionNames[0]].preconds = LiteralConjunction(list(precond))
                if ref == Location.EFFECTS:
                    new_child.mode = mappings[0][1]
                    effect = copy.deepcopy(model.actions[actionNames[0]].effects.literals)
                    if new_child.mode == Literal.POS:
                        effect.append(literal)
                        new_child.actions[actionNames[0]].effects = LiteralConjunction(list(effect))
                        # model.actions[actionNames[0]].preconds.append(literal)
                    elif new_child.mode == Literal.NEG:
                        effect.append(literal.inverted_anti)
                        new_child.actions[actionNames[0]].effects = LiteralConjunction(list(effect))

                    if model.discarded:
                        new_child.discarded = True

                child_models.append(new_child)

        # Assuming that we will never need child
        if child_id in self.lattice.nodes.keys():
            child_node = self.lattice.nodes[child_id]
            child_node.add_models(child_models)
        else:
            pred_dict = {}
            for p in child_predicates:
                pred_dict[p] = 0
            child_node = LatticeNode(self.lattice, child_models, pred_dict, self.action_pred_dict)
        return child_node

    def get_model_partitions(self, model, predicate, ref, action, modes):
        child_node = self.get_specific_children(model, predicate, ref, action, modes)
        child_models = child_node.models

        return child_models
