#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import os
import subprocess
from collections import OrderedDict

import numpy as np
# import pddlgym
from utils.parser import PDDLDomainParser, structs

import utils.translate.pddl_fd as pddl
import utils.translate.pddl_parser as pddl_parser
from config import *
from utils import FileUtils


def extract_task(domain_file_path, problem_file_path):
    # Extract the domain specific args.
    domain_pddl = pddl_parser.pddl_file.parse_pddl_file(
        "domain", domain_file_path)
    domain_name, \
    domain_requirements, \
    types, \
    type_dict, \
    constants, \
    predicates, \
    predicate_dict, \
    functions, \
    actions, \
    axioms = pddl_parser.parsing_functions.parse_domain_pddl(
        domain_pddl)

    task_pddl = pddl_parser.pddl_file.parse_pddl_file(
        "task", problem_file_path)
    task_name, \
    task_domain_name, \
    task_requirements, \
    objects, \
    init, \
    goal, \
    use_metric = pddl_parser.parsing_functions.parse_task_pddl(
        task_pddl, type_dict, predicate_dict)

    assert domain_name == task_domain_name
    requirements = pddl.Requirements(sorted(set(
        domain_requirements.requirements +
        task_requirements.requirements)))
    objects = constants + objects
    pddl_parser.parsing_functions.check_for_duplicates(
        [o.name for o in objects],
        errmsg="error: duplicate object %r",
        finalmsg="please check :constants and :objects definitions")

    #############################
    init += [pddl.Atom("=", (obj.name, obj.name))
             for obj in objects]
    #############################

    task = pddl.Task(domain_name,
                     task_name,
                     requirements,
                     types,
                     objects,
                     predicates,
                     functions,
                     init,
                     goal,
                     actions,
                     axioms,
                     use_metric)
    return task


def check_nested(test_dict):
    for key1, val in test_dict.items():
        for key2 in test_dict.keys():
            if key2 in val and key1 != 'object':
                return True, key2, key1
    return False, None, None


class PredicateDetails:
    def __init__(self, literal, param_dict, predTypeMapping):
        name_set = False
        self.param_matching = OrderedDict()
        try:
            for param in literal.variables:
                self.param_matching[param.name] = param_dict[param.name]
        except KeyError as e:
            print("KeyError")
        self.isnegative = literal.is_anti
        if literal.is_negative == True and literal.is_anti == False:
            self.isnegative = True
        for pred in predTypeMapping:
            if literal.predicate.name in pred and sorted(list(self.param_matching.values())) == sorted(
                    list(predTypeMapping[pred])):
                self.name = pred
                name_set = True
        if not name_set:
            self.name = literal.predicate.name
            print("pred not found")

    def __str__(self):
        return self.name + "(" + str(self.param_matching) + ")" + str(self.isnegative)


class ActionDetails:
    def __init__(self, action, param_types, predTypeMapping):
        self.name = action.name
        self.precondition = []
        self.effects = []
        self.param_matching = OrderedDict()
        self.precondition_literal_names = []
        self.add_effects_literal_names = []
        self.del_effects_literal_names = []

        for i, param in enumerate(action.params):
            self.param_matching[param.name] = param_types[i]
        try:
            if isinstance(action.preconds,  structs.LiteralConjunction):
                [self.precondition.append(PredicateDetails(lit, self.param_matching, predTypeMapping)) for lit in
                 action.preconds.literals]
                [self.precondition_literal_names.append(p.name) for p in self.precondition]

            elif isinstance(action.preconds, structs.Literal):
                self.precondition.append(PredicateDetails(action.preconds, self.param_matching, predTypeMapping))

            else:
                print("Some other action precondition type")
        except AttributeError as e:
            print("Attribute Error!")

        for lit in action.effects.literals:
            self.effects.append(PredicateDetails(lit, self.param_matching, predTypeMapping))

        for p in self.effects:
            if p.isnegative:
                self.del_effects_literal_names.append(p.name)
            else:
                self.add_effects_literal_names.append(p.name)

    def __str__(self):
        return self.name + "\nParams: [\n" + str(self.params) + "\n]\n Precond:[\n " + str(
            self.precondition) + "\n] Add_effects:[\n" + str(self.add_effects) + "\n] Del_effects:[\n" + str(
            self.del_effects)


def generate_ds(domain_file, problem_file):
    task = extract_task(domain_file, problem_file)
    domain_parser = PDDLDomainParser(domain_file)
    domain_parser._parse_domain()
    ##########pddlgym's parser###############
    predicates = domain_parser.predicates
    operators = domain_parser.operators
    #########################################
    predTypeMapping = {}
    absActParamType = {}
    reverse_types = {}
    types = {}
    objects = {}
    init_state = []
    ################reverse typ mapping#######################
    for typ in task.types:
        if typ.basetype_name != None:
            if typ.basetype_name not in reverse_types.keys():
                reverse_types[typ.basetype_name] = [typ.name]
            else:
                reverse_types[typ.basetype_name].append(typ.name)

    for obj in task.objects:
        if str(obj.type_name) not in objects.keys():
            objects[str(obj.type_name)] = [str(obj.name)]
        else:
            objects[str(obj.type_name)].append(str(obj.name))

    # check for heirarchy....should be using a tree for this instead####
    while True:
        is_nested, nested_key, parent_key = check_nested(reverse_types)
        if is_nested:
            reverse_types[parent_key].remove(nested_key)
            reverse_types[parent_key].extend(reverse_types[nested_key])
        else:
            break

    ####################predTypeMapping######################
    for pred_name in predicates.keys():
        pred = predicates[pred_name]
        args = pred.var_types
        nested = False
        for i, arg in enumerate(args):
            if arg in reverse_types.keys():
                args[i] = reverse_types[arg]
                nested = True
            else:
                args[i] = [args[i]]
        if nested:
            args = itertools.product(*args)
            for i, arg_p in enumerate(args):
                predTypeMapping[pred_name + '-' + str(i + 1)] = arg_p
        else:
            predTypeMapping[pred_name] = list(itertools.chain.from_iterable(args))

    ####################init_state###############################
    for item in task.init:
        if item.predicate == "=":
            continue

        temp_pred = copy.deepcopy(item.predicate)
        if len(item.args) > 0:
            for _arg in item.args:
                temp_pred += "|" + _arg
        init_state.append(temp_pred)
    ####################action_parameters########################
    action_parameters = {}
    action_details = {}
    for op_name in operators:
        op = operators[op_name]
        op_params = [];
        [op_params.append(i.var_type) for i in op.params]
        nested = False
        for i, arg in enumerate(op_params):
            if arg in reverse_types.keys():
                op_params[i] = reverse_types[arg]
                nested = True
                heirarchial_types = True
            else:
                op_params[i] = [op_params[i]]
        if nested:
            args = itertools.product(*op_params)
            for i, arg_p in enumerate(args):
                action_parameters[op_name + '-' + str(i + 1)] = arg_p
                action_details[op_name + '-' + str(i + 1)] = ActionDetails(op, arg_p, predTypeMapping)
        else:
            action_parameters[op_name] = list(itertools.chain.from_iterable(op_params))
            action_details[op_name] = ActionDetails(op, list(itertools.chain.from_iterable(op_params)), predTypeMapping)
    ##########################abstract_model#######################
    abstract_model = {}
    for action in action_parameters.keys():
        abstract_model[action] = {}
    ########################agent_model#################
    agent_model = {}
    for action_name, action in action_details.items():
        agent_model[action_name] = {}
        action_params = list(action.param_matching.values())
        for pred, pred_params in predTypeMapping.items():
            try:
                if len(set(pred_params).difference(set(action_params))) == 0:
                    # check for multiple presence
                    param_indices = []
                    [param_indices.append(list(np.where(np.array(action_params) == p))[0].tolist()) for p in
                     pred_params]
                    combinations = list(itertools.product(*param_indices))
                    if len(combinations) > 1:
                        for c in combinations:
                            if len(c) != len(set(c)):
                                continue
                            agent_model[action_name][pred + '|' + "|".join(map(str, c))] = [Literal.ABS, Literal.ABS]
                            for l in action.precondition:
                                if l.name == pred:
                                    action_local_params = [list(action.param_matching.keys())[i] for i in list(c)]
                                    if action_local_params == list(l.param_matching.keys()):
                                        if l.isnegative:
                                            agent_model[action_name][pred + '|' + "|".join(map(str, c))][
                                                0] = Literal.NEG
                                        else:
                                            agent_model[action_name][pred + '|' + "|".join(map(str, c))][
                                                0] = Literal.POS
                            for l in action.effects:
                                if l.name == pred:
                                    action_local_params = [list(action.param_matching.keys())[i] for i in list(c)]
                                    if action_local_params == list(l.param_matching.keys()):
                                        if l.isnegative:
                                            agent_model[action_name][pred + '|' + "|".join(map(str, c))][
                                                1] = Literal.NEG
                                        else:
                                            agent_model[action_name][pred + '|' + "|".join(map(str, c))][
                                                1] = Literal.POS
                    else:
                        if len(combinations[0]) != len(set(combinations[0])):
                            continue
                        str_app = "|".join(map(str, combinations[0]))
                        if str_app:
                            modified_pred = pred + '|' + str_app
                        else:
                            modified_pred = copy.deepcopy(pred)
                        agent_model[action_name][modified_pred] = [Literal.ABS, Literal.ABS]
                        for l in action.precondition:
                            if l.name == pred:
                                if l.isnegative:
                                    agent_model[action_name][modified_pred][0] = Literal.NEG
                                else:
                                    agent_model[action_name][modified_pred][0] = Literal.POS
                        for l in action.effects:
                            if l.name == pred:
                                if l.isnegative:
                                    agent_model[action_name][modified_pred][1] = Literal.NEG
                                else:
                                    agent_model[action_name][modified_pred][1] = Literal.POS
            except KeyError as e:
                print("Key Error")
    return action_parameters, predTypeMapping, agent_model, abstract_model, objects, reverse_types, init_state, task.domain_name


def get_plan(domain_file, problem_file):
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

    :rtype: list

    """
    plan = ""

    if PLANNER == "FF":
        result_file = temp_output_file
        param = FF_PATH + "ff"
        param += " -o " + domain_file
        param += " -f " + problem_file
        param += " > " + result_file

        p = subprocess.Popen([param], shell=True)
        p.wait()
        plan = FileUtils.get_plan_from_file(result_file)

    elif PLANNER == "FD":
        cmd = FD_PATH + 'fast-downward.py ' + domain_file + ' ' + problem_file + ' --search "astar(lmcut())"'
        plan = os.popen(cmd).read()
        proc_plan = plan.split('\n')
        cost = [i for i, s in enumerate(proc_plan) if 'Plan cost:' in s]
        if 'Solution found!' not in proc_plan:
            print("No Solution")
            return [], 0
        plan = proc_plan[proc_plan.index('Solution found!') + 2: cost[0] - 1]

    return plan
