#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import sys

from config import *

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))


class ExecutePlan:
    """
    This class executes a plan on a model sarting at an initial state.

    :param targetModel: an instance of class Model on which plan is to be executed
    :type targetModel: object of class Model
    :param init_state: Initial state (list of predicates)
    :type init_state: list of strs
    :param rawPlan: list of actions 
    :type rawPlan: list of strs
    """

    def __init__(self, targetModel, init_state, rawPlan):
        """
        This method creates a new instance of ExecutePlan.

        """

        self.init_state = []
        for p, v in init_state.items():
            for items in v:
                t_init_state = p
                for i in items:
                    t_init_state += "|" + i
                self.init_state.append(t_init_state)

        self.tModel = targetModel
        self.plan = rawPlan

    def execute_plan(self, refinement_dict):
        """
        This method calculates the state after a plan is executed.
        This only works for add delete lists in preconditions and effects.

        """

        actions = self.tModel.actions
        actions = {k.lower(): v for k, v in actions.items()}

        def canActionBeApplied(actions, state, p, refinement_dict):
            plan_split_list = p.split('|')
            action_name = plan_split_list[0]
            action_params = plan_split_list[1:]

            actionPred_original = actions[action_name]

            actionPreds = {}
            for pred, v in actionPred_original.items():
                temp_pred = pred.split("|")[0]
                type_pos = pred.rstrip("|").split("|")[1:]
                for type_positions in type_pos:
                    temp_pred += "|" + action_params[int(type_positions)]
                if temp_pred in actionPreds.keys():
                    v1 = actionPreds[temp_pred]
                    v2 = v

                    if v1 == [Literal.ABS, Literal.ABS]:
                        actionPreds[temp_pred] = v2
                    elif v2 == [Literal.ABS, Literal.ABS]:
                        actionPreds[temp_pred] = v1
                    else:
                        print("Failed in canApplyAction")
                        return False, None
                else:
                    actionPreds[temp_pred] = v

            failed = False
            for pred, val in actionPreds.items():
                t_value = copy.deepcopy(val)
                if (t_value[0] == Literal.AN or t_value[0] == Literal.AP):
                    t_value[0] = Literal.ABS
                elif (t_value[0] == Literal.NP):
                    t_value[0] = Literal.POS

                if ((t_value[0] == Literal.POS) and (pred not in state)) or \
                        (t_value[0] == Literal.NEG and (pred in state)):
                    failed = True
                    pred_params = pred.split("|")[1:]
                    pred_name = pred.split("|")[0]
                    for param in pred_params:
                        indx = action_params.index(param)
                        if indx != -1:
                            pred_name += "|" + str(indx)

                    if (refinement_dict[(action_name, pred_name, Location.PRECOND)] == False):
                        return False, [pred_name, t_value[0]]
                    else:
                        continue

            return not (failed), None

        def applyAction(actions, state, p):
            plan_split_list = p.split('|')
            action_name = plan_split_list[0]
            action_params = plan_split_list[1:]

            actionPred_original = actions[action_name]

            actionPreds = {}
            for pred, v in actionPred_original.items():
                temp_pred = pred.split("|")[0]
                type_pos = pred.rstrip("|").split("|")[1:]
                for type_positions in type_pos:
                    temp_pred += "|" + action_params[int(type_positions)]
                if temp_pred in actionPreds.keys():
                    v1 = actionPreds[temp_pred]
                    v2 = v
                    if v1 == [Literal.ABS, Literal.ABS]:
                        actionPreds[temp_pred] = v2
                    elif v2 == [Literal.ABS, Literal.ABS]:
                        actionPreds[temp_pred] = v1
                    else:
                        return False, None
                else:
                    actionPreds[temp_pred] = v

            tempState = copy.deepcopy(state)

            for pred, val in actionPreds.items():
                t_value = copy.deepcopy(val)
                if (t_value[1] == Literal.AN or t_value[1] == Literal.AP):
                    t_value[1] = Literal.ABS
                elif (t_value[1] == Literal.NP):
                    t_value[1] = Literal.POS

                if (t_value[1] == Literal.POS):
                    tempState.add(pred)
                elif (t_value[1] == Literal.NEG):
                    # If it was absent in precondition, we can make it negative.
                    if pred in tempState:
                        tempState.remove(pred)
                    elif (t_value[0] == Literal.ABS):
                        continue
                    else:
                        return False, None

            return True, tempState

        initialState = set(self.init_state)
        currState = copy.deepcopy(initialState)

        plan_index = 0
        for p in self.plan:
            can_apply_action, issue = canActionBeApplied(actions, currState, p, refinement_dict)
            if can_apply_action:
                is_ok, newState = applyAction(actions, currState, p)
                if is_ok == False:
                    return False, None, None
                currState = copy.deepcopy(newState)
            else:
                return False, currState, plan_index

            plan_index += 1
        return True, currState, plan_index
