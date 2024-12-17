#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import importlib
import itertools
import pickle
import pprint
import time
import subprocess
from collections import Counter, OrderedDict
from itertools import combinations

from config import *
from query import ExecutePlan
from lattice import LatticeNode, Lattice, State
from utils import *


class AgentInterrogation:
    """
    :param agent: actual agent model
    :type agent: object of class Model
    :param abstract_model: model at the most abstracted node in the lattice
    :type abstract_model: object of class Model
    :param objects: Initial state of the problem
    :type objects: dict of (str,str)
    :param domain_name: Name of the domain
    :type domain_name: String
    :param abstract_predicates:
    :param pred_type_mapping:
    :param action_parameters:
    :param types:

    """

    def __init__(self, agent, abstract_model, objects, domain_name,
                 abstract_predicates, pred_type_mapping, action_parameters, types):
        self.agent = agent
        self.abstract_model = abstract_model
        self.objects = objects
        self.domain_name = domain_name
        self.abstract_predicates = abstract_predicates
        self.pred_type_mapping = pred_type_mapping
        self.action_parameters = action_parameters
        self.types = types
        self.location = Location.ALL
        self.difficult_pal_tuple = []
        self.queries = {}
        self.pal_tuples_fixed = 0
        self.failed_plans = []
        self.agent_query_total = 0
        self.query_old = 0
        self.query_new = 0
        self.invalid_init_state = 0
        self.agent_cant_execute = 0
        self.pal_tuple_dict = {}
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        self.init_result_file()
        self.action_count = 0
        self.predicate_count = 0
        self.mod_pred_count = 0
        self.start_time = time.time()
        self.data_dict = OrderedDict()
        self.random_state_file = RANDOM_STATE_FOLDER + "random_" + domain_name + ".pkl"

    def init_result_file(self):
        f = open(final_result_dir + self.domain_name + "-" + final_result_prefix + "-" + self.timestr + ".csv", "w")
        f.write("domain_name, #action, #predicate, #modified_predicates, #pal_tuples_fixed, #queries_total, "
                "#agent_failed, #repeated_queries, #unique_queries, #model_similarity, #time_elapsed, pal_tuple\n")
        f.close()

    def fix_pal_tuple(self, pal_tuple, valid_models):
        valid_models = [i for i in valid_models if not i.discarded]
        assert (len(valid_models) <= 3)
        print("Valid Model Len: ", len(valid_models))
        print("Fixed pal tuple: ", pal_tuple)
        mod_sim = 0
        if not self.pal_tuple_dict[pal_tuple]:
            self.pal_tuple_dict[pal_tuple] = True
            self.pal_tuples_fixed += 1
            if not is_simulator_agent:
                all_diff = []
                for m in valid_models:
                    model_diff = get_model_difference(self.agent.agent_model, m, self.pal_tuple_dict)
                    print("Model Similarity: ", 1 - model_diff)
                    all_diff.append(model_diff)
                    mod_sim = 1 - max(all_diff)
                f = open(final_result_dir + self.domain_name + "-" + final_result_prefix + "-" + self.timestr + ".csv",
                         "a")
                f.write(",".join([self.domain_name, str(self.action_count), str(self.predicate_count),
                                  str(self.mod_pred_count), str(self.pal_tuples_fixed), str(self.agent_query_total),
                                  str(self.agent_cant_execute), str(self.query_old), str(self.query_new), str(mod_sim),
                                  str(time.time() - self.start_time), str(pal_tuple), "\n"]))
                f.close()

                self.data_dict[self.pal_tuples_fixed] = [self.query_new, mod_sim, time.time() - self.start_time]
        return

    def ask_query(self, init_state, plan, partial_init_check=False):
        query = dict()
        query['init_state'] = copy.deepcopy(State(init_state, self.objects))
        query['plan'] = copy.deepcopy(plan)
        self.agent_query_total += 1

        key = str("||".join(sorted(state_to_set(query['init_state'].state)))) + "|||" + str("||".join(query['plan']))

        if key not in self.queries:
            self.query_new += 1
            is_executable_agent, failure_index, possible_state = self.agent.run_query(query, self.pal_tuple_dict,
                                                                                      partial_init_check)
            self.queries[key] = [is_executable_agent, failure_index, possible_state]
            if failure_index == -1:
                self.invalid_init_state += 1
            if not is_executable_agent:
                self.agent_cant_execute += 1
        else:
            self.query_old += 1
            return self.queries[key]

        return is_executable_agent, failure_index, possible_state

    @staticmethod
    def reject_action_pred_combo(action, pred, rejected_literal, position, action_pred_comb_dict):
        """
        :param action:
        :param pred:
        :param rejected_literal:
        :param position:
        :param action_pred_comb_dict:
        :return:
        """

        # position = 0 for precondition, 1 for effect
        if action not in action_pred_comb_dict.keys():
            action_pred_comb_dict[action] = {pred: [[rejected_literal, position]]}
        else:
            rejected_preds = action_pred_comb_dict[action]
            if pred not in rejected_preds.keys():
                print("Some error, wrong call")
            else:
                rejected_pred_vals = rejected_preds[pred]
                if [rejected_literal, position] not in rejected_pred_vals:
                    rejected_pred_vals.append([rejected_literal, position])
                    action_pred_comb_dict[action][pred] = rejected_pred_vals
                else:
                    return action_pred_comb_dict
        return action_pred_comb_dict

    @staticmethod
    def is_model_rejectable(model, action_pred_comb_dict):
        """
        :param model:
        :param action_pred_comb_dict:
        :return:
        """
        for action in action_pred_comb_dict.keys():
            for pred, val in action_pred_comb_dict[action].items():
                for v in val:
                    position = v[1]
                    literal = v[0]
                    try:
                        if model.actions[action][pred][position] == literal:
                            return True
                        else:
                            return False
                    except IndexError:
                        return False

    def propagate_refinement_in_models(self, valid_models, issue, old_refinement, location=Location.PRECOND):
        """
        :param valid_models:
        :param issue:
        :param old_refinement:
        :param location:
        :return:
        """
        action = issue[0]
        pred = issue[1]
        mode = issue[2]

        valid_models = [i for i in valid_models if not i.discarded]

        for m in valid_models:
            if old_refinement[1] in (m.actions[old_refinement[0]]).keys():
                if m.actions[old_refinement[0]][old_refinement[1]] == [Literal.ABS, Literal.ABS] and \
                        not self.pal_tuple_dict[(old_refinement[0], old_refinement[1], Location.PRECOND)] and \
                        not self.pal_tuple_dict[(old_refinement[0], old_refinement[1], Location.EFFECTS)]:
                    m.actions[old_refinement[0]].pop(old_refinement[1], None)

        valid_models = list(set(valid_models))

        for m in valid_models:
            if pred not in m.predicates.keys():
                m.predicates[pred] = 0
            if pred in m.actions[action]:
                if m.actions[action][pred][location - 1] != mode and \
                        m.actions[action][pred][location - 1] != Literal.ABS:
                    m.discarded = True
                else:
                    m.actions[action][pred][location - 1] = mode
            else:
                if location == Location.PRECOND:
                    m.actions[action][pred] = [mode, Literal.ABS]
                elif location == Location.EFFECTS:
                    m.actions[action][pred] = [Literal.ABS, mode]

        return valid_models

    def get_next_pal_tuple(self, action="", predicate="", location=0):
        """

        :param action:
        :param predicate:
        :param location:
        :return:
        """
        for key, val in self.pal_tuple_dict.items():
            # Match action, predicate and refinement passed into the parameters and return if
            # refinement not already done for those params
            if not val and (action == "" or action == key[0]) and (predicate == "" or predicate == key[1]) and \
                    (location == 0 or location == key[2]) and key not in self.difficult_pal_tuple:
                return key

        for key, val in self.pal_tuple_dict.items():
            # Match action, predicate and refinement passed into the parameters and return if
            # refinement not already done for those params
            if not val and (action == "" or action == key[0]) and \
                    (predicate == "" or predicate == key[1]) and (location == 0 or location == key[2]):
                return key

        return None

    def get_possible_init_states(self, init_state):
        """
        :param init_state:
        :return:
        """

        def powerset(iterable):
            """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
            s = list(iterable)
            return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

        new_init_states = [{}]
        state_objects = []
        for pred, _type in self.pred_type_mapping.items():
            if _type == [] and pred not in init_state.keys():
                init_state[pred] = [()]

        _init_state_count = 0
        for pred, _type in self.pred_type_mapping.items():
            new_set = []
            if not _type:
                new_set = [()]
            for t in _type:
                new_set.append(self.objects[t])
            new_list = list(itertools.product(*new_set))
            pow_set = list(powerset(new_list))
            temp_init_states = copy.deepcopy(new_init_states)

            for t in temp_init_states:
                for p in pow_set:
                    if p != () or not _type:
                        new_temp_state_val = copy.deepcopy(t)
                        if _type:
                            new_temp_state_val[pred] = list(p)
                        else:
                            new_temp_state_val[pred] = [()]
                        init_state_obj = State(new_temp_state_val, self.objects)
                        state_objects.append(init_state_obj)
                        new_init_states.append(new_temp_state_val)
                        _init_state_count += 1

                if _init_state_count > 50:
                    for _t in temp_init_states:
                        p = pow_set[-1]
                        if p != () or not _type:
                            new_temp_state_val = copy.deepcopy(_t)
                            if _type:
                                new_temp_state_val[pred] = list(p)
                            else:
                                new_temp_state_val[pred] = [()]
                            init_state_obj = State(new_temp_state_val, self.objects)
                            state_objects.append(init_state_obj)
                            new_init_states.append(new_temp_state_val)
                            _init_state_count += 1
                    break
            if _init_state_count > 50:
                break

        return state_objects

    def populate_action_pred_list(self, latt):
        """
        This method populates the action_pred_list.
        action_preds_list is used to store the modified predicates ("ontable|0", etc) corresponding to each action
        For eg:  actions_pred_list = {'pick-up': ['ontable|0', 'clear|0', 'handempty'],
        'put-down': ['ontable|0', 'clear|0', 'handempty']}

        :param latt:
        :return:
        """

        action_pred_list = {}
        for action in self.abstract_model.actions.keys():
            for predicate in self.pred_type_mapping.keys():
                if action in action_pred_list.keys():
                    old_preds = action_pred_list[action]
                    if len(old_preds) > 0:
                        temp_preds = latt.generate_preds_for_action(predicate, action, self.pred_type_mapping,
                                                                    self.action_parameters)
                        if temp_preds is None:
                            continue
                        action_pred_list[action].extend(temp_preds)
                else:
                    temp_preds = latt.generate_preds_for_action(predicate, action, self.pred_type_mapping,
                                                                self.action_parameters)
                    if temp_preds is not None:
                        action_pred_list[action] = temp_preds
        return action_pred_list

    def generate_query(self, po_query_module, model_1, model_2, init_state, next_pal_tuple):
        """
        :param po_query_module:
        :param model_1:
        :param model_2:
        :param init_state:
        :param next_pal_tuple:
        :return:
        """

        po_query = po_query_module.Query(model_1, model_2, init_state, next_pal_tuple, self.pal_tuple_dict)

        plan_raw = po_query.get_plan_from_query(init_state, self.domain_name,
                                                self.objects, self.pred_type_mapping,
                                                self.action_parameters)

        return plan_raw

    def discard_models(self, m1, m2, init_state, plan_raw, next_pal_tuple, valid_models,
                       action, action_pred, action_pred_rejection_combination, state, predicate, modes,
                       po_query_module, discarded_count, ref):
        po_query = po_query_module.Query(m1, m2, init_state, next_pal_tuple, self.pal_tuple_dict)
        plan_m1 = ExecutePlan(m1, po_query.init_state, plan_raw)
        is_executable_m1, state_m1, failure_index_1 = plan_m1.execute_plan(self.pal_tuple_dict)
        plan_m2 = ExecutePlan(m2, po_query.init_state, plan_raw)
        is_executable_m2, state_m2, failure_index_2 = plan_m2.execute_plan(self.pal_tuple_dict)
        is_any_model_discarded = False
        if (not is_executable_m1 or not is_executable_m2) and next_pal_tuple[2] == Location.PRECOND:
            if not is_executable_m1:
                discarded_count += 1
                self.discard_model(m1, valid_models)

                is_any_model_discarded = True
                rejected_literal = m1.actions[action][action_pred][int(ref) - 1]

                modes.remove(rejected_literal)
                self.reject_action_pred_combo(action, action_pred, rejected_literal, int(ref) - 1,
                                              action_pred_rejection_combination)
            # {'pickupb1': {'on-floor-bottle1': [<Literal.ABS: 0>, <Literal.NEG: -1>]}}
            # {'pickupb1': {'on-floor-bottle1': [<Literal.NEG: -1>, <Literal.NEG: -1>]}}
            # M2 can't run the plan so returns None, but agent can run
            # If concretized version (agent) can run, then abstracted version should definitely run,
            # reverse need not be true
            if not is_executable_m2:
                discarded_count += 1
                self.discard_model(m2, valid_models)
                is_any_model_discarded = True
                rejected_literal = m2.actions[action][action_pred][int(ref) - 1]
                modes.remove(rejected_literal)
                self.reject_action_pred_combo(action, action_pred, rejected_literal,
                                              int(ref) - 1,
                                              action_pred_rejection_combination)

        if not m1.discarded and not m2.discarded and \
                next_pal_tuple[2] == Location.EFFECTS:

            original_predicate = predicate.split("|")[0]

            final_action = plan_raw[-1].split("|")[1:]
            for z in predicate.split("|")[1:]:
                original_predicate += "|" + final_action[int(z)]

            """
            if pi(M_A) -> p1 \\in S_F(M_A)
             \foll M_i if pi(M_i) -> ~p1 \\in S_F(M_i) REJECT
                        else POSSIBLE, CAN'T REJECT

            Why NOT ACCEPTING, a1, a2 in pi
            if in M_A a1 making p1 true, but in M_i a2 making p1 true
            then M_i not equal to M_A but we can't know.
            We can ACCEPT only if PLAN LEN = 1
            """

            if (original_predicate in state and original_predicate not in state_m2) or \
                    (original_predicate not in state and original_predicate in state_m2):
                discarded_count += 1
                self.discard_model(m2, valid_models)
                is_any_model_discarded = True
                rejected_literal = m2.actions[action][action_pred][int(ref) - 1]
                self.reject_action_pred_combo(action, action_pred, rejected_literal,
                                              int(ref) - 1,
                                              action_pred_rejection_combination)

            if (original_predicate in state and original_predicate not in state_m1) or \
                    (original_predicate not in state and original_predicate in state_m1):
                discarded_count += 1
                self.discard_model(m1, valid_models)
                is_any_model_discarded = True
                rejected_literal = m1.actions[action][action_pred][int(ref) - 1]
                self.reject_action_pred_combo(action, action_pred, rejected_literal,
                                              int(ref) - 1,
                                              action_pred_rejection_combination)

        return discarded_count, is_any_model_discarded, valid_models

    def is_action_pred_compatible(self, action, pred):
        possible = True
        for p in self.pred_type_mapping[pred]:
            if p not in self.action_parameters[action]:
                possible = False

        return possible

    def get_additional_plans(self, action, model1, model2, next_pal_tuple):
        action_name = action.split("|")[0]
        param = FF_PATH + "ff"
        param += " -o " + Q_DOMAIN_FILE
        param += " -f " + Q_PROBLEM_FILE
        param += " -i 120 | grep -i \"Action " + action_name + "\" |"
        param += " sed 's/Action //'"
        param += " > " + ALL_ACTION_FILE
        p = subprocess.Popen([param], shell=True)
        p.wait()
        with open(ALL_ACTION_FILE) as f:
            possible_actions = f.read().splitlines()
        if not is_simulator_agent:
            f = open(Q_DOMAIN_FILE, "w")
            FileUtils.write_domain_to_file(f, self.domain_name, self.objects, self.pred_type_mapping,
                                           self.action_parameters, model1, model2, self.pal_tuple_dict, next_pal_tuple)
            f.close()

        valid_plans = []
        t_act = " ".join(action.lower().split("|"))
        for _act in possible_actions:
            if _act.lower() == t_act:
                continue
            f = open(temp_plan_file, "w")
            f.write("0.00000: (" + _act + ")\n")
            f.close()

            param = VAL_PATH + " -v"
            param += " " + Q_DOMAIN_FILE
            param += " " + Q_PROBLEM_FILE
            param += " " + temp_plan_file
            param += " | grep -i \"successfully\""
            param += " > " + Q_RESULT_FILE
            p = subprocess.Popen([param], shell=True)
            p.wait()
            if "successfully" in open(Q_RESULT_FILE).read():
                _temp_val = _act.lower().split(" ")
                valid_plans.append(["|".join(_temp_val)])

        return valid_plans

    def update_pal_ordering(self, init_state, failure_index, plan_raw, valid_models,
                            next_pal_tuple, model1, model2, po_query_module, lattice_node):
        full_states_used = False
        refined_for_agent_failure = False
        if failure_index > 0:
            temp_plan_raw = copy.deepcopy(plan_raw[failure_index:])
            plan = ExecutePlan(model1, init_state, plan_raw[:failure_index])
            is_executable_agent, state_abstracted, fail_index = plan.execute_plan(self.pal_tuple_dict)
            assert (is_executable_agent and failure_index == fail_index)
        # Assertion failure? Then how was query generated
        else:
            temp_plan_raw = copy.deepcopy(plan_raw)

        found_state = False
        possible_state = None
        state_full = self.get_full_state(temp_plan_raw[0], lattice_node)
        if not is_simulator_agent:
            # State with all the predicates
            is_executable_agent, failure_index, state = self.ask_query(set_to_state(state_full), temp_plan_raw[0:1])

            if is_executable_agent:
                found_state = True
                possible_state = copy.deepcopy(set_to_state(state_full))
                full_states_used = True
            else:
                print("FAILURE")
                start = 0
                with open(self.random_state_file, 'rb') as f:
                    random_states = pickle.load(f)
                i = 0
                _curr_state_set = copy.deepcopy(state_full)
                for _state in state_full:

                    _curr_state_set.remove(_state)
                    _curr_state = set_to_state(_curr_state_set)
                    is_executable_agent, failure_index, state = self.ask_query(_curr_state, temp_plan_raw[0:1])

                    if is_executable_agent:
                        possible_state = _curr_state
                        found_state = True
                        break
                    else:
                        _curr_state_set.append(_state)
                if not found_state:
                    # Getting state from random states start
                    for _state in random_states:
                        i += 1
                        self.objects = _state.objects
                        _curr_state = _state.state
                        _temp_plan_raw = self.generate_query(po_query_module, model1, model2, _curr_state,
                                                             next_pal_tuple)
                        if len(_temp_plan_raw) != 0:
                            # Last action of generated plan should be distinguishing
                            curr_action = temp_plan_raw[0].split("|")[0]
                            _i = 0
                            for _i in range(len(_temp_plan_raw)):
                                if _temp_plan_raw[_i].split("|")[0] == curr_action:
                                    break
                            if _i == len(_temp_plan_raw):
                                continue
                            if _i > 0:
                                is_executable_agent, failure_index, state = self.ask_query(_state.state,
                                                                                           _temp_plan_raw[0:_i])
                                if is_executable_agent:
                                    _curr_state = State(set_to_state(_curr_state), _state.objects)
                                    start = _i
                                else:
                                    continue
                        else:
                            continue
                        is_executable_agent, failure_index, state = self.ask_query(_curr_state,
                                                                                   _temp_plan_raw[start:start + 1])

                        if is_executable_agent:
                            found_state = True
                            possible_state = _curr_state
                            temp_plan_raw = copy.deepcopy(_temp_plan_raw[start:start + 1])
                            break
                        else:
                            print("Failed: ", i)
        else:
            valid_plans = [[temp_plan_raw[-1]]]
            valid_plans.extend(
                self.get_additional_plans(temp_plan_raw[-1], model1, model2, next_pal_tuple))
            init_state_tried = 0
            temp_new_init_states = copy.deepcopy(self.agent.agent_model.random_states)
            while not found_state and init_state_tried < 1000:
                for plan in valid_plans:
                    print("Plan: ", plan)
                    for _state in temp_new_init_states:
                        if VERBOSE:
                            print(_state)
                        self.objects = _state.objects
                        init_state_tried += 1

                        is_executable_agent, failure_index, state = self.ask_query(_state.state, plan)

                        if is_executable_agent:
                            possible_state = _state.state
                            temp_plan_raw = copy.deepcopy(plan)
                            found_state = True
                            break
                    if found_state:
                        break
                    print("Exhausted states")

                if not found_state:
                    temp_new_init_states = copy.deepcopy(
                        self.agent.agent_model.get_more_random_states(10,
                                                                      save=False,
                                                                      random_walk=True))

                if temp_new_init_states is None or len(temp_new_init_states) == 0:
                    break
                print(len(temp_new_init_states), " new states generated")

        if not found_state:
            print("Here after failing")
            self.failed_plans.append(temp_plan_raw)
            return False, valid_models

        if found_state:
            # Now check if any predicate found in possible state is redundant?
            # Remove one of them at a time and see if plan still executes successfully
            abs_preds_precond = []
            current_state = None
            neg_case = None
            for i in range(0, len(temp_plan_raw) + 1):
                if i > 0:
                    # This is a golden chance to get the possible effects
                    # We have already found the minimal state that needs to be true
                    # to execute the action temp_plan_raw[i-1], now whatever effect we get is minimal effect

                    is_executable_agent, failure_index, possible_state = self.ask_query(current_state,
                                                                                        temp_plan_raw[i - 1:i], True)

                    try:
                        assert (is_executable_agent and failure_index == 1)
                    except AssertionError as e:
                        # We cannot make calls on effects as agent didn't run the plan.
                        print("Assertion Error: ", e)
                        break

                    poss_state_set = possible_state
                    curr_state_set = state_to_set(current_state)
                    add_effects = poss_state_set - curr_state_set
                    del_effects = curr_state_set - poss_state_set
                    abs_effects = curr_state_set & poss_state_set
                    doubtful_preds = set(state_full) - curr_state_set
                    neg_precond = set()
                    temp_act = temp_plan_raw[i - 1].split("|")[0]

                    instantiated_pred_doubtful = set()
                    for d_pred in doubtful_preds:
                        if not (
                                self.is_action_pred_compatible(temp_plan_raw[i - 1].split("|")[0],
                                                               d_pred.split("|")[0])):
                            continue
                        action_name, instantiated_pred = map_pred_action_param(d_pred, temp_plan_raw[i - 1])
                        if instantiated_pred is not None and action_name is not None:
                            instantiated_pred_doubtful.add(instantiated_pred)

                    if not full_states_used:
                        instantiated_pred_doubtful = instantiated_pred_doubtful | abs_preds_precond

                    for d_pred in instantiated_pred_doubtful:
                        if self.pal_tuple_dict[(temp_act, d_pred, Location.PRECOND)]:
                            continue
                        d_pred = instantiate_pred_with_action(d_pred, temp_plan_raw[i - 1])
                        if d_pred is None:
                            continue
                        _new_state_set = {d_pred} | curr_state_set
                        _new_state = set_to_state(_new_state_set)

                        is_executable_agent, failure_index, possible_state = self.ask_query(_new_state,
                                                                                            temp_plan_raw[i - 1:i])
                        if not is_executable_agent:
                            action_name, instantiated_pred = map_pred_action_param(d_pred, temp_plan_raw[i - 1])
                            if failure_index != -1:
                                if instantiated_pred in abs_preds_precond:
                                    abs_preds_precond.remove(instantiated_pred)
                                neg_precond.add(instantiated_pred)

                        elif d_pred not in possible_state:
                            if d_pred in abs_effects:
                                abs_effects.remove(d_pred)
                            del_effects.add(d_pred)

                    for preds in abs_preds_precond:
                        if not self.pal_tuple_dict[(temp_act, preds, Location.PRECOND)]:
                            valid_models = self.propagate_refinement_in_models(valid_models,
                                                                               [temp_act, preds, Literal.ABS],
                                                                               next_pal_tuple, Location.PRECOND)
                            self.fix_pal_tuple((temp_act, preds, Location.PRECOND), valid_models)

                    for preds in neg_precond:
                        if not self.pal_tuple_dict[(temp_act, preds, Location.PRECOND)]:
                            valid_models = self.propagate_refinement_in_models(valid_models,
                                                                               [temp_act, preds, Literal.NEG],
                                                                               next_pal_tuple, Location.PRECOND)
                            self.fix_pal_tuple((temp_act, preds, Location.PRECOND), valid_models)

                    for e in add_effects:
                        action_name, instantiated_pred = map_pred_action_param(e, temp_plan_raw[i - 1])
                        if action_name is None or \
                            (action_name, instantiated_pred, Location.EFFECTS) not in self.pal_tuple_dict.keys() or\
                                self.pal_tuple_dict[(action_name, instantiated_pred, Location.EFFECTS)]:
                            continue
                        valid_models = self.propagate_refinement_in_models(valid_models,
                                                                           [action_name, instantiated_pred,
                                                                            Literal.POS], next_pal_tuple,
                                                                           Location.EFFECTS)
                        self.fix_pal_tuple((action_name, instantiated_pred, Location.EFFECTS), valid_models)

                    for e in del_effects:
                        action_name, instantiated_pred = map_pred_action_param(e, temp_plan_raw[i - 1])
                        if action_name is None or \
                            (action_name, instantiated_pred, Location.EFFECTS) not in self.pal_tuple_dict.keys() or\
                                self.pal_tuple_dict[(action_name, instantiated_pred, Location.EFFECTS)]:
                            continue
                        valid_models = self.propagate_refinement_in_models(valid_models,
                                                                           [action_name, instantiated_pred,
                                                                            Literal.NEG], next_pal_tuple,
                                                                           Location.EFFECTS)
                        self.fix_pal_tuple((action_name, instantiated_pred, Location.EFFECTS), valid_models)

                    # Since we are not using all possible predicates in init state, any inferences about effect being
                    # absent is incorrect.
                    for e in abs_effects:
                        action_name, instantiated_pred = map_pred_action_param(e, temp_plan_raw[i - 1])
                        if action_name is None or \
                            (action_name, instantiated_pred, Location.EFFECTS) not in self.pal_tuple_dict.keys() or\
                                self.pal_tuple_dict[(action_name, instantiated_pred, Location.EFFECTS)]:
                            continue
                        valid_models = self.propagate_refinement_in_models(valid_models,
                                                                           [action_name, instantiated_pred,
                                                                            Literal.ABS], next_pal_tuple,
                                                                           Location.EFFECTS)
                        self.fix_pal_tuple((action_name, instantiated_pred, Location.EFFECTS), valid_models)

                    # ASSUMING AGENT CAN EXECUTE TEMP_PLAN_RAW[i]
                    temp_act = temp_plan_raw[i - 1].split("|")[0]
                    all_preds_abs = [tup[1] for tup in self.pal_tuple_dict.keys() if
                                     tup[0] == temp_act and
                                     tup[2] == Location.EFFECTS and
                                     not self.pal_tuple_dict[tup]]
                    for preds in all_preds_abs:
                        if not self.pal_tuple_dict[(temp_act, preds, Location.EFFECTS)]:
                            valid_models = self.propagate_refinement_in_models(valid_models,
                                                                               [temp_act, preds, Literal.ABS],
                                                                               next_pal_tuple, Location.EFFECTS)
                            self.fix_pal_tuple((temp_act, preds, Location.EFFECTS), valid_models)

                    if i == len(temp_plan_raw):
                        continue

                    possible_state = set_to_state(current_state)

                current_state = copy.deepcopy(possible_state)
                _full_state = copy.deepcopy(possible_state)
                new_preds_temp = []
                for key, val in _full_state.items():
                    if not (self.is_action_pred_compatible(temp_plan_raw[i].split("|")[0], key)):
                        continue
                    for v in val:
                        temp_init = copy.deepcopy(current_state)
                        if len(val) == 1:
                            del temp_init[key]
                            assert (key not in list(temp_init.keys()))
                        else:
                            temp_init[key].remove(v)

                        is_executable_agent, failure_index, possible_state = self.ask_query(temp_init,
                                                                                            temp_plan_raw[i:], True)
                        # For plans of length more than 1, it is possible that agent failed for 2nd or later action
                        # in the plan, so even if is_executable_agent false, check if failure_index = 1 or later
                        if is_executable_agent or failure_index >= 1:
                            initial_val_len = len(current_state[key])
                            current_state[key].remove(v)
                            if initial_val_len > 0 and len(current_state[key]) == 0:
                                del current_state[key]
                        else:
                            if isinstance(v, (list, tuple)) and len(v) > 1:
                                final_val = key
                                for ind in range(0, len(v)):
                                    final_val += "|" + v[ind]
                                predicate_temp = [final_val]
                            else:
                                predicate_temp = list(state_to_set({key: tuple(v, )}))

                            action = temp_plan_raw[i]
                            action_name, instantiated_pred = map_pred_action_param(predicate_temp[0], action)
                            if action_name is None:
                                continue
                            new_preds_temp.append(instantiated_pred)
                            if self.pal_tuple_dict[(action_name, instantiated_pred, Location.PRECOND)]:
                                continue
                            valid_models = self.propagate_refinement_in_models(valid_models,
                                                                               [action_name, instantiated_pred,
                                                                                Literal.POS], next_pal_tuple,
                                                                               Location.PRECOND)
                            self.fix_pal_tuple((action_name, instantiated_pred, Location.PRECOND), valid_models)
                # ASSUMING AGENT CAN EXECUTE TEMP_PLAN_RAW[i]
                temp_act = temp_plan_raw[i].split("|")[0]
                all_preds_poss = [tup[1] for tup in self.pal_tuple_dict.keys() if
                                  tup[0] == temp_act and tup[2] == Location.PRECOND]
                abs_preds_precond = set(all_preds_poss) - set(new_preds_temp)

                neg_case = False

            if not neg_case:
                refined_for_agent_failure = True

        return refined_for_agent_failure, valid_models

    @staticmethod
    def discard_model(m, valid_models):
        m.discarded = True
        for tm in valid_models:
            if tm == m:
                tm.discarded = True

    def initialize_pal_tuple_dict(self, lattice_node):
        # To keep track which of the action_predicate_refinement is done
        # Dict{} with keys (action, predicate, refinement)
        # Values are Boolean
        # pal_tuple_dict = {}

        replaced_actions = []
        if self.agent.agent_type == "simulator":
            for action in self.abstract_model.actions.keys():
                replaced_action = copy.deepcopy(action)
                replaced_action = replaced_action.replace('-', '')
                replaced_actions.append(replaced_action)

        for action in self.abstract_model.actions.keys():
            for predicate in self.pred_type_mapping.keys():
                # Generate the predicates with action parameter index inbuilt into them
                # ontable takes one argument, if action a has 2 possible locations of that argument type, say 0 and 2
                # so it'll get converted to ontable|0 and ontable|2 when called with action a
                temp_preds = lattice_node.generate_preds_for_action(predicate, action, self.pred_type_mapping,
                                                                    self.action_parameters)
                if temp_preds is not None:
                    for i in range(2):
                        for p in temp_preds:
                            # Use tuples as key
                            key = (action, p, Location(i + 1))
                            p_name = p.lower().split("|")[0]
                            if self.agent.agent_type == "simulator" and p_name in replaced_actions:
                                self.pal_tuple_dict[key] = True
                            # We can do this at the end too if p_name == action.replace('-', '') and p_params ==
                            # sorted(p_params) and p_params[0] == "0": self.abstract_model.actions[action][p] =
                            # [Literal.POS, Literal.ABS] else: self.abstract_model.actions[action][p] = [Literal.ABS,
                            # Literal.ABS]
                            else:
                                self.pal_tuple_dict[key] = False

        if is_simulator_agent:
            _, _, pal_tuples_finalized = self.agent.agent_model.bootstrap_model()
            for tup in pal_tuples_finalized:
                assert (tup in self.pal_tuple_dict.keys())
                self.pal_tuple_dict[tup] = True

        return

    def get_full_state(self, action, lattice_node):
        action_name = action.split("|")[0]
        action_params = action.split("|")[1:]
        full_state = []

        for predicate in self.pred_type_mapping.keys():
            # Generate the predicates with action parameter index inbuilt into them
            # ontable takes one argument, if action a has 2 possible locations of that argument type, say 0 and 2
            # so it'll get converted to ontable|0 and ontable|2 when called with action a
            temp_preds = lattice_node.generate_preds_for_action(predicate, action_name, self.pred_type_mapping,
                                                                self.action_parameters)
            if temp_preds is not None:
                for temp_pred in temp_preds:
                    pred = temp_pred.split("|")[0]
                    pred_params = temp_pred.split("|")[1:]
                    for p in pred_params:
                        pred += "|" + action_params[int(p)]

                    full_state.append(pred)

        return full_state

    @staticmethod
    def get_modified_init_states(next_pal_tuple, m1, m2, possible_init_states):
        action_pred = next_pal_tuple[1]
        action = next_pal_tuple[0]
        # loc = next_pal_tuple[2]
        # pred_name = action_pred.split("|")[0]
        modified_init_states = []

        for p, val in m1.actions[action].items():
            pos = False
            neg = False

            pred = p.split("|")[0] + "|"
            count = 0
            for preds in m1.actions[action].keys():
                if pred in preds:
                    count += 1

            if count > 1:
                continue

            if p != action_pred and val == m2.actions[action][p]:
                if val[0] == Literal.POS:
                    pos = True
                else:
                    neg = True

            elif p == action_pred and val != m2.actions[action][p]:
                if {Literal.POS, Literal.NEG} == {val[0], m2.actions[action][p][0]}:
                    continue
                elif {Literal.POS, Literal.ABS} == {val[0], m2.actions[action][p][0]}:
                    neg = True
                elif {Literal.ABS, Literal.NEG} == {val[0], m2.actions[action][p][0]}:
                    pos = True

            if len(modified_init_states) == 0:
                temp_init_states = copy.deepcopy(possible_init_states)
            else:
                temp_init_states = copy.deepcopy(modified_init_states)

            modified_init_states = []
            if pos:
                for s in temp_init_states:
                    if p.split("|")[0] in s.state.keys():
                        modified_init_states.append(s)
            elif neg:
                for s in temp_init_states:
                    if p.split("|")[0] not in s.state.keys():
                        modified_init_states.append(s)

        if len(modified_init_states) == 0:
            modified_init_states = copy.deepcopy(possible_init_states)
        return copy.deepcopy(modified_init_states)

    def agent_interrogation_algo(self):
        """
        :return: true for successful execution, false otherwise
        :rtype: bool
        """

        # Import modules for both kind of queries
        # pr_query_module = importlib.import_module("query.pr_query")
        po_query_module = importlib.import_module("query.po_query")
        init_state = {}
        # exit(0)

        # Create list of predicates and abstracted predicates
        abs_predicates = list(self.abstract_model.predicates.keys())
        all_predicates = list(self.agent.agent_model.predicates.keys())

        # Create a lattice object
        latt = Lattice()
        lattice_node = LatticeNode(latt, [self.abstract_model], self.abstract_predicates)

        ####################################################################
        # This logic limits number of objects and counts number of objects
        object_count = 0

        max_obj_type_count = {}
        for obj_types, obj_cont in self.objects.items():
            object_count += len(obj_cont)
            max_obj_type_count[obj_types] = 0
        for v in self.action_parameters.values():
            type_count = Counter(v)
            for tc in type_count:
                max_obj_type_count[tc] = max(max_obj_type_count[tc], type_count[tc])
        temp_objects = {}
        for o, item in self.objects.items():
            if len(item) > max_obj_type_count[o] + 1:
                temp_key = copy.deepcopy(item[0:max_obj_type_count[o]])
            else:
                temp_key = copy.deepcopy(item)
            temp_objects[o] = temp_key

        ########################
        # Hypothesis: Number of objects needed = max arity of that object type in any action
        ########################
        self.objects = copy.deepcopy(temp_objects)
        #############################################

        # To keep track which of the action_predicate_refinement is done
        # Dict{} with keys (action, predicate, refinement)
        # Values are Boolean
        self.initialize_pal_tuple_dict(lattice_node)

        # To calculate actual number of predicates
        pred_set = set()
        for p in list(self.pal_tuple_dict.keys()):
            pred_set.add(p[1])
        modified_preds_count = len(pred_set)

        self.action_count = len(self.abstract_model.actions.keys())
        self.predicate_count = len(self.pred_type_mapping.keys())
        self.mod_pred_count = copy.deepcopy(modified_preds_count)

        #################################

        # int_parent_models holds the possible models at any given point of time.
        int_parent_models = [self.abstract_model]

        # action_preds_list is used to store the modified predicates ("ontable|0", etc) corresponding to each action
        # For eg:  actions_pred_list = {'pick-up': ['ontable|0', 'clear|0', 'handempty'], \
        # 'put-down': ['ontable|0', 'clear|0', 'handempty']}
        action_preds_list = self.populate_action_pred_list(lattice_node)

        lattice_node.action_pred_dict = copy.deepcopy(action_preds_list)
        temp_init_state = copy.deepcopy(init_state)
        original_abs_preds = copy.deepcopy(abs_predicates)
        original_action_pred_list = copy.deepcopy(action_preds_list)

        preprocess_start_time = time.time()

        orig_possible_state_object_comb = None
        if self.agent.agent_type == "simulator":
            orig_possible_state_object_comb = copy.deepcopy(self.agent.agent_model.random_states)
        elif self.agent.agent_type == "symbolic":
            orig_possible_state_object_comb = self.get_possible_init_states(temp_init_state)
        preprocess_time = time.time() - preprocess_start_time
        agent_exec = 0

        print("Actual Predicates = " + str(len(self.pred_type_mapping.keys())) + str("\n"))
        print("Modified Predicates = " + str(modified_preds_count) + str("\n"))
        print("Action Count = " + str(len(self.abstract_model.actions.keys())) + str("\n"))
        print("Object Count = " + str(object_count) + str("\n"))

        tried_cont = 0
        query_1_failed = 0
        discarded_count = 0
        new_discard_count = 0
        valid_models = None
        while True:
            model_level = 0

            next_pal_tuple = self.get_next_pal_tuple()
            refined_for_agent_failure = False
            if next_pal_tuple is None:
                # All possible refinements over
                # int_parent_models should be holding possible models at most concretized level
                # return False, False, None, None, None, None, None, None
                break

            # Try to generate predicates that are part of init_state
            init_state = copy.deepcopy(temp_init_state)

            if len(abs_predicates) < len(init_state):
                predicate = get_next_predicate(init_state, abs_predicates)
            else:
                predicate = get_next_predicate(all_predicates, abs_predicates)

            # All predicates done!!
            if predicate is None:
                next_r = self.get_next_pal_tuple()
                if next_r is not None:
                    abs_predicates = copy.deepcopy(original_abs_preds)
                    action_preds_list = copy.deepcopy(original_action_pred_list)
                else:
                    exit(0)
                    return False, None
                continue

            # Pick something in init state.
            # Hack to get plans that distinguish at more abstract levels
            original_pred = predicate
            pred_valid = False
            temp_action_preds_list = copy.deepcopy(action_preds_list)

            for action in action_preds_list:
                for pred in action_preds_list[action]:
                    if predicate in pred or predicate == pred:
                        pred_valid = True
                        predicate = pred
                        action_preds_list[action].remove(pred)
            if not pred_valid:
                abs_predicates.append(original_pred)
                continue

            next_pal_tuple = self.get_next_pal_tuple()
            predicate = next_pal_tuple[1]
            print("ACTUAL NEXT PAL TUPLE: ", next_pal_tuple)
            tmp_int_parent_models = []
            action_pred_rejection_combination = {}
            modes = [Literal.POS, Literal.NEG, Literal.ABS]

            for temp_abs_model in int_parent_models:
                action_pred = next_pal_tuple[1]
                action = next_pal_tuple[0]
                ref = next_pal_tuple[2]

                # partitions stores the partitions for a refinement next_pal_tuple when called on
                # a model temp_abs_model
                intermediate_models = lattice_node.get_model_partitions(temp_abs_model, action_pred,
                                                                        ref, action, tuple(modes))

                # Run query and discard models here
                # Remove all invalid models and store only the valid ones
                valid_models = [i for i in intermediate_models if not i.discarded]

                # Generate all possible combinations of models
                for m1, m2 in combinations(valid_models, 2):
                    if m1.discarded or m2.discarded:
                        continue

                    if m1.actions == m2.actions:
                        new_discard_count += 1
                        self.discard_model(m1, valid_models)
                        continue

                    if len(action_pred_rejection_combination) > 0:
                        if self.is_model_rejectable(m1, action_pred_rejection_combination):
                            new_discard_count += 1
                            self.discard_model(m1, valid_models)
                        if self.is_model_rejectable(m2, action_pred_rejection_combination):
                            new_discard_count += 1
                            self.discard_model(m2, valid_models)

                    tried_cont += 1
                    init_state_tried = 0
                    possible_state_objects = copy.deepcopy(orig_possible_state_object_comb)

                    is_any_model_discarded = False
                    if ref == Location.PRECOND:
                        modified_state_objects = self.get_modified_init_states(next_pal_tuple, m1, m2,
                                                                               possible_state_objects)

                    else:
                        modified_state_objects = copy.deepcopy(possible_state_objects)

                    for state_objs in modified_state_objects:
                        init_state = state_objs.state
                        self.objects = state_objs.objects

                        init_state_tried += 1
                        # preds_in_m1_m2 =  list(set(list(m1.predicates.keys())) & set(list(m2.predicates.keys())))
                        # preds_in_m1_m2 = [i for j in preds_in_m1_m2 for i in j.split("|")]
                        # if list(set(preds_in_m1_m2) & set(list(init_state.keys()))) == []:
                        # 	continue

                        plan_raw = self.generate_query(po_query_module,
                                                       m1, m2, init_state, next_pal_tuple)
                        if len(plan_raw) != 0:
                            is_executable_agent, failure_index, state = self.ask_query(init_state, plan_raw)
                            agent_exec += 1

                            if failure_index != len(plan_raw) or not is_executable_agent:
                                refined_for_agent_failure, valid_models = self.update_pal_ordering(init_state,
                                                                                                   failure_index,
                                                                                                   plan_raw,
                                                                                                   valid_models,
                                                                                                   next_pal_tuple, m1,
                                                                                                   m2, po_query_module,
                                                                                                   lattice_node)
                                break

                            elif failure_index == len(plan_raw) and is_executable_agent:
                                discarded_count, is_any_model_discarded, \
                                valid_models = self.discard_models(m1, m2, init_state, plan_raw,
                                                                   next_pal_tuple, valid_models, action, action_pred,
                                                                   action_pred_rejection_combination, state, predicate,
                                                                   modes, po_query_module, discarded_count, ref)
                            else:
                                exit()
                        if is_any_model_discarded or refined_for_agent_failure:
                            break

                    if refined_for_agent_failure:
                        break

                t_valid_models = [i for i in valid_models if not i.discarded]
                if refined_for_agent_failure and len(t_valid_models) == 3:
                    break
                tmp_int_parent_models = [i for i in valid_models if not i.discarded]

            if refined_for_agent_failure:
                int_parent_models = copy.deepcopy(valid_models)
                action_preds_list = copy.deepcopy(temp_action_preds_list)
                continue

            valid_models = [i for i in valid_models if not i.discarded]
            if len(valid_models) == 1 and not self.pal_tuple_dict[next_pal_tuple]:
                self.fix_pal_tuple(next_pal_tuple, valid_models)
                tmp_int_parent_models = [i for i in valid_models if not i.discarded]

            int_parent_models = copy.deepcopy(tmp_int_parent_models)
            model_level += 1
            model_count = 1
            total_models = 0
            for m in int_parent_models:
                model_count += 1

                temp_num_models = 0
                for key, val in m.actions.items():
                    for k, v in val.items():
                        if v[0] in [Literal.AP, Literal.AN, Literal.NP]:
                            temp_num_models += 1
                        if v[1] in [Literal.AP, Literal.AN, Literal.NP]:
                            temp_num_models += 1
                total_models += 2 ** temp_num_models

            pp = pprint.PrettyPrinter(indent=2)
            if len(valid_models) <= 3:
                print("Current Model(s): ")
                for v in valid_models:
                    pp.pprint(v.actions)

            if self.get_next_pal_tuple(predicate=predicate) is None:
                abs_predicates.append(predicate)

        num_models = []
        model_count_final = 0

        for m in int_parent_models:
            model_count_final += 1

            temp_num_models = 0
            for key, val in m.actions.items():
                for k, v in val.items():
                    if v[0] in [Literal.AP, Literal.AN, Literal.NP]:
                        temp_num_models += 1
                    if v[1] in [Literal.AP, Literal.AN, Literal.NP]:
                        temp_num_models += 1
            num_models.append(temp_num_models)

        total_models = 0
        for num in num_models:
            total_models += 2 ** num
        len(self.abstract_model.actions.keys())
        pp = pprint.PrettyPrinter(indent=2)
        print("Predicted Model: ")
        for v in valid_models:
            pp.pprint(v.actions)

        print("Total Possible Models = " + str(total_models) + str("\n"))
        print("Number of times Agent Executed = " + str(agent_exec) + str("\n"))
        print("Preprocessing Time = " + str(preprocess_time) + str("\n"))
        print("Actual Predicates = " + str(len(self.pred_type_mapping.keys())) + str("\n"))
        print("Modified Predicates = " + str(modified_preds_count) + str("\n"))
        print("Action Count = " + str(len(self.abstract_model.actions.keys())) + str("\n"))
        print("Object Count = " + str(object_count) + str("\n"))

        print("Possible Model count = " + str(len(valid_models)))
        print("Combinations tried Count = " + str(tried_cont))
        print("Agent Execution Failed Count = " + str(self.agent_cant_execute))
        print("Total Agent Queries = ", self.agent_query_total)
        print("Total Unique Queries = ", self.query_new)
        print("Repeated Queries = ", self.query_old)
        print("Invalid Init State = ", self.invalid_init_state)
        print("No plan found count (q1) = " + str(query_1_failed))
        print("Discarded model count = " + str(discarded_count))
        print("New Discard count = " + str(new_discard_count))
        print("\n")

        return self.query_new, (time.time() - self.start_time), self.data_dict, self.pal_tuples_fixed
