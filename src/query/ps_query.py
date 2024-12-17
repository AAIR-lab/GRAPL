#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import config
import copy
import os
import subprocess
import string
import sys
from itertools import combinations, chain
from pddlgym.structs import LiteralDisjunction, ConditionalEffect, LiteralConjunction, Predicate, TypedEntity, ProbabilisticEffect
from pddlgym.structs import Literal as Ltr

from config import *
from utils import FileUtils
from src.model import Model as Md

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '../..'))
import query.genericQuery as gc

from planner.prp import PRP
from planner.prp import PRPPolicyNotFoundException
from pddlgym.structs import State

class Query(gc.GenericQuery):
    """
    This class is for the policy simulation query using PRP.

    Initial State has to be given as input.
    """

    def __init__(self, ground_truth_agent, base_dir, model1, model2, init_state, next_pal_tuple, dummy_problem, phase,
                 preconds_learned, applicable_action_state_cache,
                 store_logs = True, testing_only = False, qno=0):

        self.ground_truth_agent = ground_truth_agent
        self.base_dir = base_dir
        self.query_no = qno
        self.preconds_learned = preconds_learned
        self.problem = copy.deepcopy(dummy_problem)
        self.model1 = copy.deepcopy(model1)
        self.model2 = copy.deepcopy(model2)
        self.init_state = init_state
        self.phase = phase
        self.tried = 0
        self.modified_init_state = False
        self.first_call = True
        self.next_pal_tuple = next_pal_tuple
        self.agent_problem = copy.deepcopy(dummy_problem)
        self.bfs_steps = 0
        self.applicable_action_state_cache = applicable_action_state_cache
        
        # TODO: Need to add support for when it is False.
        # Ideally, create a temp directory and clean it up.
        assert store_logs == True
        self.store_logs = store_logs
        self.combined_model = None
        self.testing_only = testing_only
        if not testing_only:
            self.create_query_domain()
            self.create_query_problem()
        
    def add_suffix_to_predicate(self, predicate, suffix):
        new_pred_name = str(predicate[1].name)+suffix
        new_predicate = copy.deepcopy(predicate[1])
        new_predicate.name = new_pred_name
        return tuple([new_pred_name, new_predicate])

    def rename_literal_with_suffix(self, literal, suffix):

        literal.predicate.name += suffix
        literal._update_variable_caches()

        if suffix+suffix in literal.predicate.name:
            literal.predicate.name = literal.predicate.name[0:len(literal.predicate.name)-len(suffix)]
            literal._update_variable_caches()

    # def generate_equality_predicates(self, action):
    #     object_list = list(map(lambda x: x, action.params))
    #     combs = list(combinations(object_list, 2))
    #
    #     for _c in combs:
    #         var_types =
    #         equal_pred = Predicate("=", 2, )
    #         p_psi = Predicate("p_psi", 0, [], False, False, False)
    #         psi_literal = Ltr(p_psi, [])
    #         # initial_state.add(psi_literal)
    #         # goal_state.append(LiteralConjunction([psi_literal]))
    #         self.problem.initial_state = frozenset(initial_state)
    #         if self.phase == Location.PRECOND:
    #             self.problem.goal = LiteralConjunction([psi_literal])
    #     print(combs)

    def create_query_problem(self, objs=None):
        self.problem.predicates = copy.deepcopy(self.combined_model.predicates)

        goal_state = []
        initial_state = set()
        for _i in self.problem.initial_state.literals:

            m1_p = copy.deepcopy(_i)
            m2_p = copy.deepcopy(_i)
            self.rename_literal_with_suffix(m1_p,"_1")
            self.rename_literal_with_suffix(m2_p, "_2")
            if self.phase != Location.PRECOND:
                goal_state.append(LiteralConjunction([m1_p, m2_p.inverted_anti]))
                goal_state.append(LiteralConjunction([m2_p, m1_p.inverted_anti]))
            initial_state.add(m1_p)
            initial_state.add(m2_p)
        p_psi = Predicate("p_psi", 0, [], False, False, False)
        psi_literal = Ltr(p_psi,[])
        # initial_state.add(psi_literal)
        # goal_state.append(LiteralConjunction([psi_literal]))
        self.problem.initial_state = State(frozenset(initial_state),
                                           self.problem.initial_state.objects,
                                           self.problem.initial_state.goal)
        if self.phase == Location.PRECOND or self.phase == Location.EFFECTS:
            self.problem.goal = LiteralConjunction([psi_literal])
        else:
            self.problem.goal = LiteralDisjunction(goal_state)


    def modify_init_state(self, type_comp, policy, init_state, modeln, useful_states):
        p = copy.deepcopy(self.next_pal_tuple[0])
        a = self.next_pal_tuple[1]
        # vars = self.combined_model.actions[a].params
        vars = list(map(lambda x: x.name, self.combined_model.actions[a].params))
        idxs = []
        policy_var = policy[0].split(" ")[1:]
        d= copy.deepcopy(p)
        d2 = copy.deepcopy(p)
        if modeln == 1 or modeln == 2:
            d.predicate.name += "_1"
            d._update_variable_caches()
            d2.predicate.name += "_2"
            d2._update_variable_caches()

        _q = 0
        for i in p.variables:
            idx = vars.index(i.name)
            # d.variables[idx].name = policy_var[idx]
            # new_value =
            if modeln == 1 or modeln ==2:
                # if _q ==0:
                #     d.predicate.name += "_1"
                nam = policy_var[idx]
                d.variables[_q] = TypedEntity(nam, d.variables[_q].var_type)
                d._update_variable_caches()

                # if _q ==0:
                #     d2.predicate.name += "_2"
                nam2 = policy_var[idx]
                d2.variables[_q] = TypedEntity(nam2, d2.variables[_q].var_type)
                d2._update_variable_caches()
            else:
                d.variables[_q] = TypedEntity(policy_var[idx], d.variables[_q].var_type)
                d._update_variable_caches()
            _q+=1

        i_s = list(init_state.literals)
        # if type_comp == Literal.NP and d not in i_s:
        #     i_s.append(d)
        if type_comp == Literal.AP and d in i_s:
            if modeln == 1 or modeln == 2:
                i_s.remove(d2)
            i_s.remove(d)

        if type_comp == Literal.AN and d not in i_s:
            i_s.append(d)
            if modeln == 1 or modeln == 2:
                i_s.append(d2)

        if modeln != 1 and modeln != 2:
            useful_states[p][type_comp] = d

        ist = set(i_s)
        return State(frozenset(ist),
                     init_state.objects,
                     init_state.goal)



        # print(p)

    def create_query_domain(self, modeln = None):
        new_model = copy.deepcopy(self.model1)

        new_model.predicates = {}

        # if not( modeln == 1 or modeln == 2):
        for p in self.problem.predicates.items():
            if modeln == 1 or modeln ==2:
                key, value = self.add_suffix_to_predicate(p, "")
                new_model.predicates[key] = value
            else:
                key, value = self.add_suffix_to_predicate(p, "_1")
                new_model.predicates[key] = value
                key, value = self.add_suffix_to_predicate(p, "_2")
                new_model.predicates[key] = value

        ## Define_dummy_predicate_p_psi
        p_psi = Predicate("p_psi", 0, [], False, False, False)
        # ame, arity, var_types = None, is_negative = False, is_anti = False,
        # negated_as_failure = False)
        new_model.predicates["p_psi"] = p_psi

        # new_model.actions = {}
        for action, operator in self.model1.actions.items():
            if (action not in self.next_pal_tuple) and self.phase == Location.PRECOND:
                new_model.actions.pop(action)
                continue


            # equality_conjucntion = self.generate_equality_predicates(new_model.actions[action])
            new_operator_precond_1 = copy.deepcopy(operator.preconds)
            new_operator_precond_1_negation = copy.deepcopy(operator.preconds)
            for pre in new_operator_precond_1.literals:
                self.rename_literal_with_suffix(pre,"_1")
            neg_literal_list = []
            for pre in new_operator_precond_1_negation.literals:
                self.rename_literal_with_suffix(pre,"_1")
                if str(pre).startswith('Not'):
                    neg_literal_list.append(pre.negative)
                else:
                    neg_literal_list.append(pre.inverted_anti)
            new_operator_precond_1_negation = LiteralDisjunction(neg_literal_list)
            new_operator_precond_2 = copy.deepcopy(self.model2.actions[action].preconds)
            new_operator_precond_2_negation = copy.deepcopy(self.model2.actions[action].preconds)
            for pre in new_operator_precond_2.literals:
                self.rename_literal_with_suffix(pre,"_2")
            neg_literal_list_2 = []
            for pre in new_operator_precond_2_negation.literals:
                self.rename_literal_with_suffix(pre,"_2")
                if str(pre).startswith('Not'):
                    neg_literal_list_2.append(pre.negative)
                else:
                    neg_literal_list_2.append(pre.inverted_anti)
            new_operator_precond_2_negation = LiteralDisjunction(neg_literal_list_2)
            new_operator = LiteralDisjunction([new_operator_precond_1, new_operator_precond_2])
            new_model.actions[action].preconds = new_operator


            if modeln == 1:
                new_model.actions[action].preconds = copy.deepcopy(new_operator_precond_1)
            elif modeln == 2:
                new_model.actions[action].preconds = copy.deepcopy(new_operator_precond_2)
            if modeln == 1 or modeln == 2:
                # new_model.actions[action].precond
                p_psi = Predicate("p_psi", 0, [], False, False, False)
                # model_new.predicates["p_psi"] = p_psi
                new_model.actions[action].effects = LiteralConjunction([p_psi])
                continue

            new_operator_effect_1 = copy.deepcopy(operator.effects)
            if self.phase == Location.EFFECTS and len(new_operator_effect_1.literals) > 0 and \
                    isinstance(new_operator_effect_1.literals[0], ProbabilisticEffect):
                for _idx, _p in enumerate(new_operator_effect_1.literals[0].literals):
                    if isinstance(_p, Ltr) and _p.predicate.name == "NOCHANGE":
                        new_operator_effect_1.literals[0].literals.remove(_p)
                        new_operator_effect_1.literals[0].probabilities.pop(_idx)
                        continue
                    for pre in _p.literals:
                        self.rename_literal_with_suffix(pre, "_1")
                # HACK: Will have only one more literal at max
                if len(new_operator_effect_1.literals) == 2:
                    assert(isinstance(new_operator_effect_1.literals[1],Ltr))
                    self.rename_literal_with_suffix(new_operator_effect_1.literals[1], "_1")
                new_operator_effect_2 = copy.deepcopy(self.model2.actions[action].effects)
                for _idx, _p in enumerate(new_operator_effect_2.literals[0].literals):
                    if isinstance(_p, Ltr) and _p.predicate.name == "NOCHANGE":
                        new_operator_effect_2.literals[0].literals.remove(_p)
                        new_operator_effect_2.literals[0].probabilities.pop(_idx)
                        continue
                    for pre in _p.literals:
                        self.rename_literal_with_suffix(pre, "_2")
                # HACK: Will have only one more literal at max
                if len(new_operator_effect_2.literals) == 2:
                    assert(isinstance(new_operator_effect_2.literals[1],Ltr))
                    self.rename_literal_with_suffix(new_operator_effect_2.literals[1], "_2")
            else:
                for pre in new_operator_effect_1.literals:
                    self.rename_literal_with_suffix(pre,"_1")
                new_operator_effect_2 = copy.deepcopy(self.model2.actions[action].effects)
                for pre in new_operator_effect_2.literals:
                    self.rename_literal_with_suffix(pre,"_2")


            #### when (pre_m1 and pre_m2) (eff_m1 and eff_m2)
            conjunction_precond = copy.deepcopy(new_operator_precond_1)
            conjunction_precond.literals.extend(new_operator_precond_2.literals)

            conjunction_effect = copy.deepcopy(new_operator_effect_1)
            conjunction_effect.literals.extend(new_operator_effect_2.literals)
            # new_operator = LiteralConjunction([new_operator_effect_1, new_operator_effect_2])


            conjunction_conditional = ConditionalEffect(conjunction_precond, conjunction_effect)

            #### when ((pre_m1 and not(pre_m2)) or (pre_m2 and not(pre_m1))) (p_psi)
            disjunction_precond_part1 = copy.deepcopy(new_operator_precond_1)
            if self.phase == Location.EFFECTS:
                disjunction_precond_part1.literals = []

            if len(new_operator_precond_2_negation.literals) > 0:
                list_new = disjunction_precond_part1.literals

                list_new.extend([new_operator_precond_2_negation])
                disjunction_precond_part1 = LiteralConjunction(list_new)
            # disjunction_precond_part1.literals.extend(new_operator_precond_2_negation.literals)

            disjunction_precond_part2 = copy.deepcopy(new_operator_precond_2)
            if self.phase == Location.EFFECTS:
                disjunction_precond_part2.literals = []
            if len(new_operator_precond_1_negation.literals) > 0:
                list_new = disjunction_precond_part2.literals
                list_new.extend([new_operator_precond_1_negation])
                disjunction_precond_part2 = LiteralConjunction(list_new)

            # disjunction_precond_part2 = copy.deepcopy(new_operator_precond_2)
            # disjunction_precond_part2.literals.extend(new_operator_precond_1_negation.literals)

            # disjunction_precond = LiteralDisjunction([disjunction_precond_part1, disjunction_precond_part2])
            disjunction_effect = LiteralConjunction([p_psi])

            disjunction_conditional_1 = ConditionalEffect(disjunction_precond_part1, disjunction_effect)
            disjunction_conditional_2 = ConditionalEffect(disjunction_precond_part2, disjunction_effect)

            if self.phase == Location.EFFECTS and action in self.next_pal_tuple:
                new_model.actions[action].effects = LiteralConjunction([conjunction_conditional, disjunction_conditional_1, disjunction_conditional_2])
            elif self.phase == Location.EFFECTS and self.tried == 0:
                new_model.actions[action].effects = LiteralConjunction([conjunction_conditional])
            else:
                new_model.actions[action].effects = LiteralConjunction(
                    [conjunction_conditional, disjunction_conditional_1, disjunction_conditional_2])

            # if self.phase == Location.EFFECTS:
            new_model1_action = copy.deepcopy(new_model.actions[action])
            action_name = new_model1_action.name + "2"
            new_model1_action.name = action_name
            new_model.actions[action_name] = new_model1_action
            new_model.actions[action].preconds = new_operator_precond_1
            new_model.actions[action_name].preconds = new_operator_precond_2

            if self.phase == Location.EFFECTS:
                if (action in self.next_pal_tuple) and len(conjunction_effect.literals)>0:
                    new_model1_action2 = copy.deepcopy(new_model.actions[action])
                    action_name2 = new_model1_action2.name + "300"
                    new_model1_action2.name = action_name2

                    new_model.actions[action_name2] = new_model1_action2
                    test_ltr = copy.deepcopy(self.next_pal_tuple[0])
                    test_ltr_neg = copy.deepcopy(self.next_pal_tuple[0].inverted_anti)
                    if self.model1.mode == Literal.POS:
                        self.rename_literal_with_suffix(test_ltr, "_1")
                        self.rename_literal_with_suffix(test_ltr_neg, "_2")
                    else:
                        self.rename_literal_with_suffix(test_ltr_neg, "_1")
                        self.rename_literal_with_suffix(test_ltr, "_2")

                    new_model.actions[action_name2].preconds = LiteralConjunction([test_ltr,test_ltr_neg])
                    new_model.actions[action_name2].effects = disjunction_effect

        self.combined_model = new_model


    def get_directory_for_query_number(self, query_no):

        return "%s/q%s" % (self.base_dir, query_no)


    # def get_modified_model(self):

    def verify_if_precond_satisfied(self, model, problem, initial_state=None):
        def get_first_element(dictionary):
            for key in dictionary:
                return dictionary[key].preconds
            raise IndexError

        def create_dummy_init_state(initial_state, precond, problem):
            precond_names = set(list(map(lambda x: x.predicate.name, precond)))
            precond_vars = list(map(lambda x: x.variables, precond))
            precond_vars = set(list(chain.from_iterable(precond_vars)))
            new_init_state = {}
            objs = copy.deepcopy(problem.objects)
            for _p in precond_vars:
                remove_o = None
                for _o in objs:
                    if _o.var_type == _p.var_type:
                        new_init_state[_p] = _o
                        remove_o = _o
                        break
                if remove_o is not None:
                    objs.remove(remove_o)
            goal_formula = []
            check_precond = []
            new_precond = copy.deepcopy(precond)
            for _p in new_precond:
                temp_var = []
                for _v in _p.variables:
                    name_new = new_init_state[_v].name
                    new_ent = TypedEntity(new_init_state[_v].name, _v.var_type)
                    temp_var.append(new_ent)
                    # new_ent._update_variable_caches()
                new_goal = copy.deepcopy(_p)
                precond_n = copy.deepcopy(_p)
                new_goal.predicate.name = new_goal.predicate.name.rstrip("_1").rstrip("_2")

                new_goal.set_variables(temp_var)
                precond_n.set_variables(temp_var)
                # new_goal._update_variable_caches
                goal_formula.append(new_goal)
                check_precond.append(precond_n)
            return goal_formula, check_precond



        if initial_state is None:
            initial_state = problem.initial_state

        assert len(model.actions) >0

        precond = get_first_element(model.actions).literals
        t = copy.deepcopy(precond)
        for p in t:
            pname = p.predicate.name.rstrip("_1").rstrip("_2")
            if pname == self.next_pal_tuple[0].predicate.name and p.variables == self.next_pal_tuple[0].variables:
                precond.remove(p)


        precond_names = set(list(map(lambda x: x.predicate.name, precond)))
        filtered_init = list(filter(lambda x: (x.predicate.name in precond_names), initial_state.literals))

        filtered_init_names = set(list(map(lambda x: x.predicate.name, precond)))

        # if not filtered_init_names.issubset(precond_names):
        goal_formula, precond_to_check = create_dummy_init_state(initial_state, precond, problem)
        for p in precond_to_check:
            if p not in filtered_init:
                return False, goal_formula


        return True, goal_formula



    def get_policy (self, model, problem, domain_file, problem_file, sas_file=None, objects=None,\
                              initial_state=None,\
                              goal=None, modified_init_state=True):
        output_dir = self.get_directory_for_query_number(self.query_no)
        FileUtils.initialize_directory(output_dir, clean=True)

        model.write(domain_file)

        if self.first_call:
            self.first_call = False

            # if "stack" in self.next_pal_tuple and Location.PRECOND in self.next_pal_tuple and \
            #         self.next_pal_tuple[0].predicate.name == "on":
            #     # if "pick-upa" in self.next_pal_tuple and Location.PRECOND in self.next_pal_tuple and self.next_pal_tuple[0].predicate.name == "handfull":
            #     print("Here")
            is_needed, goal_formula = self.verify_if_precond_satisfied(model, problem, initial_state)
            # TODO: This works flawlessly only for ergodic domains
            # For non -ergodic we should write a proper goal formula
            action_to_execute = self.next_pal_tuple[1]

            if is_needed is False or action_to_execute in self.applicable_action_state_cache:
                # print('Test')
                # if "pick-up" in self.next_pal_tuple and Location.PRECOND in self.next_pal_tuple and\
                #         self.next_pal_tuple[0].predicate.name == "handempty":
                #         print("Here")

                try:

                    self.problem.initial_state = \
                        self.applicable_action_state_cache[action_to_execute]
                    modified_init_state = True
                except KeyError:

                    # Should never reach here currently.
                    assert False

                    # USE THIS ACTION
                    self.problem.initial_state, total_steps = \
                        self.ground_truth_agent.get_state_where_action_is_applicable(
                            action_to_execute)
                    assert self.problem.initial_state is not None
                    self.applicable_action_state_cache[action_to_execute] = \
                        self.problem.initial_state
                    self.bfs_steps += total_steps

                self.agent_problem.initial_state = copy.deepcopy(self.problem.initial_state)
                    

                self.create_query_problem()
                
        if modified_init_state is True:
            problem.write(problem_file, objects=objects,
                              initial_state=self.problem.initial_state.literals,
                              goal=goal,
                              fast_downward_order=True)
        else:
            assert problem is not self.problem
            problem.write(problem_file, objects=objects,
                              initial_state=problem.initial_state.literals,
                              goal=goal,
                              fast_downward_order=True)

        # if "pick-up" in self.next_pal_tuple and Location.PRECOND in self.next_pal_tuple and \
        #         self.next_pal_tuple[0].predicate.name == "handempty":
        # # if "pick-upa" in self.next_pal_tuple and Location.PRECOND in self.next_pal_tuple and self.next_pal_tuple[0].predicate.name == "handfull":
        #     print("Here")
        policy = PRP.solve(domain_file, problem_file, output_dir=output_dir, raise_exception=False)

        if self.phase == Location.PRECOND:
            if not os.path.exists(sas_file):
                return None
            plan = FileUtils.get_plan_from_file(sas_file)
            return plan
        # elif policy is None:
        #     self.tried = 1
        #     self.create_query_domain()
        #     return self.get_policy(self.combined_model, problem, domain_file, problem_file,\
        #                            sas_file=sas_file, objects=objects,\
        #                    initial_state=initial_state,\
        #                    goal=goal)
        
        assert policy is not None
        return policy

    def can_agent_execute_action(self, simulator, prp_action_policy,
                                 state=None):

        if prp_action_policy is None:
            prp_action_policy = []
        samples = []

        sim_state = simulator.save_state()

        init_action = PRP.get_pddlgym_action(simulator, prp_action_policy[0],
                                        is_separated=True)

        # Use the state from the applicable action cache.
        init_action_name = init_action.predicate.name
        assert init_action_name in self.applicable_action_state_cache

        if state is None:
            state = self.applicable_action_state_cache[init_action_name]

        simulator.set_state(state)

        actions_list = []
        for prp_action in prp_action_policy:

            action = PRP.get_pddlgym_action(simulator, prp_action,
                                            is_separated=True)

            next_state, _, _, _ = simulator.step(action, True)
            execution_status = simulator.get_step_execution_status()

            prp_action_name = prp_action.split(" ")[0]
            prp_action_args = prp_action.split(" ")[1:]
            samples.append((state,
                            (prp_action_name, prp_action_args),
                            next_state,
                            execution_status))

            if not execution_status:

                break
            else:

                actions_list.append(prp_action)

            state = next_state

        simulator.restore_state(*sim_state)

        return actions_list, samples

    def can_execute_action(self, model, policy, modeln, type_comp, useful_states, problem = None):
        output_dir = self.get_directory_for_query_number(self.query_no)
        # FileUtils.initialize_directory(output_dir, clean=True)
        domain_file = "%s/domain_temp.pddl" % (output_dir)
        problem_file = "%s/problem_temp.pddl" % (output_dir)
        sas_file = "%s/sas_plan" % (output_dir)

        self.create_query_domain(modeln)
        poss_objects = []
        if policy is None:
            print("Here")
            raise PRPPolicyNotFoundException
        for p in policy:
            poss_objects.extend(p.split(" ")[1:])
        if modeln ==1 or modeln ==2:
            model = self.combined_model
            problem = self.problem
        else:
            model_new = Md(model)
            p_psi = Predicate("p_psi", 0, [], False, False, False)
            model_new.predicates["p_psi"] = p_psi
            poss_actions = []
            for p in policy:
                act = p.split(" ")[0].rstrip(string.digits)
                poss_actions.append(act)
            new_actions = copy.deepcopy(model_new.actions)
            for action in model_new.actions:
                if action not in poss_actions:
                    new_actions.pop(action)
                else:
                    filtered_preds = []
                    for key, val in self.preconds_learned.items():
                        k = list(key)

                        if isinstance(k[0],Ltr) and k[1] == self.next_pal_tuple[1] and (val == True or k[0] == self.next_pal_tuple[0]):
                            filtered_preds.append(k[0])
                        elif isinstance(k[1],Ltr) and k[0] == self.next_pal_tuple[1] and (val == True or k[1] == self.next_pal_tuple[0]):
                            filtered_preds.append(k[1])

                    # filtered_preds = list(filter(lambda x: ( list(x[0])[1] == action and x[1]==True), self.preconds_learned.items()))
                    poss_preds = filtered_preds
                    n = []
                    for l in new_actions[action].preconds.literals or \
                             l.inverted_anti in new_actions[action].preconds.literals:
                        if l.positive in poss_preds:
                            n.append(l)
                    new_actions[action].preconds = LiteralConjunction(n)
                    new_actions[action].effects = LiteralConjunction([p_psi])
            model_new.actions = new_actions
            # self.combined_model.actions = model_new.actions

            problem_new = copy.deepcopy(self.agent_problem)
            problem_new.predicates['p_psi'] = p_psi
            psi_literal = Ltr(p_psi, [])
            problem_new.goal = LiteralConjunction([psi_literal])

            model = model_new
            problem = problem_new
        objects = []
        for _i in problem.objects:
            # initial_state.add()
            if _i.name in poss_objects:
                objects.append(_i)
            continue
        
        # Always use the complete set of objects for the problem.
        # PRP etc will automatically remove grounded predicates not required
        # as a part of their reachability analysis.
        # problem.objects = objects
        initial_state = set()
        for _i in problem.initial_state.literals:
            # initial_state.add()

            object_list = list(map(lambda x: x.name, _i.variables))
            if set(object_list).issubset(poss_objects):
                initial_state.add(_i)
            continue
        problem.initial_state = State(frozenset(initial_state),
                                      problem.initial_state.objects,
                                      problem.initial_state.goal)
        problem.initial_state = self.modify_init_state(type_comp, policy, problem.initial_state, modeln, useful_states,)

        if modeln == 1 or modeln == 2:
            policy = self.get_policy(model, problem, domain_file, problem_file, sas_file= sas_file)
        else:
            policy = self.get_policy(model, problem, domain_file, problem_file, sas_file= sas_file, modified_init_state=False)

        print("Policy: ", policy)
        if policy is None:
            return False, problem.initial_state
        # for p in policy:
        #     for e in p.split(" "):
        #         if p.count(e)>1:
        #             return False
        # return (len(policy)>0)
        print("Returned: ", policy)
        return policy, problem.initial_state

    def get_policy_from_query(self, objects=None,
                              initial_state_literals=None, 
                              goal=None, agent_model= None):
        self.query_no += 1
        print("Query #", self.query_no)
        output_dir = self.get_directory_for_query_number(self.query_no)
        FileUtils.initialize_directory(output_dir, clean=True)
        
        domain_file = "%s/domain.pddl" % (output_dir)
        problem_file = "%s/problem.pddl" % (output_dir)
        sas_file = "%s/sas_plan" % (output_dir)

        if self.testing_only:
            self.combined_model = agent_model

        return self.get_policy(self.combined_model, self.problem, domain_file, problem_file, sas_file=sas_file, objects=objects,\
                           initial_state=initial_state_literals,\
                           goal=goal), self.query_no, self.agent_problem.initial_state
