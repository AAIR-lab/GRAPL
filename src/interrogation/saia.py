#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import importlib
import itertools
import pickle
import pprint
import random
import time
import re
import subprocess
from collections import Counter, OrderedDict
from itertools import combinations
from itertools import product
from pddlgym.structs import Literal as Ltr
from pddlgym.structs import State, TypedEntity, Predicate, LiteralConjunction, ProbabilisticEffect

from config import *
from query import ExecutePlan
from lattice import LatticeNode, Lattice
from utils import *
import config

from exploration import random_walk
from exploration import intelligent_bfs

from src.planner.prp import PRP


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

    def __init__(self, agent, abstract_model, problem, query_files_dir, am,
                 evaluation_func, sampling_count, randomize=False,
                 should_count_sdm_samples=True,
                 explore_mode="all_bfs",
                 drift_mode=False):
        self.agent = agent
        self.am = am
        self.abstract_model = abstract_model
        self.sample_problem = problem
        self.randomize = randomize
        self.learned_preconditions = []
        self.predicates = abstract_model.predicates
        self.actions = abstract_model.actions
        abstract_model.predicates = {}
        self.sample_count_used = 0 # Track number of s,a,s' used
        self.learned_effects = []
        self.preconds_learned = {}
        self.effects_learned = {}
        self.effect_dict = {}
        self.effect_samples_dict = {}
        self.query_files_dir = query_files_dir
        self.learning_phase = Location.PRECOND
        self.current_pal_tuple = None
        self.useful_states = {}
        self.init_pal_tuple_dict()
        self.query_number = 0
        self.NUM_PAL_TUPLES = None
        self.sampling_count = sampling_count
        self.itr = 0
        self.evaluation_func = evaluation_func
        self.elapsed_learning_time = 0

        assert isinstance(should_count_sdm_samples, bool)
        self.should_count_sdm_samples = int(should_count_sdm_samples)

        self.empty_abstract_model = copy.deepcopy(self.abstract_model)

        # This map stores a cache of 1 state per applicable action
        # This helps reduce the total environment interactions
        self.applicable_action_state_cache = {}
        self.explore_mode = explore_mode

        self.ignore_combining_actions_set = set()
        self.drift_mode = drift_mode
        self.learned_actions = set()
        self.effect_state_dict = set()

        self.taboo_state_dict = {}

    def reset_for_task(self, agent, problem, am, query_files_dir):

        self.agent = agent
        self.sample_problem = problem
        self.am = am
        self.query_files_dir = query_files_dir

        self.query_number = 0
        self.itr = 0
        self.learning_phase = Location.PRECOND

        self.abstract_model.domain_name = problem.domain_name
        self.am.domain_name = problem.domain_name

        self.effect_dict = {}
        self.effect_samples_dict = {}
        self.learned_actions = set()
        self.taboo_state_dict = {}

        for action in self.abstract_model.actions:

            self.applicable_action_state_cache[action] = None
            self.reset_pals_for_action(action, self.preconds_learned, True)
            self.reset_pals_for_action(action, self.effects_learned, True)

    def learn_effects(self, action_name, state):

        self.applicable_action_state_cache[action_name] = state
        self.reset_pals_for_action(action_name, self.effects_learned)
        self.agent_interrogation_algo(action_name)

        return self.abstract_model

    def add_to_taboo(self, action_name):

        state = self.applicable_action_state_cache.get(action_name, None)
        taboo_state_set = self.taboo_state_dict.setdefault(action_name, set())
        if state is not None:
            taboo_state_set.add(state.literals)
        self.applicable_action_state_cache.pop(action_name, None)

    def prepare_action_for_complete_learning(self, action_name, state):

        self.applicable_action_state_cache[action_name] = state

        self.reset_pals_for_action(action_name, self.preconds_learned)
        self.prepare_action_for_effects_learning(action_name, state)

        # self.effect_dict.pop(action_name, None)
        # for name, effect in list(self.effect_samples_dict):
        #
        #     if name == action_name:
        #         del self.effect_samples_dict[(name, effect)]

        self.abstract_model.actions[action_name] = copy.deepcopy(
            self.empty_abstract_model.actions[action_name])


    def prepare_action_for_effects_learning(self, action_name,
                                            state):

        self.applicable_action_state_cache[action_name] = state
        self.reset_pals_for_action(action_name, self.effects_learned)



    def reset_action(self, action_name):

        self.applicable_action_state_cache.pop(action_name, None)
        self.prepare_action_for_learning(action_name)

    def prepare_action_for_learning(self, action_name):

        self.abstract_model.actions[action_name] = copy.deepcopy(
            self.empty_abstract_model.actions[action_name])

    def set_all_pals_to_true(self, action_name):

        self.reset_pals_for_action(action_name, self.preconds_learned,
                                   True)
        self.reset_pals_for_action(action_name, self.effects_learned,
                                   True)

    def reset_pals_for_action(self, action_name, pals, value=False):

        for pal in pals:
            if action_name in pal:
                pals[pal] = value

    def product_dict(self, **kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def get_compatible_literals(self, action, predicate):
        mapping = {}
        # predicate.var_types=['location','location']
        for i in range(len(predicate.var_types)):
            _v = predicate.var_types[i]
            action_var_list = list(map(lambda x: x.var_type, action.params))
            if action_var_list.count(_v) > 1:
                indices = [i for i in range(len(action_var_list)) if action_var_list[i] == _v]
                mapping[i] = indices
            elif action_var_list.count(_v) == 1:
                mapping[i] = [action_var_list.index(_v)]
            else:
                return None

        literal_mapping = list(dict(zip(mapping, x)) for x in itertools.product(*mapping.values()))
        poss_values = []
        for i in literal_mapping:
            t = []
            k = list(i.values())
            for j in k:
                if k.count(j) >1:
                    break
                t.append(action.params[j])
            if len(t) == len(k):
                poss_values.append(t)
        return poss_values

    def collect_pals_for_action(self, action_name):

        if action_name is None:

            return self.preconds_learned, self.effects_learned

        preconds_learned = {}
        for pal in self.preconds_learned:
            if action_name in pal:

                preconds_learned[pal] = self.preconds_learned[pal]

        effects_learned = {}
        for pal in self.effects_learned:

            if action_name in pal:

                effects_learned[pal] = self.effects_learned[pal]

        return preconds_learned, effects_learned

    def init_pal_tuple_dict(self):
        for _a in self.actions.keys():
            for _p in self.predicates.keys():
                if not self.predicates[_p].var_types:
                    l = Ltr(self.predicates[_p],[])
                    s = set([_a, l])
                    # Change to True for testing of effect learning
                    self.preconds_learned[frozenset(s)] = False
                    self.effects_learned[frozenset(s)] = False
                    if l not in self.useful_states.keys():
                        self.useful_states[l] = {Literal.NP: None, Literal.AP: None}
                else:
                    indexes = self.get_compatible_literals(self.actions[_a],self.predicates[_p])
                    if indexes is not None:
                        for _l in indexes:
                            l = Ltr(self.predicates[_p], _l)
                            s = set([_a,l])
                            # Change to True for testing of effect learning
                            self.preconds_learned[frozenset(s)] = False
                            self.effects_learned[frozenset(s)] = False
                            if l not in self.useful_states.keys():
                                self.useful_states[l] = {Literal.AN : None, Literal.AP : None}
                            
        self.NUM_PAL_TUPLES = len(self.preconds_learned) \
            + len(self.effects_learned)

    def get_action_for_current_pal_tuple(self):

        for _i in self.current_pal_tuple:
            if isinstance(_i, str) and _i in self.actions.keys():
                return _i

        return None

    def set_current_pal_tuple(self):
        for _i in self.current_pal_tuple:
            if isinstance(_i, str) and _i in self.actions.keys():
                _a = _i
            elif isinstance(_i, Ltr):
                l = _i
        s = set([_a, l])
        assert (frozenset(s) in self.preconds_learned.keys())
        if self.current_pal_tuple[2] == Location.PRECOND:
            self.preconds_learned[frozenset(s)] = True
        else:
            self.effects_learned[frozenset(s)] = True
    def get_next_pal_tuple(self, preconds_learned, effects_learned):
        pa_tuple = None
        if self.learning_phase == Location.PRECOND and False in preconds_learned.values():
            if True not in preconds_learned.values():
                # Special case for first pal tuple
                # TODO: Buggy if init state is empty
                temp_preconds_learned = copy.deepcopy(preconds_learned)
                while True:
                    pa_tuple = list(filter(lambda x: temp_preconds_learned[x] is False, temp_preconds_learned))[0]
                    predicate = None
                    for i in pa_tuple:
                        if isinstance(i, Ltr) and i.predicate.name in self.predicates.keys():
                            predicate = copy.deepcopy(i)
                    init_state_preds = set(map(lambda x: str(x).split("(")[0], self.sample_problem.initial_state.literals))
                    if predicate.predicate.name in init_state_preds:
                        break
                    else:
                        temp_preconds_learned.pop(pa_tuple)
            pa_tuple_list = list(filter(lambda x: preconds_learned[x] is False, preconds_learned))
            if self.randomize:
                pa_tuple = random.choice(pa_tuple_list)
            else:
                pa_tuple = pa_tuple_list[0]
        else:
            self.learning_phase = Location.EFFECTS
            if False in effects_learned.values():
                pa_tuple_list = list(filter(lambda x: effects_learned[x] is False, effects_learned))
                if self.randomize:
                    pa_tuple = random.choice(pa_tuple_list)
                else:
                    pa_tuple = pa_tuple_list[0]
            else:
                return None

        if pa_tuple is None:
            return pa_tuple

        for i in pa_tuple:
            if i in self.actions.keys():
                action = copy.deepcopy(i)
            else:
                predicate = copy.deepcopy(i)

        self.current_pal_tuple = [predicate, action, self.learning_phase]

        return self.current_pal_tuple

    def make_precond_inference(self, model1, model2, policy, ps_query, type_comp):

        agent_response, init_state = ps_query.can_execute_action(self.agent.get_domain(), policy, 0, type_comp, self.useful_states,
                                                    problem=self.sample_problem)

        agent_response_from_simulator, s = ps_query.can_agent_execute_action(
            self.agent.get_simulator(), policy, state=init_state)
        if config.SDMA_MODE:

            samples = []
            for _ in range(self.sampling_count):
                samples.append(s)

            self.analyze_samples(samples, self.abstract_model)

        m1_response, _ = ps_query.can_execute_action(model1, policy, 1, type_comp, self.useful_states)
        m2_response, _ = ps_query.can_execute_action(model2, policy, 2, type_comp, self.useful_states)

        print(m1_response, m2_response, agent_response, agent_response_from_simulator)
        assert ((agent_response) or ( m1_response) or ( m2_response))
        if "pick-up" in self.current_pal_tuple and Location.PRECOND in self.current_pal_tuple and \
                self.current_pal_tuple[0].predicate.name == "destroyed":
            print("Here")

        return (m1_response != agent_response), (m2_response != agent_response)

    def ground_effect(self, action_name, effect, objs):

        params_list = self.abstract_model.actions[action_name].params

        params_map = {}
        for i, obj in enumerate(objs):
            params_map[params_list[i]] = params_list[i].var_type(
                obj)

        new_effect = set()
        for literal in effect:

            literal = literal.ground(params_map, with_copy=True)
            new_effect.add(literal)

        return frozenset(new_effect)

    def effect_is_distinguishable(self, action_name, new_effect, samples):

        for s, a, objs, s_dash, execution_status in samples:

            assert execution_status
            if a != action_name:
                continue

            s = set(s.literals)
            s_dash = s_dash.literals

            grounded_new_effect = self.ground_effect(a, new_effect, objs)
            for literal in grounded_new_effect:

                if literal.is_anti:
                    s.discard(literal.inverted_anti)
                else:
                    s.add(literal)

            if frozenset(s) != s_dash:

                return True

        return False

    def try_merge(self, action_name, new_effect, s, a, objs, s_dash, es):

        if len(new_effect) == 0:

            return

        effects = self.effect_dict.setdefault(action_name, {})
        found = False
        for effect in list(effects):

            if new_effect == effect:

                samples = self.effect_samples_dict[(action_name, effect)]
                samples.append((s, a, objs, s_dash, es))
                self.effect_dict[action_name][effect] = len(samples)
                found = True
                continue


            samples = self.effect_samples_dict[(action_name, effect)]
            if not self.effect_is_distinguishable(action_name,
                                                  new_effect,
                                                  samples):


                samples.append((s, a, objs, s_dash, es))

                new_effect = frozenset(new_effect.union(effect))
                del self.effect_samples_dict[(action_name, effect)]
                del self.effect_dict[action_name][effect]

                prev_list = self.effect_samples_dict.setdefault(
                    (action_name, new_effect), [])
                self.effect_samples_dict[(action_name, new_effect)] = \
                    prev_list + samples
                self.effect_dict[action_name][new_effect] = \
                    len(self.effect_samples_dict[(action_name, new_effect)])
                found = True

        if not found:

            prev_list = self.effect_samples_dict.setdefault(
                    (action_name, new_effect), [])
            prev_list.append((s, a, objs, s_dash, es))
            self.effect_dict[action_name][new_effect] = len(prev_list)

    def analyze_samples(self, samples, model):

        # current_pal_action = self.get_action_for_current_pal_tuple()

        for s in samples:

            self.sample_count_used += len(s)

            if len(s) == 0:
                continue

            for pre_state, action, post_state, execution_status in s:
                if execution_status:

                    action_name = action[0]
                    action_vars = action[1]
                    action_name = PRP.remove_trailing_numbers_from_action(
                        action_name)
                    action_object = model.actions[action_name]
                    action_params = action_object.params
                    assert len(action_params) == len(action_vars)

                    action_preconditions = action_object.preconds
                    ground_preconditions = []
                    for _lit in action_preconditions.literals:
                        precond_var = []
                        for _v in _lit.variables:
                            precond_var.append(action_vars[action_params.index(_v)])
                        ground_preconditions.append(Ltr(_lit.predicate, precond_var))

                    ## Check if precond Satisfied --- Sanity check
                    for g in ground_preconditions:
                        if g not in pre_state and g.is_negative:
                            continue
                        # print(g,"-----", pre_state)
                        # assert g in pre_state

                    extra_literals = post_state.literals - pre_state.literals
                    missing_literals = pre_state.literals - post_state.literals
                    new_effect = []
                    for _lit in extra_literals:
                        lit_ground = []
                        for _v in _lit.variables:
                            lit_ground.append(action_params[action_vars.index(_v)])
                        new_effect.append(Ltr(_lit.predicate, lit_ground))
                    for _lit in missing_literals:
                        lit_ground = []
                        for _v in _lit.variables:
                            lit_ground.append(action_params[action_vars.index(_v)])
                        new_effect.append(Ltr(_lit.predicate.inverted_anti, lit_ground))
                        # new_effect.append(Ltr(_lit.predicate, precond_var))
                    new_effect = frozenset(set(new_effect))

                    # self.try_merge(action_name, new_effect, pre_state,
                    #                action_name, action_vars,
                    #                post_state, execution_status)

                    if action_name not in self.effect_dict.keys():
                        self.effect_dict[action_name] = {new_effect: 1}
                    elif new_effect in self.effect_dict[action_name].keys():
                        self.effect_dict[action_name][new_effect] += 1
                    else:
                        self.effect_dict[action_name][new_effect] = 1

                    sample_list = self.effect_samples_dict.setdefault(
                        (action_name, new_effect), [])
                    sample_list.append((pre_state, action_name, action_vars,
                                        post_state, execution_status))

        # Learn Common Effects
        # for action_name, effects in self.effect_dict.items():
        #     assert action_name in model.actions.keys()
        #     common_effects = None
        #     for effect in effects.keys():
        #         if not common_effects:
        #             common_effects = set(effect)
        #         else:
        #             common_effects = common_effects.intersection(set(effect))

        for action_name, effects in self.effect_dict.items():
            assert action_name in model.actions.keys()
            all_effects = copy.deepcopy(model.actions[action_name].effects.literals)
            assert (len(all_effects) == 0 or isinstance(all_effects[0], ProbabilisticEffect))
            existing_effects = []
            old_effects = False
            updated = False
            if len(all_effects) > 0:
                old_effects = True
                for _e in all_effects[0].literals:
                    if isinstance(_e, Ltr):
                        continue  # To handle existing nochange
                    existing_effects.append(frozenset(set(list(_e.literals))))

            # We learn only effect of this type


            for effect in effects.keys():
                if effect == frozenset():
                    mod_effect = Predicate("NOCHANGE", 0)()
                else:
                    mod_effect = LiteralConjunction(list(effect))

                if (effect not in existing_effects) and old_effects:
                    all_effects[0].literals.append(mod_effect)
                    all_effects[0].probabilities.append(0)
                elif (effect not in existing_effects):
                    all_effects.append(mod_effect)
                    updated = True

            if updated:
                probs = [0]*(len(all_effects))
                prob_effect = ProbabilisticEffect(all_effects, probs)
                prob_effect.is_flattened = True
                model.actions[action_name].effects = LiteralConjunction([prob_effect])

            if config.OVERRIDE_ACTION_EFFECTS_WHEN_ANALYZING_SAMPLES:
                for action_name, effects in self.effect_dict.items():
                    new_effects = [LiteralConjunction(list(effect)) for effect in effects]
                    if LiteralConjunction([]) in new_effects:
                        new_effects.remove(LiteralConjunction([]))
                    new_probs = [0] * len(new_effects)
                    prob_effect = ProbabilisticEffect(new_effects, new_probs)
                    model.actions[action_name].effects = LiteralConjunction([prob_effect])

                self.populate_probabilities(model)
                for action_name in self.effect_dict:
                    self.combine_action(model.actions[action_name])

    def make_effects_inference(self, policy, model, initial_state):
        samples = self.agent.generate_samples(
            policy,
            initial_state=initial_state,
            sampling_count=self.sampling_count)
        self.analyze_samples(samples, model)
        return [model]

    def populate_probabilities(self, model):
        for action_name, t_effects in self.effect_dict.items():
            effects = copy.deepcopy(t_effects)

            if sum(effects.values()) < 10:
                continue

            factor = 1.0 / sum(effects.values())

            idx = 0


            if len(model.actions[action_name].effects.literals) == 0:
                continue

            for _i in model.actions[action_name].effects.literals[0].literals:
                if isinstance(_i, Ltr) and _i.predicate.name.lower() == "nochange":
                    if frozenset(set()) in effects.keys():
                        val = effects[frozenset(set())]
                    else:
                        val = 0
                    model.actions[action_name].effects.literals[0].probabilities[idx] = val * factor
                    idx += 1
                    continue
                key = frozenset(set(list(_i.literals)))
                if key in effects.keys():
                    model.actions[action_name].effects.literals[0].probabilities[idx] = effects[key] * factor
                idx += 1

        return model


    def interrogate(self):

        assert self.should_count_sdm_samples
        self.elapsed_learning_time = 0

        if self.explore_mode == "intelligent_bfs":

            learned_model = self.abstract_model
            while len(self.applicable_action_state_cache) != len(self.actions):
                print("Performing intelligent bfs")
                total_steps, action_name = intelligent_bfs.intelligent_bfs(
                    self.agent.get_simulator(),
                    self.applicable_action_state_cache,
                    learned_model)
                self.sample_count_used += total_steps

                assert action_name is not None
                print("Learning action:", action_name)
                learned_model = self.agent_interrogation_algo(action_name)
        elif self.explore_mode == "random_walk":

            learned_model = self.abstract_model
            failed_count = 0
            while len(self.applicable_action_state_cache) != len(self.actions):
                print("Performing random walk")
                total_steps, action_name = random_walk.random_walk(
                    self.agent.get_simulator(),
                    self.applicable_action_state_cache,
                    learned_model,
                    drift_mode=self.drift_mode,
                    taboo_state_dict=self.taboo_state_dict)
                self.sample_count_used += total_steps

                assert action_name is not None
                print("Learning action:", action_name,
                      " Current sim step count:", self.get_total_steps())

                # Full learning of the action will commence.
                self.prepare_action_for_learning(action_name)

                try:
                    learned_model = self.agent_interrogation_algo(action_name)
                    failed_count = 0
                except Exception as e:

                    failed_count += 1
                    print("Failed...Attempt", failed_count)
                    taboo_state_set = self.taboo_state_dict.setdefault(
                        action_name, set())
                    taboo_state_set.add(
                        self.applicable_action_state_cache[action_name].literals)
                    del self.applicable_action_state_cache[action_name]

                    if failed_count == 10:

                        raise e

        elif self.explore_mode == "all_bfs":

            print("Performing exploration")
            total_steps = self.agent.initialize_applicable_action_cache(
                self.applicable_action_state_cache)
            self.sample_count_used += total_steps
            print("Learning. Current sim step count:", self.get_total_steps())
            learned_model = self.agent_interrogation_algo(None)

        print("Yayy!! All Done with %s samples" % (self.get_total_steps()))
        return learned_model

    def get_total_steps(self):

        simulator = self.agent.get_simulator()
        return simulator.get_total_steps()

    def combine_probabilities(self, effect, i, j, copy=False):

        effect.literals[i] = effect.literals[j]
        effect.probabilities[i] += effect.probabilities[j]
        effect.literals.remove(j)

    def can_find_distinguishing_sample(self, action_name, params_list,
                                       e1, e2):

        if (action_name, e1) not in self.effect_samples_dict:
            print("HERE", str(e1))
            assert (action_name, e1) in self.effect_samples_dict

        if (action_name, e2) not in self.effect_samples_dict:
            print("HERE", str(e2))
            assert (action_name, e2) in self.effect_samples_dict

        samples = self.effect_samples_dict[(action_name, e1)]
        for s, a, objs, s_dash, es in samples:

            assert es
            params_map = {}
            for i, obj in enumerate(objs):
                params_map[params_list[i]] = params_list[i].var_type(obj)

            e2_l = LiteralConjunction(list(e2)).ground(params_map, with_copy=True)
            if not e2_l.holds(s_dash.literals):

                return True

        return False

    def clean_literals(self, literal_set):

        literal_set = set(literal_set)
        for literal in list(literal_set):
            if literal.predicate.name.lower() == "nochange":
                literal_set.discard(literal)
                break

        return frozenset(literal_set)

    def get_effect_literals(self, effect, idx):

        if isinstance(effect.literals[idx], Ltr):
            assert effect.literals[idx].predicate.name.lower() == "nochange"
            el = frozenset(set())
        else:
            el = frozenset(effect.literals[idx].literals)

        return self.clean_literals(el)

    def check_and_combine_into(self, action_name, params_list,
                               effect, i, j):

        if effect.probabilities[i] == 0.0 \
            or effect.probabilities[j] == 0.0:

            return []

        e1 = self.get_effect_literals(effect, i)
        e2 = self.get_effect_literals(effect, j)

        e1_eq_e2 = not self.can_find_distinguishing_sample(action_name,
                                                           params_list,
                                                           e1, e2)
        e2_eq_e1 = not self.can_find_distinguishing_sample(action_name,
                                                           params_list,
                                                           e2, e1)

        if e2_eq_e1 and e1_eq_e2:

            return [(i, j)]

        return []

    def check_and_delete(self, action, new_effect, old_effect, i, j):

        old_i = self.get_effect_literals(old_effect, i)
        old_j = self.get_effect_literals(old_effect, j)

        new_i = self.get_effect_literals(new_effect, i)

        if old_i != new_i:

            self.effect_dict[action.name].pop(old_i, None)
            self.effect_samples_dict.pop((action.name, old_i), None)

        if old_j != new_i:
            self.effect_dict[action.name].pop(old_j, None)
            self.effect_samples_dict.pop((action.name, old_j), None)


    def combine_action(self, action):

        if not len(action.effects.literals) > 0:
            return

        assert isinstance(action.effects, LiteralConjunction)
        assert isinstance(action.effects.literals[0], ProbabilisticEffect)
        assert len(action.effects.literals) == 1

        effect = action.effects.literals[0]
        changed = True
        while changed:

            changed = False
            new_effects = []
            for i, j in itertools.combinations(
                    range(len(effect.literals)), 2):

                new_effects += self.check_and_combine_into(action.name,
                                                      action.params,
                                                      effect, i, j)

            if len(new_effects) > 0:
                changed = True
                indexes_to_pop = set()
                old_effect = copy.deepcopy(effect)
                for i, j in new_effects:

                    e1 = self.get_effect_literals(effect, i)
                    e2 = self.get_effect_literals(effect, j)
                    merged_effect = frozenset(e1.union(e2))

                    assert e1 in self.effect_dict[action.name]
                    assert e2 in self.effect_dict[action.name]

                    assert (action.name, e1) in self.effect_samples_dict
                    assert (action.name, e2) in self.effect_samples_dict

                    self.effect_samples_dict[(action.name, merged_effect)] = \
                        self.effect_samples_dict[(action.name, e1)] \
                        + self.effect_samples_dict[(action.name, e2)]

                    self.effect_dict[action.name][merged_effect] = \
                        self.effect_dict[action.name][e1] \
                        + self.effect_dict[action.name][e2]

                    effect.literals[i] = LiteralConjunction(
                        list(merged_effect))
                    effect.probabilities[i] += effect.probabilities[j]
                    indexes_to_pop.add(j)

                # Delete all effects that were merged.
                for i, j in new_effects:

                    self.check_and_delete(action, effect, old_effect, i, j)

                # Pop the indices
                for idx in sorted(list(indexes_to_pop), reverse=True):
                    effect.probabilities.pop(idx)
                    effect.literals.pop(idx)


    def agent_interrogation_algo(self, action_name=None):
        ps_query_module = importlib.import_module("query.ps_query")
        init_state = {}
        latt = Lattice()
        lattice_node = LatticeNode(latt, [self.abstract_model], self.abstract_model.predicates)

        # int_parent_models holds the possible models at any given point of time.
        int_parent_models = [self.abstract_model]

        self.learning_phase = Location.PRECOND
        preconds_learned, effects_learned = self.collect_pals_for_action(action_name)

        next_pal_tuple = self.get_next_pal_tuple(preconds_learned, effects_learned)
        init_state = self.sample_problem.initial_state

        preprocess_start_time = time.time()
        tried_cont = 0
        query_1_failed = 0
        discarded_count = 0
        new_discard_count = 0
        valid_models = None
        learning_time = time.time()
        while True:
            model_level = 0
            refined_for_agent_failure = False
            if next_pal_tuple is None:
                # All possible refinements over
                # int_parent_models should be holding possible models at most concretized level
                # return False, False, None, None, None, None, None, None
                break
            print("ACTUAL NEXT PAL TUPLE: ", next_pal_tuple)
            tmp_int_parent_models = []
            action_pred_rejection_combination = {}
            modes = [Literal.POS, Literal.NEG, Literal.ABS]

            for temp_abs_model in int_parent_models:

                pred = next_pal_tuple[0]
                action = next_pal_tuple[1]
                ref = next_pal_tuple[2]

                # partitions stores the partitions for a refinement next_pal_tuple when called on
                # a model temp_abs_model
                intermediate_models = lattice_node.get_model_partitions(temp_abs_model, pred,
                                                                        ref, action, tuple(modes))

                # Run query and discard models here
                # Remove all invalid models and store only the valid ones
                valid_models = [i for i in intermediate_models if not i.discarded]

                # Generate all possible combinations of models
                for m1, m2 in combinations(valid_models, 2):
                    if m1.discarded or m2.discarded:
                        continue
                    type_comp = type_comparison(m1.mode, m2.mode)


                    ps_query = ps_query_module.Query(
                        self.agent, self.query_files_dir, m1, m2, init_state,
                        next_pal_tuple, self.sample_problem,
                        self.learning_phase, preconds_learned,
                        self.applicable_action_state_cache, qno=self.query_number)

                    policy, self.query_number, new_init_state = ps_query.get_policy_from_query(agent_model=self.am)

                    # Count any steps that the query took as interactions
                    # with the environment.
                    assert ps_query.bfs_steps == 0
                    self.sample_count_used += \
                        self.should_count_sdm_samples * ps_query.bfs_steps

                    if self.learning_phase == Location.PRECOND:
                        m1.discarded, m2.discarded = self.make_precond_inference(m1, m2, policy, ps_query, type_comp)
                    if self.learning_phase == Location.EFFECTS:
                        valid_models = self.make_effects_inference(policy, temp_abs_model, new_init_state)

                valid_models = [i for i in valid_models if not i.discarded]
                # if len(valid_models) <3:
                self.set_current_pal_tuple()
                break
            int_parent_models = [i for i in valid_models if not i.discarded]
            # int_parent_models = [int_parent_models[0]]
            
            learning_time = time.time() - learning_time

            evaluation_model = copy.deepcopy(valid_models[0])
            evaluation_model.predicates = copy.deepcopy(self.am.predicates)
            evaluation_model.action_predicate_map = self.am.action_predicate_map
            evaluation_model = self.populate_probabilities(evaluation_model)

            self.elapsed_learning_time += learning_time

            if self.evaluation_func is not None:
                
                self.evaluation_func(
                    evaluation_model=evaluation_model,
                    itr=self.sample_count_used,
                    num_pal_tuples=self.NUM_PAL_TUPLES,
                    output_dir=ps_query.get_directory_for_query_number(
                        self.query_number),
                    elapsed_time=self.elapsed_learning_time,
                    query_number=self.query_number)
            
            learning_time = time.time()

            preconds_learned, effects_learned = self.collect_pals_for_action(action_name)
            self.current_pal_tuple = self.get_next_pal_tuple(preconds_learned, effects_learned)
            if self.current_pal_tuple is None:
                # Learned all tuples
                assert len(valid_models) == 1
                valid_models = valid_models[0]
                break
            next_pal_tuple = self.current_pal_tuple

        learned_model = valid_models
        learned_model.action_predicate_map = self.am.action_predicate_map
        # learned_model = self.populate_probabilities(valid_models)

        if not config.OVERRIDE_ACTION_EFFECTS_WHEN_ANALYZING_SAMPLES:
            learned_model = self.populate_probabilities(valid_models)

            for action in learned_model.actions.values():
                self.combine_action(action)

        self.abstract_model = learned_model
        assert not self.abstract_model.is_optimized

        if action_name is None:

            for action in learned_model.actions:
                self.learned_actions.add(action)
        else:
                self.learned_actions.add(action_name)

        return copy.deepcopy(learned_model)
