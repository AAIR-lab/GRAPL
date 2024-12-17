#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from glib.planning_modules import base_planner

from pddlgym.structs import ProbabilisticEffect
from pddlgym.structs import Literal as Ltr
from pddlgym.structs import LiteralConjunction
from config import *
import config

class UnconformantTransitionException(Exception):

    def __init__(self, s=None, a=None,
                 s_dash=None, execution_status=None):

        self.s = s
        self.a = a
        self.s_dash = s_dash
        self.execution_status = execution_status

class UnconformantPreconditionException(UnconformantTransitionException):

    def __init__(self, s, a, s_dash, execution_status):

        super(UnconformantPreconditionException, self).__init__(
            s, a, s_dash, execution_status)

class UnconformantEffectException(UnconformantTransitionException):

    def __init__(self, s, a, s_dash, execution_status):

        super(UnconformantEffectException, self).__init__(
            s, a, s_dash, execution_status)

def get_probabilistic_effect_npddl_str(self):
    
    assert len(self.literals) == len(self.probabilities)
    
    string = "(oneof"
    for literal in self.literals:
        
        if self.is_optimized:
            
            literal = LiteralConjunction(self.common_effects.literals +
                                         literal.literals)
        
        if isinstance(literal, Ltr) \
            and literal.predicate.name.lower() == "nochange":
            
            assert len(literal.variables) == 0
            string += " (and)"
        else:
            
            string += " " + literal.pddl_str()
    string +=")"
    return string

# Make sure that pddlgym still does not support converting probabilistic effects
# to strings.
#
# If it does (perhaps due to a newer version), then this might need to be 
# revisited.
dummy_effect = ProbabilisticEffect([], [0])
# try:
#     dummy_effect.pddl_str()
#     raise Exception("pddl_str() is working in pddlgym for probabilistic effects!")
# except NotImplementedError:
#
#     pass

# Override the probabilistic effect's pddl_str() of pddlgym with our own. 
ProbabilisticEffect.pddl_str = get_probabilistic_effect_npddl_str 

class Model:
    
    @staticmethod
    def clean_pddlgym_action_predicates(model, actions):
        
        # Clean the predicates list.
        for action in actions:
            
            model.predicates.pop(action, None)

        # Clean each action
        action_predicate_map = {}
        for action_name, action in model.actions.items():
            
            assert isinstance(action.preconds, LiteralConjunction)
            
            preconditions = action.preconds.literals
            
            for i in range(len(preconditions) -1, -1, -1):
                
                if preconditions[i].predicate.name in actions:
                    
                    action.action_predicate = preconditions.pop(i)
                    action_predicate_map[action_name] = action.action_predicate
                    
        model.action_predicate_map = action_predicate_map
    
    def restore_action_predicates(self):

        if self.operators_as_actions:

            return

        # Glib models do not use the map since they are guaranteed
        # to have the operators having the correct action predicate.
        if not self.is_glib and self.action_predicate_map is not None:
            for action_name, action in self.actions.items():
                
                assert isinstance(action.preconds, LiteralConjunction)
                
                preconditions = action.preconds.literals
                preconditions.append(self.action_predicate_map[action_name])
        
        else:
            for _, action in self.actions.items():
                
                assert isinstance(action.preconds, LiteralConjunction)
                
                preconditions = action.preconds.literals
                if action.action_predicate is not None:
                    preconditions.append(action.action_predicate)
    
    def __init__(self, domain, 
                 clean=not config.SHOULD_FIND_GYM_ACTION_PREDS, 
                 actions=None):
        self.discarded = False
        self.domain_name = domain.domain_name
        self.mode = Literal.ABS
        self.predicates = copy.deepcopy(domain.predicates)
        if not actions:
            actions = domain.operators
        self.actions = copy.deepcopy(actions)
        self.action_predicate_map = None
        self.is_glib = False
        self.operators_as_actions=True
        self.types = domain.types
        
        if clean:
            Model.clean_pddlgym_action_predicates(self, domain.actions)
            
        # TODO: Not in use currently.
        self.is_flattened = None
        self.is_optimized = None
        
        self.is_determinized = False

    def determinize(self, with_copy=True):

        if with_copy:
            
            model = copy.deepcopy(self)
        else:
            model = self
            
        for name in model.actions:
            
            model.actions[name] = model.actions[name].determinize(
                with_copy=False)
        
        model.is_determinized = True
        return model

    def flatten(self, with_copy=True):
        
        if with_copy:
            
            model = copy.deepcopy(self)
        else:
            model = self
            
        for name in model.actions:
            
            model.actions[name] = model.actions[name].flatten(with_copy=False)

        model.is_flattened = True
        return model

    def optimize(self, with_copy=True):

        if with_copy:
            
            model = copy.deepcopy(self)
        else:
            model = self

        assert model.is_flattened
        for name in model.actions:
            
            model.actions[name] = model.actions[name].optimize(with_copy=False)

        model.is_optimized = True
        return model

    def print_model(self):
        print("Domain Name: ", self.domain_name)
        print("Predicates: ", self.predicates.keys())
        print("Actions:")
        for _a in self.actions.items():
            print("Action: ", _a[0])
            print("\tParameters: ", _a[1].params)
            print("\tPrecondition: ", _a[1].preconds)
            print("\tEffects: ", _a[1].effects)
            
    def get_domain_name_pddl_str(self):
        
        return "(domain %s)" % (self.domain_name)
    
    def get_domain_requirements_pddl_str(self, with_probabilities=False):
        
        if self.is_determinized:
            
            assert not with_probabilities
            return "(:requirements :typing :strips :disjunctive-preconditions :negative-preconditions :equality)"
        elif with_probabilities:
            return "(:requirements :typing :strips :probabilistic-effects :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)"
        else:
            return "(:requirements :typing :strips :non-deterministic :disjunctive-preconditions :conditional-effects :negative-preconditions :equality)"
    
    def get_domain_types_pddl_str(self):
        
        types = {}
        for predicate in self.predicates.values():
            
            if isinstance(predicate, Ltr):
                
                type_dict = {str(t): t for t in predicate.predicate.var_types}
            else:
                type_dict = {str(t) : t for t in predicate.var_types}
        
            types.update(type_dict)
        
        # Untyped domains not supported atm.
        assert len(types) > 0
        return "(:types " + " ".join(types.values()) + ")"
    
    
    def get_domain_predicates_pddl_str(self):
        
        string = "(:predicates"
        
        for predicate in self.predicates.values():
            
            string += " " + predicate.pddl_str()
        
        string += ")"
        return string
    
    def _check_if_actions_optimized(self):
        
        for action in self.actions.values():
            
            if not action.is_optimized:
                
                return False
            
        return True
    
    def get_domain_actions_pddl_str(self, with_probabilities=False):
        
        if with_probabilities:
            
            assert self._check_if_actions_optimized()
            for action in self.actions.values():
                
                assert isinstance(action.effects, ProbabilisticEffect)
                action.effects.pddl_str = \
                    action.effects.get_probabilistic_pddl_str
        
        string = ""
        
        for action in self.actions.values():
            
            string += action.pddl_str()
            string += "\n"
        
        return string
    
    def get_domain_pddl_str(self, with_probabilities=False):
        
        return """
(define %s
%s
%s
%s
%s)""" % (self.get_domain_name_pddl_str(),
          self.get_domain_requirements_pddl_str(
              with_probabilities=with_probabilities),
          self.get_domain_types_pddl_str(),
          self.get_domain_predicates_pddl_str(),
          self.get_domain_actions_pddl_str(
              with_probabilities=with_probabilities))

    def write(self, domain_filepath, with_probabilities=False,
              close=True):

        if isinstance(domain_filepath, str):    
            filehandle = open(domain_filepath, "w")
        else:
            filehandle = domain_filepath
            
        filehandle.write(self.get_domain_pddl_str(
            with_probabilities=with_probabilities))

        filehandle.flush()
        if close:
            filehandle.close()

    def is_transition_conformant(self, s, a, s_dash,
                                 execution_status):

        action = self.actions[a.predicate.name]
        action = action.ground(a.variables, with_copy=True)

        assert action.is_optimized

        if execution_status != action.preconds.holds(s.literals):

            return False, None
        else:

            if not execution_status:
                assert s.literals == s_dash.literals
                return True, True

            next_states = action.effects.apply_all(s.literals)
            for next_state in next_states:

                if s_dash.literals == next_state:

                    return True, True

            return True, False
    def get_effect_idx(self, s, a, s_dash, execution_status):

        assert execution_status

        action = self.actions[a.predicate.name]
        action = action.ground(a.variables, with_copy=True)
        assert action.is_optimized

        next_states = action.effects.apply_all(s.literals)
        for i, next_state in enumerate(next_states):

            if action.effects.probabilities[i] == 0:
                continue

            if s_dash.literals == next_state:

                return i

        assert False
        return None
