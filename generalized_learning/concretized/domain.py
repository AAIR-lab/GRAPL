'''
Created on Jul 15, 2021

@author: anonymous
'''


import pathlib


import instantiate
import normalize
from pddl import conditions
import pddl
import translate
from util import constants
from util import file


class Domain:

    @staticmethod
    def initialize_predicate_arity_dict(predicates):

        predicate_arity_dict = {}
        arity_predicate_dict = {}

        for predicate in predicates:

            name = predicate.name
            arity = len(predicate.arguments)

            assert name not in predicate_arity_dict \
                or predicate_arity_dict[name] == arity
            predicate_arity_dict[name] = arity

            arity_set = arity_predicate_dict.setdefault(arity, set())
            arity_set.add(name)

        return predicate_arity_dict, arity_predicate_dict

    def __init__(self, domain_filepath):

        # Extract the domain specific args.
        domain_pddl = pddl.pddl_file.parse_pddl_file(
            "domain", domain_filepath)
        domain_name, \
            domain_requirements, \
            types, \
            constants, \
            predicates, \
            functions, \
            actions, \
            axioms = pddl.tasks.parse_domain(
                domain_pddl)

        self.filepath = domain_filepath
        self.domain_name = domain_name
        self.domain_requirements = domain_requirements
        self.types = types
        self.constants = constants
        self.predicates = predicates
        self.functions = functions
        self.actions = actions
        self.axioms = axioms
        self.predicate_arity_dict, self.arity_predicate_dict = \
            Domain.initialize_predicate_arity_dict(self.predicates)

    def add_predicate_arity_entry(self, name, arity):

        assert name not in self.predicate_arity_dict \
            or self.predicate_arity_dict[name] == arity
        self.predicate_arity_dict[name] = arity

        arity_set = self.arity_predicate_dict.setdefault(arity, set())
        arity_set.add(name)

    def get_filepath(self):

        return self.filepath

    def get_action_templates(self):

        return self.actions

    def get_max_action_params(self):

        max_params = 0

        for action in self.actions:

            assert action.num_external_parameters == len(action.parameters)
            max_params = max(max_params, len(action.parameters))

        return max_params
