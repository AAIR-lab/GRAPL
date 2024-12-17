
import random
import itertools
import math
from pddlgym import structs
import numpy as np
from pddlgym.parser import Operator

class ActionGenerator:

    def __init__(self, num_actions_r, action_extra_param_r,
                 action_precons_r,
                 num_effects_r,
                 action_effects_r, action_effect_bin_size,
                 types, predicates):

        self.num_actions_r = num_actions_r
        self.params_r = action_extra_param_r
        self.precons_r = action_precons_r
        self.num_effects_r = num_effects_r
        self.effects_r = action_effects_r
        self.bin_size = action_effect_bin_size

        assert self.bin_size >= 0.05 and self.bin_size <= 1.0

        self.total_bins = 1.0 / self.bin_size
        assert math.ceil(self.total_bins) == math.floor(self.total_bins)
        self.total_bins = int(self.total_bins)

        self.predicates = predicates
        self.types = types
        assert len(self.types) == 1

        self.min_params_needed = np.max([predicate.arity
            for _, predicate in self.predicates.items()])

        self.PRECON_NEGATION_PROBABILITY = 0.5
        self.EFFECT_NEGATION_PROBABILITY = 0.5

    def generate_action_name(self, idx):

        return "action%u-a" % (idx)

    def generate_param_name(self, idx):

        return "?x%u" % (idx)

    @staticmethod
    def generate_candidate_literals(param_list, predicates):

        types_param_dict = {}
        for param in param_list:

            type_param_list = types_param_dict.setdefault(param.var_type,
                                                          [])
            type_param_list.append(param.name)

        candidate_literals = []

        for predicate in predicates:

            try:
                permute_list =[types_param_dict[t]
                               for t in predicate.var_types]
                permutations = list(itertools.product(*permute_list))

                # This enforces uniqueness. A predicate p(x0, x0) can
                # never occur in the precondition due to this.
                for params in permutations:
                    if len(set(params)) == predicate.arity:
                        candidate_literals.append(predicate(*params))
            except KeyError:

                pass

        return candidate_literals

    def generate_param_list(self):

        num_params = self.min_params_needed + random.randint(*self.params_r)

        assert len(self.types) == 1
        param_list = [structs.TypedEntity(self.generate_param_name(idx),
                                          list(self.types.values())[0])
                      for idx in range(num_params)]

        return param_list

    def generate_preconditions(self, candidate_literals):

        num_precons = random.randint(*self.precons_r)
        num_precons = min(num_precons, len(candidate_literals))
        random.shuffle(candidate_literals)

        selected_candidates = random.sample(candidate_literals, k=num_precons)

        precondition_literals = []
        param_set = set()
        first = True
        for candidate in selected_candidates:

            param_set.update(candidate.variables)

            # Never add a literal directly from candidate literals since
            # the reference will be copied.
            #
            # Instead, add a new copy using literal.{positive,negative,etc}
            if not first and random.random() < self.PRECON_NEGATION_PROBABILITY:
                precondition_literals.append(candidate.negative)
            else:
                precondition_literals.append(candidate.positive)

            first = False

        return precondition_literals, param_set


    def generate_effects(self, preconditions, candidate_literals):

        num_effects = random.randint(*self.num_effects_r)
        effects = []
        param_set = set()

        for i in range(num_effects):

            num_preds = random.randint(*self.effects_r)
            num_preds = min(num_preds, len(candidate_literals))
            random.shuffle(candidate_literals)

            selected_literals = random.sample(candidate_literals, k=num_preds)

            effect_literals = []
            for literal in selected_literals:

                param_set.update(literal.variables)

                # Never add a literal directly from candidate literals since
                # the reference will be copied.
                #
                # Instead, add a new copy using literal.{positive,negative,etc}
                if literal in preconditions:
                    effect_literals.append(literal.inverted_anti)
                elif literal.negative in preconditions:
                    effect_literals.append(literal.positive)
                elif random.random() < self.EFFECT_NEGATION_PROBABILITY:
                    effect_literals.append(literal.inverted_anti)
                else:
                    effect_literals.append(literal.positive)

            effects.append(effect_literals)

        effect_probs = self.generate_effects_probabilities(effects)
        return effects, effect_probs, param_set


    def generate_effects_probabilities(self, effects):

        assert len(effects) <= self.total_bins
        current_bin_capacity = self.total_bins

        effect_probs = []
        for idx in range(len(effects)):

            if idx == len(effects) - 1:
                selected_capacity = current_bin_capacity
            else:
                max_valid_capacity = current_bin_capacity - len(effects) - idx - 1
                selected_capacity = random.randint(1, max_valid_capacity)

            effect_probs.append(round(self.bin_size * selected_capacity, 2))
            current_bin_capacity -= selected_capacity

        return effect_probs

    def optimize_param_list(self, param_list, used_params):

        new_param_list = []
        for param in param_list:

            if param in used_params:
                new_param_list.append(param)

        return new_param_list

    def generate_action(self, idx):

        action_name = self.generate_action_name(idx)

        param_list = self.generate_param_list()
        candidate_literals = ActionGenerator.generate_candidate_literals(
            param_list,
            self.predicates.values())

        used_params = set()

        precondition_literals, param_set = self.generate_preconditions(candidate_literals)
        used_params |= param_set

        effects, effect_probs, param_set = self.generate_effects(set(precondition_literals),
                                        candidate_literals)
        used_params |= param_set
        param_list = self.optimize_param_list(param_list, used_params)

        is_stochastic = len(effects) > 0

        precondition = structs.LiteralConjunction(precondition_literals)
        effects = [structs.LiteralConjunction(effect) for effect in effects]
        effect = structs.LiteralConjunction([
            structs.ProbabilisticEffect(effects, effect_probs)
        ])

        action = Operator(action_name, param_list, precondition, effect)
        action = action.flatten()
        action = action.optimize()
        action.enforce_unique_params = False

        return action, is_stochastic

    def generate_actions(self):

        num_actions = random.randint(*self.num_actions_r)
        actions = {}
        is_stochastic_domain = False

        for idx in range(num_actions):

            action, is_stochastic = self.generate_action(idx)

            actions[action.name] = action
            is_stochastic_domain |= is_stochastic

        return actions, is_stochastic_domain
