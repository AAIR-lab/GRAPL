"""Main file for NDR learning
"""
from .ndrs import NDR, NDRSet, NOISE_OUTCOME, MultipleOutcomesPossible
from pddlgym.structs import Not, Anti, ground_literal
from collections import defaultdict
from termcolor import colored
from scipy.optimize import minimize
import heapq as hq
import numpy as np
import copy
import time
import abc
import itertools


ALPHA = 0.5 # Weight on rule set size penalty
P_MIN = 1e-8 # Probability for an individual noisy outcome
VERBOSE = False
DEBUG = False

## Generic search
class SearchOperator:

    @abc.abstractmethod
    def get_children(self, node):
        raise NotImplementedError()


def run_greedy_search(search_operators, init_state, init_score, greedy_break=False, ndr_settings=None,
                      max_timeout=None, max_node_expansions=1000, rng=None, verbose=False):
    """Greedy search
    """
    start_time = time.time()

    if rng is None:
        rng = np.random.RandomState(seed=0)

    best_score, state = init_score, init_state

    if verbose:
        print("Starting greedy search with initial score", best_score)

    for n in range(max_node_expansions):
        if verbose:
            print("Expanding node {}/{}".format(n, max_node_expansions))
        found_improvement = False
        for search_operator in search_operators:
            scored_children = search_operator.get_children(state, ndr_settings=ndr_settings)
            for score, child in scored_children:
                if verbose and DEBUG:
                    import ipdb; ipdb.set_trace()
                if score > best_score:
                    state = child
                    best_score = score
                    found_improvement = True
                    if verbose:
                        print("New best score:", best_score)
                        print("New best state:", state)
                        print("from operator:", search_operator)
                    if greedy_break:
                        break
                if max_timeout and (time.time() - start_time > max_timeout):
                    print("WARNING: search timed out early")
                    return state
        if not found_improvement:
            break

    return state

def run_best_first_search(search_operators, init_state, init_score, ndr_settings=None,
                          max_timeout=None, max_node_expansions=1000, rng=None, verbose=False):
    """Best first search
    """
    start_time = time.time()

    if rng is None:
        rng = np.random.RandomState(seed=0)

    best_score, best_state = init_score, init_state

    queue = []
    hq.heappush(queue, (0, 0, init_state))

    if verbose:
        print("Starting search with initial score", best_score)

    for n in range(max_node_expansions):
        if len(queue) == 0:
            break
        if verbose:
            print("Expanding node {}/{}".format(n, max_node_expansions))
        _, _, state = hq.heappop(queue)
        for search_operator in search_operators:
            scored_children = search_operator.get_children(state, ndr_settings=ndr_settings)
            for score, child in scored_children:
                hq.heappush(queue, (score, rng.uniform(), child))
                if score > best_score:
                    best_state = child
                    best_score = score
                    if verbose:
                        print("New best score:", best_score)
                        print("New best state:", best_state)
                if max_timeout and (time.time() - start_time > max_timeout):
                    print("WARNING: search timed out early")
                    return best_state

    return best_state


## Helper functions
def iter_variable_names():
    """Generate unique variable names
    """
    i = 0
    while True:
        yield "?x{}".format(i)
        i += 1

def print_rule_set(rule_set):
    for action_predicate in sorted(rule_set):
        print(colored(action_predicate, attrs=['bold']))
        for rule in rule_set[action_predicate]:
            print(rule)

def print_transition(transition):
    print("  State:", transition[0])
    print("  Action:", transition[1])
    print("  Effects:", transition[2])

def invert_sigma(sigma):
    """
    """
    sigma_inverse = {}
    for k, v in sigma.items():
        if v not in sigma_inverse:
            sigma_inverse[v] = [k]
        else:
            sigma_inverse[v].append(k)
    return sigma_inverse

def ground_literal_multi(lit, multi_sigma):
    """
    """
    out = []
    vals_for_vars = [multi_sigma[v] for v in lit.variables]
    for choice in itertools.product(*vals_for_vars):
        subs = dict(zip(lit.variables, choice))
        ground_lit = ground_literal(lit, subs)
        out.append(ground_lit)
    return out

def get_unique_transitions(transitions):
    """Filter out transitions that are literally (pun) identical
    """
    unique_transitions = []
    seen_hashes = set()
    for s, a, e in transitions:
        hashed = (frozenset(s), a, frozenset(e))
        if hashed not in seen_hashes:
            unique_transitions.append((s, a, e))
        seen_hashes.add(hashed)
    return sorted(unique_transitions)

## Scoring
def get_pen(rule):
    """Helper for scores. Counts number of literals in rule to penalize
    """
    pen = 0
    preconds = rule.preconditions
    pen += len(preconds)
    for effect in rule.effects:
        pen += len(effect)
    return pen

def get_transition_likelihood(transition, rule, p_min=P_MIN, ndr_settings=None):
    """Calculate the likelihood of a transition for a rule that covers it
    """
    try:
        effect_idx = rule.find_unique_matching_effect_index(transition)
        prob, outcome = rule.effect_probs[effect_idx], rule.effects[effect_idx]
        # Non-noise outcome
        if NOISE_OUTCOME not in outcome:
            transition_likelihood = prob
        # Noise outcome
        else:
            transition_likelihood = p_min * prob
        # if transition_likelihood == 0.:
            # import ipdb; ipdb.set_trace()
    except MultipleOutcomesPossible:
        state, action, effects = transition
        sigma = rule.find_substitutions(state, action)
        assert sigma is not None, "Rule assumed to cover transition"
        transition_likelihood = 0.
        for prob, outcome in zip(rule.effect_probs, rule.effects):
            if NOISE_OUTCOME in outcome:
                # c.f. equation 3 in paper
                transition_likelihood += p_min * prob
            else:
                ground_outcome = {ground_literal(lit, sigma) for lit in outcome}
                # Check if the ground outcome is equivalent to the effects
                # before Anti's have been applied
                if sorted(ground_outcome) == sorted(effects):
                    transition_likelihood += prob
                # Check if the ground outcome is equivalent to the effects
                # after Anti's have been applied
                else:
                    for lit in set(ground_outcome):
                        if lit.is_anti and lit.inverted_anti in ground_outcome:
                            ground_outcome.remove(lit)
                            ground_outcome.remove(lit.inverted_anti)
                    if sorted(ground_outcome) == sorted(effects):
                        transition_likelihood += prob
    return transition_likelihood

def score_action_rule_set(action_rule_set, transitions_for_action, p_min=P_MIN, alpha=ALPHA,
                          ndr_settings=None):
    """Score a full rule set for an action

    Parameters
    ----------
    action_rule_set : NDRSet
    transitions_for_action : [ (set, Literal, set) ]
        List of (state, action, effects).
    """
    score = 0.

    # Calculate penalty for number of literals
    for rule in action_rule_set:
        pen = get_pen(rule)
        score += - alpha * pen

    # Calculate transition likelihoods per example and accumulate score
    for transition in transitions_for_action:
        # Figure out which rule covers the transition
        selected_ndr = action_rule_set.find_rule(transition)
        # Calculate transition likelihood
        transition_likelihood = get_transition_likelihood(transition, 
            selected_ndr, p_min=p_min, ndr_settings=ndr_settings)
        # Terminate early if likelihood is -inf
        if transition_likelihood == 0.:
            return -10e8
        # Add to score
        score += np.log(transition_likelihood)

    return score

def score_rule(rule, transitions_for_rule, p_min=P_MIN, alpha=ALPHA, compute_penalty=True,
               ndr_settings=None):
    """Score a single rule on examples that it covers

    Parameters
    ----------
    rule : NDR
    transitions_for_rule : [ (set, Literal, set) ]
        List of (state, action, effects).
    """
    # Calculate penalty for number of literals
    score = 0
    if compute_penalty:
        pen = get_pen(rule)
        score += - alpha * pen

    # Calculate transition likelihoods per example and accumulate score
    for transition in transitions_for_rule:
        # Calculate transition likelihood
        transition_likelihood = get_transition_likelihood(transition, rule, 
            p_min=p_min, ndr_settings=ndr_settings)
        # Add to score
        if transition_likelihood == 0.:
            return -10e8
        score += np.log(transition_likelihood)

    return score


## Learn parameters
def learn_parameters(rule, covered_transitions, maxiter=100, ndr_settings=None):
    """Learn effect probabilities given the rest of a rule

    Parameters
    ----------
    rule : NDR
    covered_transitions : [(set, Literal, set)]
    """
    # First check whether all of the rule effects are mutually exclusive.
    # If so, we can compute analytically!
    try:
        return learn_params_analytically(rule, covered_transitions, ndr_settings=ndr_settings)
    except MultipleOutcomesPossible:
        pass

    # Set up the loss
    def loss(x):
        rule.effect_probs = x
        return -1. * score_rule(rule, covered_transitions, compute_penalty=False, 
            ndr_settings=ndr_settings)

    # Set up init x
    x0 = [1./len(rule.effects) for _ in rule.effects]

    # Run optimization
    cons = [{'type': 'eq', 'fun' : lambda x: sum(x) - 1. }]
    bounds=[(0, 1) for i in range(len(x0))]
    result = minimize(loss, x0, method='SLSQP', constraints=tuple(cons), bounds=bounds,
        options={'disp' : False, 'maxiter' : maxiter})
    params = result.x
    assert all((0 <= p <= 1.) for p in params), "Optimization does not obey bounds"

    # Finish rule
    rule.effect_probs = params

def learn_params_analytically(rule, covered_transitions, ndr_settings=None):
    """Assuming effects are mutually exclusive, find best params"""
    effect_counts = [0. for _ in rule.effects]
    for transition in covered_transitions:
        # Throws a caught error if there is no unique matching effect
        idx = rule.find_unique_matching_effect_index(transition)
        effect_counts[idx] += 1
    denom =  np.sum(effect_counts)
    if denom == 0:
        rule.effect_probs = np.ones(len(effect_counts), dtype=np.float32) / len(effect_counts)
    else:
        rule.effect_probs = np.array(effect_counts) / np.sum(effect_counts)


## Induce outcomes
class InduceOutcomesSearchOperator(SearchOperator):
    """Boilerplate for searching over effect distributions
    """
    def __init__(self, rule, covered_transitions, ndr_settings=None):
        self._rule_copy = rule.copy() # feel free to modify in-place
        self._covered_transitions = covered_transitions

    def get_children(self, probs_and_effects, ndr_settings=None):
        """Get new effects, get new probs and scores, then yield
        """
        _, effects = probs_and_effects
        for new_effects in self.get_child_effects(effects, ndr_settings=ndr_settings):
            new_probs = self.get_probs(new_effects, ndr_settings=ndr_settings)
            score = self.get_score(new_probs, new_effects, ndr_settings=ndr_settings)
            yield score, (new_probs, new_effects)

    def get_probs(self, effects, ndr_settings=None):
        self._rule_copy.effects = effects
        learn_parameters(self._rule_copy, self._covered_transitions,
            ndr_settings=ndr_settings)
        return self._rule_copy.effect_probs.copy()

    def get_score(self, probs, effects, ndr_settings=None):
        self._rule_copy.effect_probs = probs
        self._rule_copy.effects = effects
        return score_rule(self._rule_copy, self._covered_transitions,
            ndr_settings=ndr_settings)

    @abc.abstractmethod
    def get_child_effects(self, effects, ndr_settings=None):
        raise NotImplementedError()


class InduceOutcomesAddOperator(InduceOutcomesSearchOperator):
    """Pick a pair of non-contradictory outcomes and conjoin them
       (making sure not to conjoin with noiseoutcome)
    """
    def get_child_effects(self, effects, ndr_settings=None):
        for i in range(len(effects)-1):
            if NOISE_OUTCOME in effects[i]:
                continue
            for j in range(i+1, len(effects)):
                if NOISE_OUTCOME in effects[j]:
                    continue
                # Check for contradiction
                contradiction = False
                for lit_i in effects[i]:
                    if contradiction:
                        break
                    for lit_j in effects[j]:
                        if Anti(lit_i.predicate) == lit_j:
                            contradiction = True
                            break
                if contradiction:
                    continue
                # Create new set of effects that combines the two
                combined_effects = sorted(set(effects[i]) | set(effects[j]))
                # Get the other effects
                new_effects = []
                for k in range(len(effects)):
                    if k in [i, j]:
                        continue
                    new_effects.append(effects[k])
                # Add the new effect
                new_effects.append(combined_effects)
                yield new_effects


class InduceOutcomesRemoveOperator(InduceOutcomesSearchOperator):
    """Drop an outcome (not the noise one though!)
    """
    def get_child_effects(self, effects, ndr_settings=None):
        for i, effect_i in enumerate(effects):
            if NOISE_OUTCOME in effect_i:
                continue
            new_effects = [e for j, e in enumerate(effects) if j != i]
            yield new_effects


def create_induce_outcomes_operators(rule, covered_transitions, ndr_settings=None):
    """Search operators for outcome induction
    """
    add_operator = InduceOutcomesAddOperator(rule, covered_transitions,
        ndr_settings=ndr_settings)
    remove_operator = InduceOutcomesRemoveOperator(rule, covered_transitions,
        ndr_settings=ndr_settings)
    return [add_operator, remove_operator]

def get_all_possible_outcomes(rule, covered_transitions, ndr_settings=None):
    """Create initial outcomes as all possible ones
    """
    # For default rule, the only possible outcomes are noise and nothing
    if len(rule.preconditions) == 0:
        all_possible_outcomes = { (NOISE_OUTCOME,), tuple() }
    else:
        all_possible_outcomes = { (NOISE_OUTCOME,) }
        for state, action, effects in covered_transitions:
            sigma = rule.find_substitutions(state, action)
            assert sigma is not None
            sigma_inverse = invert_sigma(sigma)
            # If there is some object in the effects that does not appear in
            # the rule, this outcome is noise
            lifted_effects = []
            include_effects = True
            for e in effects:
                if not include_effects:
                    break
                try:
                    lifted_es = ground_literal_multi(e, sigma_inverse)
                except (KeyError, TypeError):
                    include_effects = False
                    break
                # Don't allow repeated effects, for efficiency
                if len(lifted_es) > 1:
                    include_effects = False
                    break
                lifted_effects.append(lifted_es[0])
            if include_effects:
                all_possible_outcomes.add(tuple(sorted(lifted_effects)))
    return all_possible_outcomes

def induce_outcomes(rule, covered_transitions, max_node_expansions=100, ndr_settings=None):
    """Induce outcomes for a rule

    Modifies the rule in place.
    """
    # Initialize effects with uniform distribution over all possible outcomes
    all_possible_outcomes = get_all_possible_outcomes(rule, covered_transitions,
        ndr_settings=ndr_settings)
    num_possible_outcomes = len(all_possible_outcomes)
    rule.effect_probs = [1./num_possible_outcomes] * num_possible_outcomes
    rule.effects = [list(outcome) for outcome in sorted(all_possible_outcomes)]
    # Search for better parameters
    learn_parameters(rule, covered_transitions, ndr_settings=ndr_settings)
    # Search for better effects
    init_state = (rule.effect_probs, rule.effects)
    init_score = score_rule(rule, covered_transitions, ndr_settings=ndr_settings)
    search_operators = create_induce_outcomes_operators(rule, covered_transitions,
        ndr_settings=ndr_settings)
    best_probs, best_effects = run_greedy_search(search_operators, init_state, init_score,
        max_node_expansions=max_node_expansions, ndr_settings=ndr_settings)
    rule.effect_probs = best_probs
    rule.effects = best_effects

## Main search operators
def create_default_rule_set(action, transitions_for_action, ndr_settings=None):
    """Helper for create default rule set. One default rule for action.
    """
    allow_redundant_variables = ndr_settings.get('allow_redundant_variables', False)
    variable_name_generator = iter_variable_names()
    variable_names = [next(variable_name_generator) for _ in range(action.arity)]
    lifted_action = action(*variable_names)
    ndr = NDR(action=lifted_action, preconditions=[], effect_probs=[], effects=[],
        allow_redundant_variables=allow_redundant_variables)
    covered_transitions = ndr.get_explained_transitions(transitions_for_action)
    induce_outcomes(ndr, covered_transitions, ndr_settings=ndr_settings)
    action_rule_set = NDRSet(lifted_action, [], default_ndr=ndr,
        allow_redundant_variables=allow_redundant_variables)
    score = score_action_rule_set(action_rule_set, transitions_for_action,
        ndr_settings=ndr_settings)
    return score, action_rule_set


class TrimPreconditionsSearchOperator(SearchOperator):
    """Helper for ExplainExamples step 2
    """
    def __init__(self, rule, transitions, ndr_settings=None):
        self._rule = rule
        self._transitions = transitions
        self._covered_transitions = get_unique_transitions(
            self._rule.get_explained_transitions(transitions)
        )

        # Comment this out b/c slow
        # assert self.check_if_valid(rule.preconditions)

    def get_score(self, preconditions, ndr_settings=None):
        """Get a score for a possible set of preconditions
        """
        allow_redundant_variables = ndr_settings.get('allow_redundant_variables', False)
        rule = self._rule.copy()
        rule.preconditions = preconditions
        rule_set = NDRSet(rule.action, [rule], allow_redundant_variables=allow_redundant_variables)
        # Induce outcomes for both rules
        rule_transitions, default_transitions = \
            rule_set.partition_transitions(self._transitions)
        induce_outcomes(rule, rule_transitions, ndr_settings=ndr_settings)
        induce_outcomes(rule_set.default_ndr, default_transitions, ndr_settings=ndr_settings)
        return score_action_rule_set(rule_set, self._transitions, ndr_settings=ndr_settings)

    def check_if_valid(self, preconditions, verbose=False, ndr_settings=None):
        # Covered by default rule
        if len(preconditions) == 0:
            return False
        # All objects in effect must be referenced
        rule = self._rule.copy()
        rule.preconditions = preconditions
        for transition in self._covered_transitions:
            if not rule.covers_transition(transition):
                if verbose:
                    print("Not valid because transition not covered:")
                    print(transition)
                return False
            state, action, effects = transition
            effected_objects = set(o for e in effects for o in e.variables)
            if not rule.objects_are_referenced(state, action, effected_objects):
                if verbose:
                    print("Not valid because objects not references:")
                    print(effected_objects)
                return False
        return True

    def get_children(self, remaining_preconditions, ndr_settings=None):
        for i in range(len(remaining_preconditions)):
            child_preconditions = [remaining_preconditions[j] \
                for j in range(len(remaining_preconditions)) if i != j]
            if self.check_if_valid(child_preconditions, ndr_settings=ndr_settings):
                score = self.get_score(child_preconditions, ndr_settings=ndr_settings)
                yield score, child_preconditions


class TrimObjectsSearchOperator(TrimPreconditionsSearchOperator):
    def get_children(self, remaining_preconditions, ndr_settings=None):
        all_variables = {v for lit in remaining_preconditions for v in lit.variables}
        for var_to_drop in sorted(all_variables):
            child_preconditions = []
            for lit in remaining_preconditions:
                if var_to_drop not in lit.variables:
                    child_preconditions.append(lit)
            if self.check_if_valid(child_preconditions, ndr_settings=ndr_settings):
                score = self.get_score(child_preconditions, ndr_settings=ndr_settings)
                yield score, child_preconditions



class ExplainExamples(SearchOperator):
    """Explain examples, the beefiest search operator

    Tries to follow the pseudocode in the paper as faithfully as possible
    """

    def __init__(self, action, transitions_for_action, max_ee_transitions=np.inf, 
                 rng=None, ndr_settings=None, **kwargs):
        self.action = action
        self.transitions_for_action = transitions_for_action
        self.unique_transitions = get_unique_transitions(transitions_for_action)
        self.max_transitions = max_ee_transitions
        self.rng = rng

    def _get_default_transitions(self, action_rule_set, ndr_settings=None):
        """Get unique transitions that are covered by the default rule
        """
        if not np.isinf(self.max_transitions):
            # Make sure that nontrivial transitions are first, otherwise random
            self.unique_transitions.sort(key=lambda t: (len(t[2]) == 0, self.rng.uniform()))

        default_transitions = []
        for transition in self.unique_transitions:
            covering_rule = action_rule_set.find_rule(transition)
            if covering_rule == action_rule_set.default_ndr:
                default_transitions.append(transition)
            if len(default_transitions) >= self.max_transitions:
                break
        return default_transitions

    @staticmethod
    def init_new_rule_action(transition, ndr_settings=None):
        a = transition[1]
        # Step 1.1: Create an action and context for r
        # Create new variables to represent the arguments of a
        variable_name_generator = iter_variable_names()
        # Use them to create a new action substition
        variables = [next(variable_name_generator) for _ in a.variables]
        sigma = dict(zip(variables, a.variables))
        sigma_inverse = invert_sigma(sigma)
        assert all(len(v) == 1 for v in sigma_inverse.values())
        # Set r's action
        return a.predicate(*[sigma_inverse[val][0] for val in a.variables]), \
            variable_name_generator

    @staticmethod
    def get_overfitting_preconditions_for_action(transition, ndr_settings=None):
        """Helper for Step 1.
        """
        s, a, _ = transition
        # Helper for checks
        new_rule = NDR(action=None, preconditions=[], effect_probs=[], effects=[],
            allow_redundant_variables=ndr_settings.get('allow_redundant_variables', False))
        new_rule.action, variable_name_generator = ExplainExamples.init_new_rule_action(transition,
            ndr_settings=ndr_settings)
        sigma = dict(zip(new_rule.action.variables, a.variables))
        sigma_inverse = invert_sigma(sigma)
        # Build up overfitting preconds
        overfitting_preconditions = []
        # Set r's context to be the conjunction literals that can be formed using
        # the variables
        for lit in s:
            if all(val in sigma_inverse for val in lit.variables):
                lifted_lits = ground_literal_multi(lit, sigma_inverse)
                overfitting_preconditions.extend(lifted_lits)
        return new_rule, sigma_inverse, variable_name_generator, overfitting_preconditions

    @staticmethod
    def get_changed_objects(transition, sigma_inverse, ndr_settings=None):
        """Helper for Step 1.
        """
        _, _, effs = transition
        changed_objects = set()
        for lit in effs:
            for val in lit.variables:
                if val not in sigma_inverse:
                    changed_objects.add(val)
        return changed_objects

    @staticmethod
    def add_deictic_refs(transition, changed_objects, sigma_inverse, variable_name_generator, 
                         overfitting_preconditions, new_rule, ndr_settings=None):
        """Helper for Step 1.
        """
        s, a, effs = transition
        for c in sorted(changed_objects):
            # Create a new variable and extend sigma to map v to c
            new_variable = next(variable_name_generator)
            assert c not in sigma_inverse
            sigma_inverse[c] = [new_variable]
            # Create the conjunction of literals containing c, but lifted
            d = []
            for lit in s:
                if c not in lit.variables:
                    continue
                if all(val in sigma_inverse for val in lit.variables):
                    lifted_lits = ground_literal_multi(lit, sigma_inverse)
                    d.extend(lifted_lits)
            # Check if d uniquely refers to c in s
            new_rule_copy = new_rule.copy()
            new_rule_copy.preconditions.extend(overfitting_preconditions+d)
            if new_rule_copy.objects_are_referenced(s, a, [c]):
                overfitting_preconditions.extend(d)

    @staticmethod
    def get_overfitting_preconditions(transition, ndr_settings=None):
        """Helper for Step 1. Also used by AddLits.
        """
        new_rule, sigma_inverse, variable_name_generator, overfitting_preconditions = \
            ExplainExamples.get_overfitting_preconditions_for_action(transition,
                ndr_settings=ndr_settings)
        if DEBUG: import ipdb; ipdb.set_trace()
        
        # Step 1.2: Create deictic references for r
        # Collect the set of constants whose properties changed from s to s' but 
        # which are not in sigma
        changed_objects = ExplainExamples.get_changed_objects(transition, sigma_inverse,
            ndr_settings=ndr_settings)
        # Get deictic references
        ExplainExamples.add_deictic_refs(transition, changed_objects, sigma_inverse, 
            variable_name_generator, overfitting_preconditions, new_rule,
            ndr_settings=ndr_settings)

        ## DEPARTURE FROM ZPK ##
        # Look for unreferenced objects that are appear with at least one referenced
        # object in some literal
        s, a, effs = transition
        referenced_objects = changed_objects | set(a.variables)
        nearby_objects = set()
        for lit in s:
            if len(set(lit.variables) & referenced_objects) > 0:
                nearby_objects.update(set(lit.variables) - referenced_objects)

        # Add deictic refs for nearby objects
        ExplainExamples.add_deictic_refs(transition, nearby_objects, sigma_inverse, 
            variable_name_generator, overfitting_preconditions, new_rule,
            ndr_settings=ndr_settings)
        ## END DEPARTURE ##

        return overfitting_preconditions

    def _initialize_new_rule(self, transition, ndr_settings=None):
        """Step 1: Create a new rule
        """
        new_rule = NDR(action=None, preconditions=[], effect_probs=[], effects=[],
            allow_redundant_variables=ndr_settings.get('allow_redundant_variables', False))
        # Init the action
        new_rule.action, _ = self.init_new_rule_action(transition, ndr_settings=ndr_settings)
        # Create preconditions
        new_rule.preconditions = self.get_overfitting_preconditions(transition,
            ndr_settings=ndr_settings)
        # Complete the rule
        # Call InduceOutComes to create the rule's outcomes.
        covered_transitions = new_rule.get_covered_transitions(self.transitions_for_action)
        induce_outcomes(new_rule, covered_transitions, ndr_settings=ndr_settings)

        if DEBUG: import ipdb; ipdb.set_trace()
        assert new_rule.effects is not None
        if DEBUG: import ipdb; ipdb.set_trace()
        return new_rule

    @staticmethod
    def trim_preconditions(rule, transitions_for_action, ndr_settings=None):
        """Step 2: Trim literals from the rule
        """
        # Create a rule set R' containing r and the default rule
        # Greedily trim literals from r, ensuring that r still covers (s, a, s')
        # and filling in the outcomes using InduceOutcomes until R's score stops improving
        op = TrimPreconditionsSearchOperator(rule, transitions_for_action,
            ndr_settings=ndr_settings)
        init_state = list(rule.preconditions)
        init_score = op.get_score(init_state, ndr_settings=ndr_settings)
        best_preconditions = run_greedy_search([op], init_state, init_score,
            greedy_break=True, ndr_settings=ndr_settings)
        # import ipdb; ipdb.set_trace()
        rule.preconditions = best_preconditions
        if DEBUG: import ipdb; ipdb.set_trace()
        # Greedily trim objects
        op = TrimObjectsSearchOperator(rule, transitions_for_action,
            ndr_settings=ndr_settings)
        init_state = list(rule.preconditions)
        init_score = op.get_score(init_state, ndr_settings=ndr_settings)
        best_preconditions = run_greedy_search([op], init_state, init_score,
            greedy_break=True, ndr_settings=ndr_settings)
        rule.preconditions = best_preconditions
        if DEBUG: import ipdb; ipdb.set_trace()


    def _create_new_rule_set(self, old_rule_set, new_rule, ndr_settings=None):
        """Step 3: Create a new rule set containing the new rule
        """
        allow_redundant_variables = ndr_settings.get('allow_redundant_variables', False)
        # Create a new rule set R' = R
        new_rules = [new_rule]
        # Add r to R' and remove any rules in R' that cover any examples r covers
        # Leave out default rule
        for rule in old_rule_set.ndrs:
            keep_rule = True
            for t in self.transitions_for_action:
                if new_rule.covers_transition(t) and rule.covers_transition(t):
                    keep_rule = False
                    break
            if keep_rule:
                new_rules.append(rule)
        # New rule set
        new_rule_set = NDRSet(new_rule.action, new_rules,
            allow_redundant_variables=allow_redundant_variables)
        # Recompute the parameters of the new rule and default rule
        default_rule = new_rule_set.default_ndr
        partitions = new_rule_set.partition_transitions(self.transitions_for_action)
        induce_outcomes(new_rule, partitions[0], ndr_settings=ndr_settings)
        induce_outcomes(default_rule, partitions[-1], ndr_settings=ndr_settings)
        if DEBUG: import ipdb; ipdb.set_trace()
        return new_rule_set

    def get_children(self, action_rule_set, ndr_settings=None):
        """The successor
        """
        # Get unique transitions that are covered by the default rule
        transitions = self._get_default_transitions(action_rule_set)

        for i, transition in enumerate(transitions):
            if VERBOSE:
                print("Running explain examples for action {} {}/{}".format(self.action, i, 
                    len(transitions)), end='\r')
                if i == len(transitions) -1:
                    print()
            if DEBUG: print("Considering explaining example for transition")
            if DEBUG: print_transition(transition)

            # Step 1: Create a new rule
            new_rule = self._initialize_new_rule(transition, ndr_settings=ndr_settings)
            # If preconditions are empty, don't enumerate; this should be covered by the default rule
            if len(new_rule.preconditions) == 0:
                continue
            # Filter out if not all effects explained
            if not new_rule.effects_are_referenced(transition):
                continue
            # Step 2: Trim literals from r
            self.trim_preconditions(new_rule, self.transitions_for_action, ndr_settings=ndr_settings)
            # If preconditions are empty, don't enumerate; this should be covered by the default rule
            if len(new_rule.preconditions) == 0:
                continue
            
            # Step 3: Create a new rule set containing r
            new_rule_set = self._create_new_rule_set(action_rule_set, new_rule, ndr_settings=ndr_settings)
            # Add R' to the return rule sets R_O
            score = score_action_rule_set(new_rule_set, self.transitions_for_action, ndr_settings=ndr_settings)
            yield score, new_rule_set


class DropRules(SearchOperator):
    """Search operator that drops one rule from the set
    """
    def __init__(self, transitions_for_action, ndr_settings=None, **kwargs):
        self.transitions_for_action = transitions_for_action

    def get_children(self, action_rule_set, ndr_settings=None):
        # Don't drop the default rule
        for i in range(len(action_rule_set.ndrs)):
            new_rule_set = action_rule_set.copy()
            del new_rule_set.ndrs[i]
            # Refit default rule
            partitions = new_rule_set.partition_transitions(self.transitions_for_action)
            learn_parameters(new_rule_set.default_ndr, partitions[-1], ndr_settings=ndr_settings)
            score = score_action_rule_set(new_rule_set, self.transitions_for_action,
                ndr_settings=ndr_settings)
            yield score, new_rule_set


class DropLits(SearchOperator):
    """Search operator that drops one lit per rule from the set
    """
    def __init__(self, transitions_for_action, ndr_settings=None, **kwargs):
        self.transitions_for_action = transitions_for_action

    def get_children(self, action_rule_set, ndr_settings=None):
        # Don't drop the default rule
        for i, ndr in enumerate(action_rule_set.ndrs):
            num_preconds = len(ndr.preconditions)
            # Can't overlap with default rule
            if num_preconds <= 1:
                continue
            for drop_i in range(num_preconds):
                new_rule_set = action_rule_set.copy()
                new_ndr = new_rule_set.ndrs[i]
                del new_ndr.preconditions[drop_i]
                # Validate
                if not new_rule_set.is_valid(self.transitions_for_action):
                    continue
                partitions = new_rule_set.partition_transitions(self.transitions_for_action)
                # Induce new outcomes for modified ndr
                induce_outcomes(new_ndr, partitions[i], ndr_settings=ndr_settings)
                # Update default rule parameters
                learn_parameters(new_rule_set.default_ndr, partitions[-1], ndr_settings=ndr_settings)
                score = score_action_rule_set(new_rule_set, self.transitions_for_action, 
                    ndr_settings=ndr_settings)
                yield score, new_rule_set


class DropObjects(SearchOperator):
    """Search operator that drops all lits associated with one object in each rule set
    """
    def __init__(self, transitions_for_action, ndr_settings=None, **kwargs):
        self.transitions_for_action = transitions_for_action

    def get_children(self, action_rule_set, ndr_settings=None):
        # Don't drop the default rule
        for i, ndr in enumerate(action_rule_set.ndrs):
            all_variables = {v for lit in ndr.preconditions for v in lit.variables}
            for var_to_drop in sorted(all_variables):
                new_rule_set = action_rule_set.copy()
                new_ndr = new_rule_set.ndrs[i]
                for j in range(len(ndr.preconditions)-1, -1, -1):
                    lit = ndr.preconditions[j]
                    if var_to_drop in lit.variables:
                        del new_ndr.preconditions[j]
                # Validate
                if not new_rule_set.is_valid(self.transitions_for_action):
                    continue
                partitions = new_rule_set.partition_transitions(self.transitions_for_action)
                # Induce new outcomes for modified ndr
                induce_outcomes(new_ndr, partitions[i], ndr_settings=ndr_settings)
                # Update default rule parameters
                learn_parameters(new_rule_set.default_ndr, partitions[-1], 
                    ndr_settings=ndr_settings)
                score = score_action_rule_set(new_rule_set, self.transitions_for_action, 
                    ndr_settings=ndr_settings)
                yield score, new_rule_set


class AddLits(SearchOperator):
    """Search operator that adds one lit per rule from the set
    """

    def __init__(self, transitions_for_action, ndr_settings=None, **kwargs):
        self.transitions_for_action = transitions_for_action
        self._all_possible_additions = self._get_all_possible_additions(transitions_for_action,
            ndr_settings=ndr_settings)

    def _get_all_possible_additions(self, transitions_for_action, ndr_settings=None):
        # Get all possible lits to add
        all_possible_additions = set()
        unique_transitions = get_unique_transitions(transitions_for_action)

        for transition in unique_transitions:
            preconds = ExplainExamples.get_overfitting_preconditions(transition, 
                ndr_settings=ndr_settings)
            all_possible_additions.update(preconds)
        return all_possible_additions

    def get_children(self, action_rule_set, ndr_settings=None):
        for i in range(len(action_rule_set.ndrs)):
            for new_lit in self._all_possible_additions:
                new_rule_set = action_rule_set.copy()
                new_ndr = new_rule_set.ndrs[i]
                # No use adding lits that are already here
                if new_lit in new_ndr.preconditions:
                    continue
                # Add the new lits
                new_ndr.preconditions.append(new_lit)
                # Trim preconditions
                # import ipdb; ipdb.set_trace()
                # This line leads to issues b/c preconditions may overlap
                # ExplainExamples.trim_preconditions(new_ndr, self.transitions_for_action)
                partitions = new_rule_set.partition_transitions(self.transitions_for_action)
                # Induce new outcomes for modified ndr
                induce_outcomes(new_ndr, partitions[i], ndr_settings=ndr_settings)
                # Update default rule parameters
                learn_parameters(new_rule_set.default_ndr, partitions[-1], ndr_settings=ndr_settings)
                # import ipdb; ipdb.set_trace()
                score = score_action_rule_set(new_rule_set, self.transitions_for_action, 
                    ndr_settings=ndr_settings)
                yield score, new_rule_set


class SplitOnLits(AddLits):
    """Search operator that splits on a literal, creating two new rules
    """

    def get_children(self, action_rule_set, ndr_settings=None):
        for i in range(len(action_rule_set.ndrs)):
            for new_lit in self._all_possible_additions:
                # if new_lit.predicate.name == "start":
                    # import ipdb; ipdb.set_trace()
                # No use adding a lit that's already there
                if new_lit in action_rule_set.ndrs[i].preconditions or \
                   Not(new_lit) in action_rule_set.ndrs[i].preconditions:
                    continue
                new_rule_set = action_rule_set.copy()
                pos_ndr = new_rule_set.ndrs[i]
                pos_ndr.preconditions.append(new_lit)
                neg_ndr = action_rule_set.ndrs[i].copy()
                neg_ndr.preconditions.append(Not(new_lit))
                new_rule_set.ndrs.insert(i+1, neg_ndr)
                partitions = new_rule_set.partition_transitions(self.transitions_for_action)
                # Induce new outcomes for modified ndrs
                induce_outcomes(pos_ndr, partitions[i], ndr_settings=ndr_settings)
                induce_outcomes(neg_ndr, partitions[i+1], ndr_settings=ndr_settings)
                # Update default rule parameters
                learn_parameters(new_rule_set.default_ndr, partitions[-1], 
                    ndr_settings=ndr_settings)
                score = score_action_rule_set(new_rule_set, self.transitions_for_action,
                    ndr_settings=ndr_settings)
                yield score, new_rule_set


def get_search_operators(action, transitions_for_action, ndr_settings=None, **kwargs):
    """Main search operators
    """
    explain_examples = ExplainExamples(action, transitions_for_action, 
        ndr_settings=ndr_settings, **kwargs)
    add_lits = AddLits(transitions_for_action, ndr_settings=ndr_settings, **kwargs)
    drop_rules = DropRules(transitions_for_action, ndr_settings=ndr_settings, **kwargs)
    drop_lits = DropLits(transitions_for_action, ndr_settings=ndr_settings, **kwargs)
    drop_objects = DropObjects(transitions_for_action, ndr_settings=ndr_settings, **kwargs)
    split_on_lits = SplitOnLits(transitions_for_action, ndr_settings=ndr_settings, **kwargs)

    return [
        explain_examples, 
        add_lits, 
        drop_rules,
        drop_lits,
        split_on_lits,
        drop_objects,
    ]

## Main
def run_main_search(transition_dataset, max_node_expansions=1000, rng=None, 
                    max_timeout=None, max_action_batch_size=None, get_batch_probs=lambda x : None,
                    init_rule_sets=None, search_method="greedy", allow_redundant_variables=False, 
                    **kwargs):
    """Run the main search
    """
    if rng is None:
        rng = np.random.RandomState(seed=0)
    
    ndr_settings = {'allow_redundant_variables' : allow_redundant_variables}

    rule_sets = {}

    for action, transitions_for_action in transition_dataset.items():
        if VERBOSE:
            print("Running search for action", action)

        if max_action_batch_size is not None and len(transitions_for_action) > max_action_batch_size:
            batch_probs = get_batch_probs(transitions_for_action)
            idxs = rng.choice(len(transitions_for_action), 
                size=max_action_batch_size, replace=False, p=batch_probs)
            transitions_for_action = [transitions_for_action[i] for i in idxs]

        search_operators = get_search_operators(action, transitions_for_action, 
            ndr_settings=ndr_settings, rng=rng, **kwargs)

        if init_rule_sets is None:
            init_score, init_state = create_default_rule_set(action, transitions_for_action,
                ndr_settings=ndr_settings)
        else:
            init_state = init_rule_sets[action]
            init_score = score_action_rule_set(init_state, transitions_for_action,
                ndr_settings=ndr_settings)

        if VERBOSE:
            print("Initial rule set (score={}):".format(init_score))
            print_rule_set({action : init_state})

        if search_method == "greedy":
            action_rule_set = run_greedy_search(search_operators, init_state, init_score, 
                max_timeout=max_timeout, max_node_expansions=max_node_expansions, ndr_settings=ndr_settings,
                rng=rng, verbose=VERBOSE)
        elif search_method == "best_first":
            action_rule_set = run_best_first_search(search_operators, init_state, init_score, 
                max_timeout=max_timeout, max_node_expansions=max_node_expansions, ndr_settings=ndr_settings,
                rng=rng, verbose=VERBOSE)
        else:
            raise NotImplementedError()

        rule_sets[action] = action_rule_set

    return rule_sets
