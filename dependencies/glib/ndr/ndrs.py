from pddlgym.structs import Predicate, ground_literal, LiteralConjunction
from pddlgym.parser import Operator
from pddlgym.inference import find_satisfying_assignments
import numpy as np

### Noisy deictic rules
NOISE_OUTCOME = Predicate("noiseoutcome", 0, [])()

class MultipleOutcomesPossible(Exception):
    pass


class NDR:
    """A Noisy Deictic Rule has a lifted action, lifted preconditions, and a
       distribution over effects.

    Parameters
    ----------
    action : Literal
    preconditions : [ Literal ]
    effect_probs : np.ndarray
    effects : [Literal]
    """
    def __init__(self, action, preconditions, effect_probs, effects,
                 allow_redundant_variables=False):
        self._action = action
        self._preconditions = preconditions
        self._effect_probs = effect_probs
        self._effects = effects
        self._allow_redundant_variables = allow_redundant_variables

        assert isinstance(preconditions, list)
        assert len(effect_probs) == len(effects)

        # Exactly one effect should have the noise outcome
        if len(effects) > 0:
            assert sum([NOISE_OUTCOME in e for e in effects]) == 1

        self._reset_precondition_cache()
        self._reset_effect_cache()

    def __str__(self):
        effs_str = "\n        ".join(["{}: {}".format(p, eff) \
            for p, eff in zip(self.effect_probs, self.effects)])
        return """{}:
  Pre: {}
  Effs: {}""".format(self.action, self.preconditions, effs_str)

    def __repr__(self):
        return str(self)

    @property
    def action(self):
        return self._action
    
    @property
    def preconditions(self):
        return self._preconditions

    @property
    def effect_probs(self):
        return self._effect_probs

    @property
    def effects(self):
        return self._effects

    @action.setter
    def action(self, x):
        self._reset_precondition_cache()
        self._action = x

    @preconditions.setter
    def preconditions(self, x):
        self._reset_precondition_cache()
        self._preconditions = x

    @effect_probs.setter
    def effect_probs(self, x):
        # No need to reset any caches
        self._effect_probs = x

    @effects.setter
    def effects(self, x):
        self._reset_effect_cache()
        self._effects = x

    def _reset_precondition_cache(self):
        self._precondition_cache = {}
        self._reset_effect_cache()

    def _reset_effect_cache(self):
        self._effect_cache = {}

    def copy(self):
        """Create a new NDR. Literals are assumed immutable.
        """
        action = self.action
        preconditions = [p for p in self.preconditions]
        effect_probs = np.array(self.effect_probs)
        effects = [eff.copy() for eff in self.effects]
        return NDR(action, preconditions, effect_probs, effects,
            allow_redundant_variables=self._allow_redundant_variables)

    def find_substitutions(self, state, action):
        """Find a mapping from variables to objects in the state
        and action. If non-unique or none, return None.
        """
        cache_key = hash((frozenset(state), action))
        if cache_key not in self._precondition_cache:
            kb = state | { action }
            assert action.predicate == self.action.predicate
            conds = [self.action] + list(self.preconditions)
            assignments = find_satisfying_assignments(kb, conds,
                allow_redundant_variables=self._allow_redundant_variables)
            if len(assignments) != 1:
                result = None
            else:
                result = assignments[0]
                if not self._allow_redundant_variables:
                    assert len(result.values()) == len(set(result.values()))
            self._precondition_cache[cache_key] = result
        return self._precondition_cache[cache_key]

    def covers_transition(self, transition):
        """Check whether the action and preconditions cover the transition
        """
        state, action, effects = transition
        sigma = self.find_substitutions(state, action)
        return sigma is not None

    def get_covered_transitions(self, transitions):
        """Filter out only covered transitions
        """
        return [t for t in transitions if self.covers_transition(t)]

    def get_explained_transitions(self, transitions):
        """Filter out only covered and referenced effect transitions
        """
        return [t for t in transitions if self.covers_transition(t) and \
            self.effects_are_referenced(t)]

    def find_unique_matching_effect_index(self, transition):
        """Find the unique effect index that matches the transition.

        Note that the noise outcome always holds, but only return it
        if no other effects hold.

        Used for quickly learning effect probabilities.
        """
        state, action, effects = transition
        cache_key = hash((frozenset(state), action, frozenset(effects)))
        if cache_key not in self._effect_cache:
            sigma = self.find_substitutions(state, action)
            try:
                assert sigma is not None, "Rule assumed to cover transition"
            except AssertionError:
                import ipdb; ipdb.set_trace()
            selected_outcome_idx = None
            noise_outcome_idx = None
            for i, outcome in enumerate(self.effects):
                if NOISE_OUTCOME in outcome:
                    assert noise_outcome_idx is None
                    noise_outcome_idx = i
                else:
                    ground_outcome = {ground_literal(lit, sigma) for lit in outcome}
                    match = False
                    # Check if the ground outcome is equivalent to the effects
                    # before Anti's have been applied
                    if sorted(ground_outcome) == sorted(effects):
                        match = True
                    # Check if the ground outcome is equivalent to the effects
                    # after Anti's have been applied
                    else:
                        for lit in set(ground_outcome):
                            if lit.is_anti and lit.inverted_anti in ground_outcome:
                                ground_outcome.remove(lit)
                                ground_outcome.remove(lit.inverted_anti)
                        if sorted(ground_outcome) == sorted(effects):
                            match = True
                    if match:
                        if selected_outcome_idx is not None:
                            raise MultipleOutcomesPossible()
                        selected_outcome_idx = i
            if selected_outcome_idx is not None:
                result = selected_outcome_idx
            else:
                assert noise_outcome_idx is not None
                result = noise_outcome_idx
            self._effect_cache[cache_key] = result
        return self._effect_cache[cache_key]

    def objects_are_referenced(self, state, action, objs):
        """Make sure that each object is uniquely referenced
        """
        sigma = self.find_substitutions(state, action)
        if sigma is None:
            return False
        return set(objs).issubset(set(sigma.values()))

    def effects_are_referenced(self, transition):
        """Make sure that each object is uniquely referenced
        """
        state, action, effects = transition
        objs = set(o for lit in effects for o in lit.variables)
        return self.objects_are_referenced(state, action, objs)

    def _predict(self, state, action, ind):
        lifted_effects = self._effects[ind]
        sigma = self.find_substitutions(state, action)
        return { ground_literal(e, sigma) for e in lifted_effects }

    def predict_max(self, state, action):
        """Make the most likely prediction
        """
        ind = np.argmax(self._effect_probs)
        return self._predict(state, action, ind)

    def predict_sample(self, state, action):
        """Sample a prediction
        """
        ind = np.random.choice(len(self._effect_probs), p=self._effect_probs)
        return self._predict(state, action, ind)

    def determinize(self, name_suffix=0):
        """Create a deterministic operators with the most likely effects
        """
        op_name = "{}{}".format(self.action.predicate.name, name_suffix)
        probs, effs = self.effect_probs, self.effects
        max_idx = np.argmax(probs)
        max_effects = LiteralConjunction(sorted(effs[max_idx]))
        preconds = LiteralConjunction(sorted(self.preconditions) + [self.action])
        params = sorted({ v for lit in preconds.literals for v in lit.variables })
        return Operator(op_name, params, preconds, max_effects)


class NDRSet:
    """A set of NDRs with a special default rule.

    Parameters
    ----------
    action : Literal
        The lifted action that all NDRs are about.
    ndrs : [ NDR ]
        The NDRs. Order does not matter.
    default_ndr : NDR or None
        If None, one is created. Only should be not
        None when an existing NDR is getting copied.
    """
    def __init__(self, action, ndrs, default_ndr=None,
                 allow_redundant_variables=False):
        self.action = action
        self.ndrs = list(ndrs)
        self._allow_redundant_variables = allow_redundant_variables
        if default_ndr is None:
            self.default_ndr = self._create_default_ndr(action,
                allow_redundant_variables=allow_redundant_variables)
        else:
            self.default_ndr = default_ndr

        # Cannot have empty preconds
        for ndr in ndrs:
            assert len(ndr.preconditions) > 0
            assert ndr.action == action
        assert self.default_ndr.action == action

    def __str__(self):
        s = "\n".join([str(r) for r in self])
        return s

    def __iter__(self):
        return iter(self.ndrs + [self.default_ndr])

    def __len__(self):
        return len(self.ndrs) + 1

    @staticmethod
    def _create_default_ndr(action, allow_redundant_variables=False):
        """Either nothing or noise happens by default
        """
        preconditions = []
        effect_probs = [0.5, 0.5]
        effects = [{ NOISE_OUTCOME }, set()]
        return NDR(action, preconditions, effect_probs, effects,
            allow_redundant_variables=allow_redundant_variables)

    def find_rule(self, transition):
        """Find the (assumed unique) rule that covers this transition
        """
        for ndr in self.ndrs:
            if ndr.covers_transition(transition):
                return ndr
        return self.default_ndr

    def partition_transitions(self, transitions):
        """Organize transitions by rule
        """
        rules = list(self)
        assert rules[-1] == self.default_ndr
        transitions_per_rule = [ [] for _ in rules ]
        for t in transitions:
            rule = self.find_rule(t)
            idx = rules.index(rule)
            transitions_per_rule[idx].append(t)
        return transitions_per_rule

    def copy(self):
        """Copy all NDRs in the set.
        """
        action = self.action
        ndrs = [ndr.copy() for ndr in self.ndrs]
        default_ndr = self.default_ndr.copy()
        return NDRSet(action, ndrs, default_ndr=default_ndr)

    def predict_max(self, state, action):
        """Make the most likely prediction
        """
        rule = self.find_rule((state, action, None))
        return rule.predict_max(state, action)

    def predict_sample(self, state, action):
        """Sample a prediction
        """
        rule = self.find_rule((state, action, None))
        return rule.predict_sample(state, action)

    def is_valid(self, transitions):
        """Make sure each transition is covered once
        """
        for transition in transitions:
            selected_ndr = None
            for ndr in self.ndrs:
                if ndr.covers_transition(transition):
                    if selected_ndr is not None:
                        return False
                    selected_ndr = ndr
        return True


