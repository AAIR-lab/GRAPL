"""Base class for a curiosity module.
"""

import time
import copy
import itertools
from collections import defaultdict
import abc
import random
from settings import AgentConfig as ac
from operator_learning_modules.foldt.foldt_operator_learning import FOLDTOperatorLearningModule
from pddlgym.inference import find_satisfying_assignments
from pddlgym import structs
from pddlgym.parser import PDDLProblemParser


class BaseCuriosityModule:
    """Base class for a curiosity module.
    """
    def __init__(self, action_space, observation_space, planning_module,
                 learned_operators, operator_learning_module, domain_name):
        self._action_space = action_space
        self._observation_space = observation_space
        self._planning_module = planning_module
        self._learned_operators = learned_operators
        self._operator_learning_module = operator_learning_module
        self._domain_name = domain_name

        self._action_space.seed(ac.seed)
        self._observation_space.seed(ac.seed)

        self._mutex_cache = {}
        self._initialize()

    @abc.abstractmethod
    def _initialize(self):
        """Initialize anything needed.
        """
        pass

    @abc.abstractmethod
    def reset_episode(self, state):
        """Defines whatever happens when a training episode ends.
        """
        pass

    @abc.abstractmethod
    def get_action(self, state):
        """Get the next action to take, given the current state.
        """
        pass

    @abc.abstractmethod
    def observe(self, state, action, effects):
        """Observe transitions.
        """
        pass

    def learning_callback(self):
        """Called when an operator changes.
        """
        self._mutex_cache = {}  # need to dump mutex_pairs cache because operators changed

    def _create_problem_pddl(self, state, goal, prefix):
        fname = "/tmp/{}_problem_{}.pddl".format(
            prefix, random.randint(0, 9999999))
        objects = state.objects
        all_action_lits = self._action_space.all_ground_literals(state)
        initial_state = state.literals | all_action_lits
        problem_name = "{}_problem".format(prefix)
        domain_name = self._planning_module.domain_name
        PDDLProblemParser.create_pddl_file(fname, objects, initial_state,
                                           problem_name, domain_name, goal)

        return fname

    def _get_predicted_next_state(self, state, action):
        """Get the next state resulting from the given state and action,
        under the current self._learned_operators. Returns None if either
        there is no learned operator for this action, or the preconditions
        are not satisfied.
        """
        if ac.learning_name == "TILDE":
            for act_pred, dt in self._operator_learning_module.learned_dts.items():
                if act_pred.name != action.predicate.name:
                    continue
                prediction = FOLDTOperatorLearningModule.get_prediction(state.literals | {action}, dt)
                if prediction is None:
                    return prediction
                effects = [structs.effect_to_literal(effect) for effect in prediction]
                return self._execute_effects(state, effects)
        elif ac.learning_name == "LNDR":
            for act_pred, ndrs in self._operator_learning_module._ndrs.items():
                if act_pred.name != action.predicate.name:
                    continue
                prediction = ndrs.predict_max(state.literals, action)
                return self._execute_effects(state, prediction)
        else:
            raise NotImplementedError()

        # No operator learned yet
        return None

    def sample_next_state(self, state, action):
        """Sample a next state. Only works if LNDR. If TILDE, just calls
        self._get_predicted_next_state(). Returns None if there is no
        learned operator for this action.
        """
        if ac.learning_name == "TILDE":
            return self._get_predicted_next_state(state, action)
        elif ac.learning_name == "LNDR":
            for act_pred, ndrs in self._operator_learning_module._ndrs.items():
                if act_pred.name != action.predicate.name:
                    continue
                prediction = ndrs.predict_sample(state.literals, action)
                return self._execute_effects(state, prediction)
        else:
            raise NotImplementedError()
        # No operator learned yet
        return None

    def _get_predicted_next_state_ops(self, state, action, mode="max"):
        """WARNING: Only use this method when self._learned_operators is
        GROUND TRUTH OPS!!!
        """
        for op in self._learned_operators:
            assignments = self._preconds_satisfied(state, action, op.preconds.literals)
            if assignments is not None:
                ground_effects = []
                for l in op.effects.literals:
                    if isinstance(l, structs.ProbabilisticEffect):
                        if mode == "max":
                            # assume max-probability event
                            chosen_effect = l.max()
                        elif mode == "sample":
                            # sample an event
                            chosen_effect = l.sample()
                        else:
                            raise Exception("Invalid mode: {}".format(mode))
                        if chosen_effect == "NOCHANGE":
                            continue
                        if isinstance(chosen_effect, structs.LiteralConjunction):
                            for lit in chosen_effect.literals:
                                ground_effects.append(structs.ground_literal(
                                    lit, assignments))
                        else:
                            ground_effects.append(structs.ground_literal(
                                chosen_effect, assignments))
                    else:
                        ground_effects.append(structs.ground_literal(l, assignments))
                return self._execute_effects(state, ground_effects)
        return state  # no change

    @staticmethod
    def _preconds_satisfied(state, action, literals):
        """Helper method for _get_predicted_next_state.
        """
        kb = state.literals | {action}
        assignments = find_satisfying_assignments(kb, literals)
        # NOTE: unlike in the actual environment, here num_found could be
        # greater than 1. This is because the learned operators can be
        # wrong by being overly vague, so that multiple assignments get
        # found. Here, if num_found is greater than 1, we just use the first.
        if len(assignments) == 1:
            return assignments[0]
        return None

    @staticmethod
    def _execute_effects(state, literals):
        """Helper method for _get_predicted_next_state.
        """
        new_literals = set(state.literals)
        for effect in literals:
            if effect.predicate.name.lower() == "nochange":
                continue
            # Negative effect
            if effect.is_anti:
                literal = effect.inverted_anti
                if literal in new_literals:
                    new_literals.remove(literal)
        for effect in literals:
            if effect.predicate.name.lower() == "nochange":
                continue
            if not effect.is_anti:
                new_literals.add(effect)
        return state.with_literals(new_literals)

    def _compute_static_preds(self):
        """Compute the static predicates under the current
        self._learned_operators.
        """
        static_preds = set()
        for pred in self._observation_space.predicates:
            if any(self._op_changes_predicate(op, pred)
                   for op in self._learned_operators):
                continue
            static_preds.add(pred)
        return static_preds

    @staticmethod
    def _op_changes_predicate(op, pred):
        """Helper method for computing static predicates.
        """
        for lit in op.effects.literals:
            assert not lit.is_negative
            if lit.is_anti:
                eff_pred = lit.inverted_anti.predicate
            else:
                eff_pred = lit.predicate
            if eff_pred == pred:
                return True
        return False

    def _compute_lifted_mutex_literals(self, initial_state):
        """Lifted mutex computation using MMM sampling-based algorithm.
        """
        print("Computing mutexes...(cache size = {})".format(len(self._mutex_cache)))
        if initial_state.literals in self._mutex_cache:
            mutex_pairs = self._mutex_cache[initial_state.literals]
            print("\tUsing cache with {} mutex pairs".format(len(mutex_pairs)))
            return mutex_pairs
        reachable_states = set()
        start_time = time.time()
        print("\tStep 1/2: constructing reachable states...")
        for _1 in range(ac.mutex_num_episodes[self._domain_name]):
            state = initial_state
            reachable_states.add(state.literals)
            for _2 in range(ac.mutex_episode_len[self._domain_name]):
                # Pick a random action.
                for _3 in range(ac.mutex_num_action_samples):
                    action = self._action_space.sample(state)
                    ground_effects = self._get_ground_effects(state, action)
                    if ground_effects is None:
                        continue
                    state = self._execute_effects(state, ground_effects)
                    reachable_states.add(state.literals)
                    break
        print("\tStep 2/2: finding mutex pairs...")
        mutex_pairs = set()
        for pred_pair in itertools.combinations(
                self._observation_space.predicates, 2):
            ph_to_pred_slot = {}
            lit_pair_with_phs = []
            # Create the placeholders
            for i, pred in enumerate(pred_pair):
                ph_for_lit = []
                for j, var_type in enumerate(pred.var_types):
                    ph = var_type('ph{}_{}'.format(i, j))
                    ph_to_pred_slot[ph] = (i, j)
                    ph_for_lit.append(ph)
                ph_lit = pred(*ph_for_lit)
                lit_pair_with_phs.append(ph_lit)
            phs = sorted(ph_to_pred_slot.keys())
            # Consider all substitutions of placeholders to variables
            for vs in self._iter_vars_from_phs(phs):
                lit_pair = [copy.deepcopy(lit) for lit in lit_pair_with_phs]
                # Perform substitution
                for k, v in enumerate(vs):
                    ph = phs[k]
                    (i, j) = ph_to_pred_slot[ph]
                    lit_pair[i].update_variable(j, v)
                # Lits cannot have repeated vars
                pair_valid = True
                for lit in lit_pair:
                    if len(set(lit.variables)) != len(lit.variables):
                        pair_valid = False
                        break
                if pair_valid:
                    # Call it mutex if it can't bind to any of the reachable states.
                    if not any(len(find_satisfying_assignments(state_lits, lit_pair)) > 0
                               for state_lits in reachable_states):
                        mutex_pairs.add(frozenset(lit_pair))
        print("\tFound {} mutex pairs in {} seconds".format(
            len(mutex_pairs), time.time()-start_time))
        self._mutex_cache[initial_state.literals] = mutex_pairs
        return mutex_pairs

    @staticmethod
    def _iter_vars_from_phs(phs):
        """Helper for _init_unseen_goal_actions."""
        num_phs = len(phs)
        for v_nums in itertools.product(range(num_phs), repeat=num_phs):
            # Filter out if any number is skipped
            valid = True
            for lo in range(num_phs-1):
                hi = lo+1
                if hi in v_nums and lo not in v_nums:
                    valid = False
                    break
            if not valid:
                continue

            # Filter out if any types are inconsistent
            v_num_to_type = {}
            for ph, v_num in zip(phs, v_nums):
                if v_num in v_num_to_type:
                    if v_num_to_type[v_num] != ph.var_type:
                        valid = False
                        break
                else:
                    v_num_to_type[v_num] = ph.var_type
            if not valid:
                continue

            # v_nums are valid
            vs = []
            for i, num in enumerate(v_nums):
                vt = phs[i].var_type
                v = vt("?x{}".format(num))
                vs.append(v)
            yield vs

    def _get_ground_effects(self, state, action):
        for op in self._learned_operators:
            assignments = self._preconds_satisfied(state, action, op.preconds.literals)
            if assignments is not None:
                ground_effects = [structs.ground_literal(l, assignments)
                                  for l in op.effects.literals]
                return ground_effects
        return None

    def _compute_mutex_literals(self, initial_state):
        """Top-level method for mutex. Compute the pairs of mutex literals
        under the current self._learned_operators, from the given initial
        state, up to the given max_level.
        """
        state = initial_state
        level_to_literal_mutex_pairs = {}
        level = 1
        while True:
            literal_mutex_pairs, next_links = self._mutex_one_level(state)
            level_to_literal_mutex_pairs[level] = literal_mutex_pairs
            next_state = self._execute_effects(state, next_links.keys())
            if state.literals == next_state.literals:  # planning graph has converged
                return level_to_literal_mutex_pairs
            state = next_state
            level += 1

    def _mutex_one_level(self, state):
        """Helper method for mutex. Computes mutex literals for a single level.
        """
        current_links, next_links = self._compute_links(state)
        acts_inconsistent_preconds = self._compute_inconsistent(current_links)
        acts_inconsistent_effects = self._compute_inconsistent(next_links)
        mutex_actions = acts_inconsistent_preconds + acts_inconsistent_effects
        literal_mutex_pairs = []
        for lit1 in next_links:
            for lit2 in next_links:
                if lit1.is_anti or lit2.is_anti:
                    continue
                if all({op1, op2} in mutex_actions
                       for op1 in next_links[lit1]
                       for op2 in next_links[lit2]):
                    if {lit1, lit2} not in literal_mutex_pairs:
                        literal_mutex_pairs.append({lit1, lit2})
        return literal_mutex_pairs, next_links

    @staticmethod
    def _compute_inconsistent(links):
        """Helper method for mutex. Computes actions that are mutex due to
        having inconsistent links (either current or next).
        """
        actions_inconsistent = []
        pos_lits = []
        neg_lits = []
        for lit in links:
            if lit.is_anti or lit.is_negative:
                neg_lits.append(lit)
            else:
                pos_lits.append(lit)
        for neg_lit in neg_lits:
            for pos_lit in pos_lits:
                if (neg_lit.is_anti and neg_lit.inverted_anti == pos_lit) or \
                   (neg_lit.is_negative and neg_lit.positive == pos_lit):
                    for op1 in links[pos_lit]:
                        for op2 in links[neg_lit]:
                            if {op1, op2} not in actions_inconsistent:
                                actions_inconsistent.append({op1, op2})
        return actions_inconsistent

    def _compute_links(self, state):
        """Helper method for mutex. Sets up state/action dependencies.
        """
        # Map from literal to all actions that it's a precond of.
        current_links = defaultdict(list)
        # Map from literal to all actions that it's an effect of.
        next_links = defaultdict(list)
        for lit in state.literals:
            pred = lit.predicate
            persist_pred = structs.Predicate(
                "PERSIST"+pred.name, pred.arity, pred.var_types,
                pred.is_negative, pred.is_anti, pred.negated_as_failure)
            persist_lit = structs.Literal(persist_pred, lit.variables)
            current_links[lit].append(persist_lit)
            next_links[lit].append(persist_lit)
        for action in self._action_space.all_ground_literals(state):
            for op in self._learned_operators:
                assignments = self._preconds_satisfied(
                    state, action, op.preconds.literals)
                if assignments is None:
                    continue
                ground_preconds = [structs.ground_literal(l, assignments)
                                   for l in op.preconds.literals]
                for precond in ground_preconds:
                    current_links[precond].append(action)
                ground_effects = [structs.ground_literal(l, assignments)
                                  for l in op.effects.literals]
                for effect in ground_effects:
                    next_links[effect].append(action)
        return current_links, next_links
