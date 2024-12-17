import numpy as np
from .foldt import FOLDTClassifier, LearningFailure
from settings import AgentConfig as ac
from pddlgym.parser import Operator
from pddlgym.structs import Predicate, Literal, LiteralConjunction, NULLTYPE

from collections import defaultdict


def Effect(x):  # pylint:disable=invalid-name
    """An effect predicate or literal.
    """
    if isinstance(x, Predicate):
        return Predicate("Effect"+x.name, x.arity, var_types=x.var_types,
            is_negative=x.is_negative, is_anti=x.is_anti)
    assert isinstance(x, Literal)
    new_predicate = Effect(x.predicate)
    return new_predicate(*x.variables)

def effect_to_literal(literal):
    assert isinstance(literal, Literal)
    assert literal.predicate.name.startswith("Effect")
    non_effect_pred = Predicate(literal.predicate.name[len("Effect"):], literal.predicate.arity,
        literal.predicate.var_types, is_negative=literal.predicate.is_negative, 
        is_anti=literal.predicate.is_anti)
    return non_effect_pred(*literal.variables)



class FOLDTOperatorLearningModule:

    _NoChange = Predicate("NoChange", 0)

    def __init__(self, learned_operators):
        self._learned_operators = learned_operators
        self._learned_operators_for_action = defaultdict(set)
        self._Xs = defaultdict(list)
        self._Ys = defaultdict(list)
        self._seed = ac.seed
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self.learned_dts = {}
        self._fits_all_data = defaultdict(bool)
        self._learning_on = True

    def observe(self, state, action, effects):
        if not self._learning_on:
            return
        x = (state.literals | {action})
        y = sorted([Effect(e) for e in effects])

        if not y:
            y.append(Effect(self._NoChange()))

        self._Xs[action.predicate].append(x)
        self._Ys[action.predicate].append(y)
        # print("observed data:")
        # print(x)
        # print(y)
        # input("!!")

        # Check whether we'll need to relearn
        if self._fits_all_data[action.predicate]:
            dt = self.learned_dts[action.predicate]
            if not self._dt_fits_data(dt, x, y):
                self._fits_all_data[action.predicate] = False

    def learn(self):
        if not self._learning_on:
            return False
        is_updated = False
        for action_predicate in self._fits_all_data:
            if not self._fits_all_data[action_predicate]:
                # Learn one main tree.
                try:
                    dt = self._learn_single_dt(action_predicate, self._seed,
                                               self._Xs[action_predicate],
                                               self._Ys[action_predicate])
                except LearningFailure:
                    # Aggressive!!!!!!!!!
                    self._Xs[action_predicate] = []
                    self._Ys[action_predicate] = []
                    continue
                self.learned_dts[action_predicate] = dt
                operators = self._foldt_to_operators(dt, action_predicate.name)
                for op in operators:
                    if not sum(lit.predicate.name == action_predicate
                               for lit in op.preconds.literals) == 1:
                        # Something is wrong...why are multiple action
                        # predicates appearing in the preconditions?
                        import ipdb; ipdb.set_trace()
                # Remove old operators
                self._learned_operators_for_action[action_predicate].clear()
                # Add new operators
                self._learned_operators_for_action[action_predicate].update(operators)
                self._fits_all_data[action_predicate] = True
                is_updated = True

        # Update all learned operators
        if is_updated:
            self._learned_operators.clear()
            for operators in self._learned_operators_for_action.values():
                self._learned_operators.update(operators)

        return is_updated

    @staticmethod
    def _learn_single_dt(action_predicate, seed, X, Y,
                         bag_predicates=False, bag_features=False):
        root_literal = action_predicate(*['Placeholder{}'.format(i)
                                          for i in range(action_predicate.arity)])
        dt = FOLDTClassifier(seed=seed, root_feature=root_literal,
                             max_feature_length=ac.max_foldt_feature_length,
                             max_learning_time=ac.max_foldt_learning_time,
                             max_exceeded_strategy=ac.max_foldt_exceeded_strategy,
                             bag_predicates=bag_predicates, bag_features=bag_features)
        dt.fit(X, Y)
        return dt

    def turn_off(self):
        self._learning_on = False

    def _dt_fits_data(self, dt, x, y):
        prediction = self.get_prediction(x, dt)
        if prediction is None:
            match = y is None
        else:
            match = set(y) == set(prediction)
        # if not match:
        #     print("Mismatch:")
        #     print("x =", x)
        #     print("y =", y)
        #     print("prediction =", prediction)
        return match

    @classmethod
    def get_prediction(cls, x, dt):
        prediction = dt.predict(x)
        if prediction is None:
            return prediction
        # Cancel out effects if possible (e.g. At(x) and AntiAt(x))
        final_prediction = set()
        for lit in prediction:
            if lit.inverted_anti in final_prediction:
                final_prediction.remove(lit.inverted_anti)
            else:
                final_prediction.add(lit)
        if not final_prediction:
            final_prediction = [Effect(cls._NoChange())]
        return sorted(list(final_prediction))

    def _foldt_to_operators(self, dt, action_name, suffix=''):
        op_counter = 0
        operators = set()
        for (path, leaf) in dt.get_conditional_literals(dt.root):
            if leaf is None:
                continue
            if any(lit.predicate == Effect(self._NoChange) for lit in leaf):
                continue
            name = "LearnedOperator{}{}{}".format(action_name, op_counter, suffix)
            op_counter += 1
            # NoChange appears only in effects in the training data, so it
            # should never be in the preconditions of the learned operators.
            assert not any(lit.predicate.positive == self._NoChange
                           for lit in path)
            preconds = LiteralConjunction(path)
            effects = LiteralConjunction([effect_to_literal(l) for l in leaf])
            params = self._get_params_from_preconds(preconds)
            operator = Operator(name, params, preconds, effects)
            operators.add(operator)
        return operators

    def _get_params_from_preconds(self, preconds):
        param_set = set()
        for lit in preconds.literals:
            if lit.negated_as_failure:
                continue
            for v in lit.variables:
                param_set.add(v)
        return sorted(list(param_set))
