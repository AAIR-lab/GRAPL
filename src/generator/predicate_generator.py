
import random
from pddlgym import structs

class PredicateGenerator:

    def __init__(self, types, predicates_r, predicate_arity_r):

        self.types = types
        self.predicates_r = predicates_r
        self.predicate_arity_r = predicate_arity_r

        self.types_list = [self.types[type_name] for type_name in self.types]

    def generate_name(self, idx):

        return "pred%u-p" % (idx)

    def generate_var_types(self, arity):

        return [random.choice(self.types_list) for _ in range(arity)]

    def generate_predicates(self):

        predicates = {}
        num_predicates = random.randint(*self.predicates_r)

        for idx in range(num_predicates):

            pred_name = self.generate_name(idx)
            arity = random.randint(*self.predicate_arity_r)
            var_types = self.generate_var_types(arity)

            predicates[pred_name] = structs.Predicate(pred_name, arity,
                                                      var_types)

        return predicates