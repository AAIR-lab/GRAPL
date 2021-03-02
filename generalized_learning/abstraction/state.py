

from abstraction.role import AbstractRole
from concretized.state import State


class AbstractState:

    _0ARY_PHANTOM_OBJ_NAME = "null-object"

    def __init__(self, problem, concrete_state):

        self._object_role_dict = {}
        self._role_object_dict = {}

        self._arity_predicate_dict = {}
        self._predicate_arity_dict = {}

        self._predicate_roles_dict = {}

        # Now update the roles of the objects in the provided concrete state.
        self.compute(concrete_state, problem)

    def get_roles(self):

        return self._role_object_dict.keys()

    def get_role_count(self, role):

        try:

            return len(self._role_object_dict[role])
        except KeyError:

            return 0

    def get_objects(self, role):

        return self._role_object_dict[role]

    def get_role(self, obj):

        return self._object_role_dict[obj]

    def get_binary_role_predicates(self):

        return self._binary_pred_role_count_dict.keys()

    def get_arity_predicate_dict(self):

        return self._arity_predicate_dict

    def compute_n_ary_role_count(self, predicate):

        name = predicate.predicate
        arity = len(predicate.args)
        assert arity > 1

        assert name not in self._predicate_arity_dict \
            or name in self._arity_predicate_dict[arity]

        self._predicate_arity_dict[name] = arity

        try:

            self._arity_predicate_dict[arity].add(name)
        except KeyError:

            self._arity_predicate_dict[arity] = set([name])

        predicate_roles_dict = {}
        try:

            predicate_roles_dict = self._predicate_roles_dict[name]
        except KeyError:

            self._predicate_roles_dict[name] = predicate_roles_dict

        role_tuple = ()
        for i in range(arity):

            role = self.get_role(predicate.args[i])
            role_tuple += (role, )

        try:

            predicate_roles_dict[role_tuple] = \
                predicate_roles_dict[role_tuple] + 1
        except KeyError:

            predicate_roles_dict[role_tuple] = 1

    def get_nary_role_dict(self, predicate):

        return self._predicate_roles_dict[predicate]

    def get_n_ary_role_count(self, predicate, role_tuple, strategy="raw"):

        name = predicate

        try:

            role_product = 1
            for role in role_tuple:

                role_product *= self.get_role_count(role)

            predicate_roles_dict = self._predicate_roles_dict[name]
            assert len(role_tuple) == self._predicate_arity_dict[name]

            total_count = predicate_roles_dict[role_tuple]
        except KeyError:

            total_count = 0

        if strategy == "raw":

            return total_count
        elif strategy == "tvla":

            if total_count == role_product:

                return 1
            elif total_count == 0:

                return 0
            else:

                return 0.5
        else:

            raise Exception("Unknown strategy")

    def update_object_role(self, obj, role):

        assert obj not in self._object_role_dict

        # Update the new role of the object.
        self._object_role_dict[obj] = role

        try:

            self._role_object_dict[role].add(obj)
        except KeyError:

            # New role, creating a new set.
            self._role_object_dict[role] = set([obj])

    def compute_roles(self, atom_dict):

        object_unary_pred_dict = {}
        for atom in atom_dict:

            assert len(atom.args) < 2
            if len(atom.args) == 0:

                obj = AbstractState._0ARY_PHANTOM_OBJ_NAME
            else:

                obj = atom.args[0]

            try:
                unary_preds_set = object_unary_pred_dict[obj]
            except KeyError:

                unary_preds_set = set()
                object_unary_pred_dict[obj] = unary_preds_set

            unary_preds_set.add(atom.predicate)

        for obj in object_unary_pred_dict.keys():

            role = AbstractRole(object_unary_pred_dict[obj])
            self.update_object_role(obj, role)

    def compute(self, state, problem):

        # Get all n-ary predicates required.
        arity_atom_dict = state.get_arity_atom_set_dict()

        # Compute the roles for 0-ary and 1-ary predicates.
        if 0 in arity_atom_dict:

            self.compute_roles(arity_atom_dict[0])

        if 1 in arity_atom_dict:

            self.compute_roles(arity_atom_dict[1])

        # Assign all other objects to the empty role.
        empty_role = AbstractRole([])

        for typed_obj in problem.get_typed_objects():

            obj = typed_obj.name
            if obj not in self._object_role_dict:

                self.update_object_role(obj, empty_role)

        if AbstractState._0ARY_PHANTOM_OBJ_NAME not in self._object_role_dict:

            self.update_object_role(AbstractState._0ARY_PHANTOM_OBJ_NAME,
                                    empty_role)

        for arity in arity_atom_dict.keys():

            if arity >= 2:

                for atom in arity_atom_dict[arity]:

                    self.compute_n_ary_role_count(atom)
