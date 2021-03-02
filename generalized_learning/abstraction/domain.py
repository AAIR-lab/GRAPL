"""
abstract_domain.py
===================
Represents stuff for abstract domains.
"""


import copy

from abstraction.role import AbstractRole
from neural_net.nn import NNPkg
import numpy as np


class AbstractDomain:

    @staticmethod
    def merge_abstract_domains(abstract_domains):

        roles_set = set()
        action_names_set = set()
        max_action_params = float("-inf")
        unary_predicates_set = set()

        dummy_abstract_domain = AbstractDomain()

        for abstract_domain in abstract_domains:

            dummy_abstract_domain._update_arity_dict(
                abstract_domain._arity_predicate_dict)
            roles_set.update(abstract_domain._role_index_map.keys())
            action_names_set.update(abstract_domain._action_index_map.keys())
            max_action_params = max(
                max_action_params, abstract_domain._max_action_params)
            unary_predicates_set.update(
                abstract_domain._action_param_index_map.keys())

        merged_abstract_domain = AbstractDomain(
            roles_set,
            action_names_set,
            max_action_params,
            unary_predicates_set,
            dummy_abstract_domain._arity_predicate_dict)

        return merged_abstract_domain

    def __init__(self, roles=set(),
                 action_names=set(),
                 max_action_params=float("-inf"),
                 unary_predicates=set(),
                 arity_predicate_dict={}):

        # Set the neural net as uninitialized.
        self._is_nn_initialized = False
        self._nn_role_index_map_len = float("-inf")
        self._nn_action_index_map_len = float("-inf")
        self._nn_action_param_index_map_len = float("-inf")

        self._arity_predicate_dict = copy.deepcopy(arity_predicate_dict)

        # Initialize the role -> index and index -> role maps.
        # This establishes the representation of the abstract state vector.
        self._role_index_map, self._index_role_map = self._generate_initial_role_index_map(
            roles)

        # Initialize the action -> index and index -> action maps.
        # This establishes the representation of the action vector.
        self._action_index_map, self._index_action_map = self._extract_action_index_map(
            action_names)
        self._max_action_params = max_action_params

        # Initialize the action_param -> index and index -> action_param maps.
        # This establishes the representation of the action-parameter vector.
        self._action_param_index_map, self._index_action_param_map = self._extract_action_param_index_map(
            unary_predicates)

    def _generate_initial_role_index_map(self, roles):

        possible_roles = set([AbstractRole([])])
        possible_roles.update(roles)

        i = 0
        role_index_map = {}
        index_role_map = {}
        for possible_role in possible_roles:

            assert isinstance(possible_role, AbstractRole)
            role_index_map[possible_role] = i
            index_role_map[i] = possible_role
            i += 1

        return role_index_map, index_role_map

    def _extract_action_index_map(self, action_names):

        i = 0
        action_index_map = {}
        index_action_map = {}
        for action_name in action_names:

            action_index_map[action_name] = i
            index_action_map[i] = action_name
            i += 1

        return action_index_map, index_action_map

    def _extract_action_param_index_map(self, unary_predicates):

        i = 0
        action_param_index_map = {}
        index_action_param_map = {}
        for unary_predicate in unary_predicates:

            action_param_index_map[unary_predicate] = i
            index_action_param_map[i] = unary_predicate
            i += 1

        return action_param_index_map, index_action_param_map

    def get_arities(self):

        return self._arity_predicate_dict.keys()

    def get_predicates(self, arity):

        return self._arity_predicate_dict[arity]

    def get_roles(self):

        return self._role_index_map.keys()

    def initialize_nn_parameters(self):

        if not self._is_nn_initialized:

            self._nn_role_index_map_len = len(self._role_index_map)
            self._nn_action_index_map_len = len(self._action_index_map)
            self._nn_action_param_index_map_len = len(
                self._action_param_index_map)

            self._nn_input_shape_dict = {}
            self._nn_input_shape_dict["state_unary_preds"] = (
                1, self._nn_role_index_map_len)
            self._nn_input_shape_dict["role_counts"] = (
                1, self._nn_role_index_map_len)

            for arity in self._arity_predicate_dict:

                for predicate in self._arity_predicate_dict[arity]:

                    pred_shape = (self._nn_role_index_map_len, ) * arity
                    self._nn_input_shape_dict[predicate] = pred_shape

                    rc_name = "role_count_%s" % (predicate)
                    self._nn_input_shape_dict[rc_name] = pred_shape

            self._nn_output_shape_dict = {}

            # Use np.prod() to convert (1, x) shapes to just (x, ).
            # This makes it compatible to use as outpupts in the neural
            # network.
            self._nn_output_shape_dict["action"] = np.prod(
                (1, self._nn_action_index_map_len))

            self._nn_output_shape_dict["plan_length"] = 1

            for i in range(self._max_action_params):

                self._nn_output_shape_dict["action_param_%u_preds" % (i)] = \
                    np.prod((1, self._nn_action_param_index_map_len))

            self._is_nn_initialized = True

    def get_action_index(self, action_name):

        try:

            index = self._action_index_map[action_name]
        except KeyError:

            if self._is_nn_initialized:

                index = float("inf")
            else:

                index = len(self._action_index_map)
                self._action_index_map[action_name] = index
                self._index_action_map[index] = action_name

        return index

    def get_index_action(self, index):

        return self._index_action_map[index]

    def get_action_param_index(self, unary_predicate):

        try:

            index = self._action_param_index_map[unary_predicate]
        except KeyError:

            if self._is_nn_initialized:

                index = float("inf")
            else:

                index = len(self._action_param_index_map)
                self._action_param_index_map[unary_predicate] = index
                self._index_action_param_map[index] = unary_predicate

        return index

    def get_index_action_param(self, index):

        return self._index_action_param_map[index]

    def get_role_index(self, role):

        try:

            index = self._role_index_map[role]
        except KeyError:

            if self._is_nn_initialized:

                index = float("inf")
            else:

                index = len(self._role_index_map)
                self._role_index_map[role] = index
                self._index_role_map[index] = role

        return index

    def get_index_role(self, index):

        return self._index_role_map[index]

    def _encode_nary_predicate(self, abstract_state, predicate, arity, strategy):

        index_role_dict = {}

        try:

            roles_dict = abstract_state.get_nary_role_dict(predicate)
            for role_tuple in roles_dict:

                index_t = ()
                for role in role_tuple:

                    index_t += (self.get_role_index(role), )

                index_role_dict[index_t] = role_tuple

        except KeyError:

            pass

        if self._is_nn_initialized:

            vector_shape = (self._nn_role_index_map_len, ) * arity
        else:

            vector_shape = (len(self._role_index_map), ) * arity

        vector = np.zeros(vector_shape)
        for index_tuple in index_role_dict.keys():

            role_tuple = index_role_dict[index_tuple]
            try:

                vector[index_tuple] = \
                    abstract_state.get_n_ary_role_count(predicate,
                                                        role_tuple,
                                                        strategy)
            except IndexError:

                # Ignore all the out-of-bounds errors. These will only come
                # after the nn has been initialized.
                assert self._is_nn_initialized
                pass

        return vector

    def _encode_state(self, abstract_state):

        index_role_dict = {}

        for role in abstract_state.get_roles():

            index = self.get_role_index(role)

            # Store the entry. We post-process them once all indexes have been discovered.
            # We can't do it here since the size of the array may change.
            assert index not in index_role_dict or (
                self._is_nn_initialized and index == float("inf"))
            index_role_dict[index] = role

        # Determine the shape of the output based on whether the neural network
        # has been initialized or not.
        abstract_state_vector_shape = (1, self._nn_role_index_map_len) if self._is_nn_initialized \
            else (1, len(self._role_index_map))

        # Finally, populate the vector.
        abstract_state_vector = np.zeros(
            abstract_state_vector_shape, np.uint32)

        role_count_vector = np.zeros(abstract_state_vector_shape, np.uint32)

        for index in index_role_dict.keys():

            role = index_role_dict[index]

            try:

                abstract_state_vector[0][index] = min(
                    2, abstract_state.get_role_count(role))

                role_count_vector[0][index] = abstract_state.get_role_count(
                    role)
            except IndexError:

                # Ignore all the out-of-bounds errors. These will only come
                # after the nn has been initialized.
                assert self._is_nn_initialized
                pass

        return abstract_state_vector, role_count_vector

    def _encode_action(self, action):

        index = self.get_action_index(action.get_name())

        action_vector_shape = (1, self._nn_action_index_map_len) if self._is_nn_initialized \
            else (1, len(self._action_index_map))
        action_vector = np.zeros(action_vector_shape)

        try:

            action_vector[0][index] = 1
        except IndexError:

            assert self._is_nn_initialized
            pass

        return action_vector

    def _encode_action_param(self, abstract_state, action):

        action_param_list = action.get_param_list()
        index_param_tuple_list = []

        for i in range(len(action_param_list)):

            action_param = action_param_list[i]
            role = abstract_state.get_role(action_param)

            for unary_predicate in role.get_unary_predicates():

                index = self.get_action_param_index(unary_predicate)
                index_param_tuple_list.append((i, index))

        if self._is_nn_initialized:

            action_param_vector_shape = (
                self._max_action_params, 1, self._nn_action_param_index_map_len)
        else:

            self._max_action_params = max(
                self._max_action_params, len(action_param_list))
            action_param_vector_shape = (
                self._max_action_params, 1, len(self._action_param_index_map))

        # Finally, populate the vector.
        action_param_vector = np.zeros(action_param_vector_shape, np.uint32)
        for index_param_tuple in index_param_tuple_list:

            action_param_index = index_param_tuple[0]
            unary_predicate_index = index_param_tuple[1]

            try:

                action_param_vector[action_param_index][0][unary_predicate_index] = 1
            except IndexError:

                # Ignore all the out-of-bounds errors. These will only come
                # after the nn has been initialized.
                assert self._is_nn_initialized
                pass

        # TODO: Can this code be removed?
        # For an action with fewer parameters, we just fill the rest of the vector with dummy data.
        # Simply use a one hot format.
#         for i in range(len(action_param_list), self._max_action_params):
#
#             action_param_vector[i][0][0] = 0

        return action_param_vector

    def _update_arity_dict(self, arity_predicate_dict):

        for arity in arity_predicate_dict:

            try:

                self._arity_predicate_dict[arity].update(
                    arity_predicate_dict[arity])
            except KeyError:

                self._arity_predicate_dict[arity] = set(
                    arity_predicate_dict[arity])

    def encode_nn_input(self, abstract_state, *args):

        (args)
        nn_input_pkg = NNPkg()

        abstract_state_vector, role_count_vector = self._encode_state(
            abstract_state)
        nn_input_pkg.encode("state_unary_preds", abstract_state_vector)
        nn_input_pkg.encode("role_counts", role_count_vector)

        if not self._is_nn_initialized:

            self._update_arity_dict(abstract_state.get_arity_predicate_dict())

        for arity in self._arity_predicate_dict:

            for predicate in self._arity_predicate_dict[arity]:

                vector = self._encode_nary_predicate(abstract_state,
                                                     predicate,
                                                     arity,
                                                     "tvla")
                nn_input_pkg.encode("%s" % (predicate), vector)

                vector = self._encode_nary_predicate(abstract_state,
                                                     predicate,
                                                     arity,
                                                     "raw")
                nn_input_pkg.encode("role_count_%s" % (predicate), vector)

        return nn_input_pkg

    def decode_nn_output(self, nn_output_pkg, abstract_state, action):

        assert self._is_nn_initialized

        action_vector = nn_output_pkg.decode("action")
        action_vector = action_vector.reshape(self._nn_action_index_map_len)

        param_vectors = []
        for i in range(self._max_action_params):

            param_vector_name = "action_param_%u_preds" % (i)
            param_vector = nn_output_pkg.decode(param_vector_name)
            param_vector = param_vector.reshape(
                self._nn_action_param_index_map_len)
            param_vectors.append(param_vector)

        action_value = self._get_action_value(abstract_state, action,
                                              action_vector, param_vectors)

        return action_value

    def get_max_action_params(self):

        return self._max_action_params

    def get_nn_input_shape_dict(self):

        return self._nn_input_shape_dict

    def get_nn_input_shape(self, input_name):

        return self._nn_input_shape_dict[input_name]

    def get_nn_output_shape_dict(self):

        return self._nn_output_shape_dict

    def get_nn_output_shape(self, output_name):

        return self._nn_output_shape_dict[output_name]

    def get_nn_input_dict(self, nn_pkg_list):

        nn_input_dict = {}

        for nn_input in self.get_nn_input_shape_dict().keys():

            data = map(lambda nn_pkg: nn_pkg.decode(nn_input), nn_pkg_list)
            data = np.asarray(list(data))

            nn_input_dict[nn_input] = data

        return nn_input_dict

    def get_nn_train_dict(self, nn_train_pkg_list):

        nn_input_dict = {}

        for nn_input in self.get_nn_input_shape_dict().keys():

            data = map(lambda nn_train_pkg: nn_train_pkg.decode(
                nn_input), nn_train_pkg_list)
            data = np.asarray(list(data))

            nn_input_dict[nn_input] = data

        nn_label_dict = {}
        for nn_label in self.get_nn_output_shape_dict().keys():

            data = map(lambda nn_train_pkg: nn_train_pkg.decode(
                nn_label), nn_train_pkg_list)
            data = np.asarray(list(data))

            nn_label_dict[nn_label] = data

        return nn_input_dict, nn_label_dict

    def encode_nn_training_data(self, abstract_state, action_taken,
                                plan_length, *args):

        (args)

        nn_training_pkg = self.encode_nn_input(
            abstract_state, action_taken, args)

        nn_training_pkg.encode("action", self._encode_action(action_taken))

        plan_length_vector = np.zeros((1, 1))
        plan_length_vector[0][0] = plan_length
        nn_training_pkg.encode("plan_length", plan_length_vector)

        action_param_vector = self._encode_action_param(
            abstract_state, action_taken)
        for i in range(self._max_action_params):

            nn_training_pkg.encode("action_param_%u_preds" %
                                   (i), action_param_vector[i])

        return nn_training_pkg

    def _get_index_tuples(self, index, arity, length, index_tuple, t_list):

        if index == arity:

            t_list.append(index_tuple)
        else:

            for i in range(length):

                self._get_index_tuples(index + 1, arity, length, index_tuple + (i, ),
                                       t_list)

    def _fix_nary_predicate(self, old_domain, old_pkg, name, arity, new_pkg):

        vector = np.zeros(self._nn_input_shape_dict[name])

        try:

            old_vector = old_pkg.decode(name)
            t_list = []
            self._get_index_tuples(0,
                                   arity,
                                   len(old_vector),
                                   (),
                                   t_list)
            for old_index_t in t_list:

                if old_vector[old_index_t] > 0:

                    new_index_t = ()
                    for i in old_index_t:

                        old_role = old_domain.get_index_role(i)
                        new_index = self.get_role_index(old_role)

                        new_index_t += (new_index, )

                    vector[new_index_t] = old_vector[old_index_t]
        except KeyError:

            pass

        new_pkg.encode(name, vector)

    def _fix_nn_state(self, old_domain, old_nn_train_pkg, fixed_nn_train_pkg):

        old_state_vector = old_nn_train_pkg.decode("state_unary_preds")
        old_role_counts = old_nn_train_pkg.decode("role_counts")

        new_state_vector = np.zeros(
            self._nn_input_shape_dict["state_unary_preds"])
        new_role_counts = np.zeros(
            self._nn_input_shape_dict["role_counts"])

        for i in range(len(old_state_vector[0])):

            if old_state_vector[0][i] > 0:

                old_role = old_domain.get_index_role(i)
                new_index = self.get_role_index(old_role)

                new_state_vector[0][new_index] = old_state_vector[0][i]
                new_role_counts[0][new_index] = old_role_counts[0][i]

        fixed_nn_train_pkg.encode("state_unary_preds", new_state_vector)
        fixed_nn_train_pkg.encode("role_counts", new_role_counts)

        for arity in self._arity_predicate_dict:

            for predicate in self._arity_predicate_dict[arity]:

                self._fix_nary_predicate(old_domain,
                                         old_nn_train_pkg,
                                         "%s" % (predicate),
                                         arity,
                                         fixed_nn_train_pkg)

                self._fix_nary_predicate(old_domain,
                                         old_nn_train_pkg,
                                         "role_count_%s" % (predicate),
                                         arity,
                                         fixed_nn_train_pkg)

    def _fix_nn_action(self, old_domain, old_nn_train_pkg, fixed_nn_train_pkg):

        old_action_vector = old_nn_train_pkg.decode("action")
        new_action_vector = np.zeros((1, self._nn_output_shape_dict["action"]))

        for i in range(len(old_action_vector[0])):

            if old_action_vector[0][i] > 0:

                action_name = old_domain.get_index_action(i)
                new_index = self.get_action_index(action_name)
                new_action_vector[0][new_index] = old_action_vector[0][i]

        fixed_nn_train_pkg.encode("action", new_action_vector)

    def _fix_nn_action_param(self, old_domain, old_nn_train_pkg, fixed_nn_train_pkg):

        for i in range(self._max_action_params):

            action_param_name = "action_param_%u_preds" % (i)

            new_action_param_vector = np.zeros(
                (1, self._nn_output_shape_dict[action_param_name]))

            try:

                old_action_param_vector = old_nn_train_pkg.decode(
                    action_param_name)

                for j in range(len(old_action_param_vector[0])):

                    if old_action_param_vector[0][j] > 0:

                        unary_predicate = old_domain.get_index_action_param(j)
                        new_index = self.get_action_param_index(
                            unary_predicate)

                        new_action_param_vector[0][new_index] = old_action_param_vector[0][j]
            except KeyError:

                pass

            fixed_nn_train_pkg.encode(
                action_param_name, new_action_param_vector)

    def fix_nn_training_data(self, old_domain, old_nn_train_pkgs):

        assert self._is_nn_initialized

        fixed_nn_train_pkgs = []
        for old_nn_train_pkg in old_nn_train_pkgs:

            fixed_nn_train_pkg = NNPkg()

            fixed_nn_train_pkg.encode("plan_length",
                                      old_nn_train_pkg.decode("plan_length"))
            self._fix_nn_state(old_domain, old_nn_train_pkg,
                               fixed_nn_train_pkg)
            self._fix_nn_action(
                old_domain, old_nn_train_pkg, fixed_nn_train_pkg)
            self._fix_nn_action_param(
                old_domain, old_nn_train_pkg, fixed_nn_train_pkg)

            fixed_nn_train_pkgs.append(fixed_nn_train_pkg)

        return fixed_nn_train_pkgs
