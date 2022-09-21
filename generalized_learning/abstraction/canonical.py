'''
Created on Jul 12, 2021

@author: anonymous
'''


import itertools
import pathlib

from generalized_learning.abstraction.role import AbstractRole
from generalized_learning.concretized import problem as problem_instance
from generalized_learning.concretized.domain import Domain
from generalized_learning.neural_net.nn import NNPkg
from generalized_learning.util import state_explorer
import numpy as np
from pddl import conditions
from pddl.pddl_types import TypedObject


class CanonicalAbstraction:

    _0ARY_OBJ = TypedObject("null-object", "0ary")

    def __init__(self, domain,
                 unsupported_predicates_set=set(
                     ["=",
                      problem_instance.Problem.DONE_ATOM.predicate]),
                 tag_goals=True, tag_types=True):

        self._domain = domain
        self._tag_goals = tag_goals
        self._tag_types = tag_types
        self._unsupported_predicates_set = unsupported_predicates_set
        self._problem = None

    def _compute_goals(self, problem):

        constant_goals_set = set()
        goal_done_dict = {}

        goal_literals = problem.get_goal_literals()
        for goal_literal in goal_literals:

            assert not goal_literal.negated

            name = "g_%s_%u" % (goal_literal.predicate,
                                len(goal_literal.args))

            goal_predicate = conditions.Atom(name, goal_literal.args)
            constant_goals_set.add(goal_predicate)

            done_predicate = conditions.Atom("d" + name, goal_literal.args)

            goal_done_dict[done_predicate] = set([goal_literal])

            for i in range(len(goal_literal.args)):

                name = "g_%s_%u_p%u" % (goal_literal.predicate,
                                        len(goal_literal.args),
                                        i)

                goal_predicate = conditions.Atom(name,
                                                 (goal_literal.args[i], ))
                constant_goals_set.add(goal_predicate)

                done_predicate = conditions.Atom("d" + name,
                                                 (goal_literal.args[i], ))

                goal_done_set = goal_done_dict.setdefault(done_predicate,
                                                          set())
                goal_done_set.add(goal_literal)

        return constant_goals_set, goal_done_dict

    def _compute_types(self, problem):

        auxiliary_constant_atoms = set()
        for typed_obj in problem.get_typed_objects():

            if typed_obj.type is not None:

                typed_atom = conditions.Atom(
                    "type_%s" % (typed_obj.type),
                    (typed_obj.name, ))
                auxiliary_constant_atoms.add(typed_atom)

        return auxiliary_constant_atoms

    def _cache_problem(self, problem):

        if id(self._problem) != id(problem):

            self._problem = problem
            self._type_atoms = self._compute_types(problem)
            self._goal_atoms, self._goal_done_dict = self._compute_goals(
                problem)

    def clear(self):

        self._problem = None

    def tag_types_if_necessary(self, state_atom_set):

        if self._tag_types:

            state_atom_set.update(self._type_atoms)

    def tag_goals_if_necessary(self, state_atom_set):

        if self._tag_goals:

            state_atom_set.update(self._goal_atoms)

            for done_atom in self._goal_done_dict:

                required_predicates = self._goal_done_dict[done_atom]

                if required_predicates.issubset(state_atom_set):

                    state_atom_set.add(done_atom)

    def create_abstract_state(self, problem, state):

        assert problem is not None
        self._cache_problem(problem)

        state_atom_set = set(state.get_atom_set())
        self.tag_types_if_necessary(state_atom_set)
        self.tag_goals_if_necessary(state_atom_set)

        return CanonicalAbstractionState(problem, state_atom_set,
                                         self._unsupported_predicates_set)


class CanonicalAbstractionMLP(CanonicalAbstraction):

    @staticmethod
    def _create_nary_pred_dict_key(predicate_name, roles):

        dict_key = (predicate_name, )
        for role in roles:

            dict_key += (role, )

        return dict_key

    @staticmethod
    def get_problem_filepath(domain_filepath):

        root = pathlib.Path(__file__).parent / "../../experiments/"

        if "gripper" in str(domain_filepath):

            domain_root = root / "gripper" / "l0" / "l1"
            domain = domain_root / "gripper.domain.pddl"
            problem = domain_root / "problem_0.problem.pddl"

        elif "blocksworld_on_all" in str(domain_filepath):

            domain_root = root / "blocksworld_on_all" / "l0" / "l1"
            domain = domain_root / "blocksworld.domain.pddl"
            problem = domain_root / "problem_0.problem.pddl"
        elif "miconic" in str(domain_filepath):

            domain_root = root / "miconic" / "l0" / "l2"
            domain = domain_root / "miconic.domain.pddl"
            problem = domain_root / "problem_0.problem.pddl"
        elif "visitall" in str(domain_filepath):

            domain_root = root / "visitall" / "l0" / "l1"
            domain = domain_root / "visitall.domain.pddl"
            problem = domain_root / "problem_0.problem.pddl"
        elif "sysadmin" in str(domain_filepath):

            domain_root = root / "sysadmin" / "l0" / "l0"
            domain = domain_root / "sysadmin_mdp.domain.pddl"
            problem = domain_root / "problem_0.problem.pddl"
        else:

            raise Exception("Unsupported domain.")

        return domain, [problem]

    @staticmethod
    def create(domain_filepath, problem_list, **kwargs):

        domain_filepath, problem_list = \
            CanonicalAbstractionMLP.get_problem_filepath(domain_filepath)

        #         print("Creating canonical abstraction function for %u problems" % (
        #             len(problem_list)))
        domain = Domain(domain_filepath)
        abstraction = CanonicalAbstraction(domain)

        observed_roles = set()
        observed_nary_preds = set()
        for problem_filepath in problem_list:

            problem = problem_instance.create_problem(domain_filepath.name,
                                                      problem_filepath.name,
                                                      problem_filepath.parent)

            states = state_explorer.random_walk(problem)
            for state in states:

                abstract_state = abstraction.create_abstract_state(problem,
                                                                   state)
                observed_roles.update(abstract_state.get_all_roles())
                observed_nary_preds.update(
                    [dict_key[0] for dict_key in abstract_state.get_all_nary_preds()])

                for dict_key in abstract_state.get_all_nary_preds():

                    domain.add_predicate_arity_entry(dict_key[0],
                                                     len(dict_key) - 1)

        abstraction = CanonicalAbstractionMLP(domain, observed_nary_preds,
                                              observed_roles)
        return abstraction

    def __init__(self, domain, observed_nary_preds, observed_roles,
                 allow_exceptions_in_encoding=True,
                 unsupported_predicates_set=set(["="])):

        super(CanonicalAbstractionMLP, self).__init__(
            domain,
            unsupported_predicates_set=unsupported_predicates_set)

        self._is_nn_initialized = False
        self._allow_exceptions_in_encoding = allow_exceptions_in_encoding

        self._role_index_map, self._index_role_map, \
            self._unary_preds_index_map, self._index_unary_preds_map = \
            self._generate_initial_role_index_map(observed_roles)

        self._action_index_map, self._index_action_map = \
            self._generate_action_maps(domain)

        self._feature_dict, self._index_feature_dict = \
            self._initialize_feature_vector_dict(observed_roles,
                                                 observed_nary_preds)

        self._max_action_params = self._domain.get_max_action_params()
        self._observed_nary_predicates = set(observed_nary_preds)
        self._encoding_funcs = {

            "features": self._encode_features,
            "action_params_features": self._encode_action_params_features,

            "role_count": self._encode_role_count,
            "role_count_bool": self._encode_role_count_bool,
            "action": self._encode_action,
            "action_params": self._encode_action_params,
            "q_s_a": self._encode_q_s_a,
            "role_count_nary": self._encode_role_count_nary,
            "role_count_nary_bool": self._encode_role_count_nary_bool,
        }

        self._nn_shape_dict = self._initialize_nn_shape_dict()

    def _initialize_feature_vector_dict(self, roles, nary_preds):

        feature_dict = {}
        index_feature_dict = {}

        assert isinstance(roles, set)
        assert isinstance(nary_preds, set)

        for role in roles:

            index_feature_dict[len(feature_dict)] = role
            feature_dict[role] = len(feature_dict)

        for predicate in nary_preds:

            arity = self._domain.predicate_arity_dict[predicate]
            assert arity > 1

            role_product = itertools.product(roles, repeat=arity)

            for role_p in role_product:
                dict_key = \
                    CanonicalAbstractionMLP._create_nary_pred_dict_key(
                        predicate, role_p)
                index_feature_dict[len(feature_dict)] = dict_key
                feature_dict[dict_key] = len(feature_dict)

        return feature_dict, index_feature_dict

    def _generate_action_maps(self, domain):

        action_index_map = {}
        index_action_map = {}

        for action in domain.get_action_templates():

            name = action.name

            if "_DETDUP" in name:
                name = name[: name.index("_DETDUP")]
            else:
                name = name

            action_index_map.setdefault(name, len(action_index_map))
            index_action_map.setdefault(len(action_index_map) - 1, name)

        return action_index_map, index_action_map

    def _generate_initial_role_index_map(self, roles):

        possible_roles = set([AbstractRole([])])
        possible_roles.update(roles)

        i = 0
        role_index_map = {}
        index_role_map = {}

        unary_preds_index_map = {}
        index_unary_preds_map = {}
        for possible_role in possible_roles:

            assert isinstance(possible_role, AbstractRole)
            role_index_map[possible_role] = i
            index_role_map[i] = possible_role
            i += 1

            unary_predicates = possible_role.get_unary_predicates()
            for unary_pred in unary_predicates:

                if unary_pred not in unary_preds_index_map:

                    index = len(unary_preds_index_map)
                    unary_preds_index_map[unary_pred] = index
                    index_unary_preds_map[index] = unary_pred

        return role_index_map, index_role_map, unary_preds_index_map, \
            index_unary_preds_map

    def _get_nary_shape(self):

        return (len(self._role_index_map), len(self._role_index_map))

    def _initialize_nn_shape_dict(self):

        nn_shape_dict = {

            "role_count": len(self._role_index_map),
            "role_count_bool": len(self._role_index_map),
            "action": len(self._action_index_map),
            "q_s_a": 1,
            "features": len(self._feature_dict),
        }

        for i in range(self._max_action_params):

            name = "action_param_%u" % (i)
            nn_shape_dict[name] = len(self._unary_preds_index_map)

            name = "action_param_features_%u" % (i)
            nn_shape_dict[name] = len(self._feature_dict)

        for predicate in self._observed_nary_predicates:

            name = "role_count_nary_%s" % (predicate)
            nn_shape_dict[name] = self._get_nary_shape()

            name = "role_count_nary_bool_%s" % (predicate)
            nn_shape_dict[name] = self._get_nary_shape()

        return nn_shape_dict

    def _initialize_nn_shapes(self, nn_names, layer_list, call_set):

        for nn_name in nn_names:

            if "features" == nn_name:

                call_set.add(nn_name)
                layer_list.append(nn_name)
            elif "role_count" == nn_name:

                call_set.add(nn_name)
                layer_list.append(nn_name)
            elif "role_count_bool" == nn_name:

                call_set.add(nn_name)
                layer_list.append(nn_name)
            elif "action" == nn_name:

                call_set.add("action")
                layer_list.append(nn_name)
            elif "action_params" == nn_name:

                call_set.add("action_params")

                for i in range(self._max_action_params):

                    name = "action_param_%u" % (i)
                    layer_list.append(name)
            elif "action_params_features" == nn_name:

                call_set.add(nn_name)

                for i in range(self._max_action_params):

                    name = "action_param_features_%u" % (i)
                    layer_list.append(name)
            elif "role_count_nary" == nn_name:

                call_set.add(nn_name)
                for predicate in self._observed_nary_predicates:

                    name = "%s_%s" % (nn_name, predicate)
                    layer_list.append(name)
            elif "role_count_nary_bool" == nn_name:

                call_set.add(nn_name)
                for predicate in self._observed_nary_predicates:

                    name = "%s_%s" % (nn_name, predicate)
                    layer_list.append(name)
            elif "q_s_a" == nn_name:

                call_set.add("q_s_a")
                layer_list.append(nn_name)
            else:

                raise Exception("Unsupported nn_shape: %s" % (nn_name))

        pass

    def initialize_nn(self,
                      nn_inputs=["role_count", "role_count_nary", "action",
                                 "action_params"],
                      nn_outputs=["q_s_a"]):

        final_inputs = []
        nn_input_call_set = set()
        self._initialize_nn_shapes(nn_inputs,
                                   final_inputs,
                                   nn_input_call_set)

        final_outputs = []
        nn_output_call_set = set()
        self._initialize_nn_shapes(nn_outputs,
                                   final_outputs,
                                   nn_output_call_set)

        return final_inputs, nn_input_call_set,\
            final_outputs, nn_output_call_set

    def get_nn_shape(self, name):

        return self._nn_shape_dict[name]

    def get_max_action_params(self):

        return self._max_action_params

    def get_role_index(self, role):

        try:

            return self._role_index_map[role]
        except KeyError:

            return float("inf")

    def get_unary_pred_index(self, unary_pred):

        try:

            return self._unary_preds_index_map[unary_pred]
        except KeyError:

            return float("inf")

    def get_index_unary_pred(self, index):

        return self._index_unary_preds_map[index]

    def get_action_index(self, action_name):

        try:

            return self._action_index_map[action_name]
        except KeyError:

            return float("inf")

    def get_index_action(self, action_index):

        return self._index_action_map[action_index]

    def _set_vector(self, vector, index, value):

        try:

            vector[index] = value
        except IndexError:

            assert self._allow_exceptions_in_encoding
            pass

    def _encode_features(self,  nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]

        vector = np.zeros(self._nn_shape_dict["features"])

        for role in abstract_state.get_all_roles():

            count = abstract_state.get_role_count(role)
            index = self._feature_dict.get(role, float("inf"))
            self._set_vector(vector, index, count)

        for dict_key in abstract_state.get_all_nary_preds():

            count = abstract_state.get_nary_count(dict_key)
            index = self._feature_dict.get(dict_key, float("inf"))
            self._set_vector(vector, index, count)

        nn_pkg.encode("features", vector)

    def _encode_action_param_features(self, vector, abstract_state, obj):

        role = abstract_state.get_obj_role(obj)
        index = self._feature_dict.get(role, float("inf"))
        self._set_vector(vector, index, 1)

        for dict_key in abstract_state.get_all_nary_preds():

            count = int(obj in abstract_state.get_nary_objects(dict_key))
            index = self._feature_dict.get(dict_key, float("inf"))
            self._set_vector(vector, index, count)

    def _encode_action_params_features(self, nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]
        action = kwargs["action"]

        param_list = action.get_param_list()
        for i in range(len(param_list)):

            name = "action_param_features_%u" % (i)
            obj = param_list[i]
            param_vector = np.zeros(self._nn_shape_dict[name])

            role = abstract_state.get_role(obj)
            index = self._feature_dict.get(role, float("inf"))
            self._set_vector(param_vector, index, 1)

            for dict_key in abstract_state.get_all_nary_preds():

                count = int(obj in abstract_state.get_nary_objects(dict_key))
                index = self._feature_dict.get(dict_key, float("inf"))
                self._set_vector(param_vector, index, count)

            nn_pkg.encode(name, param_vector)

        for i in range(len(param_list), self._max_action_params):

            name = "action_param_features_%u" % (i)
            param_vector = np.zeros(self._nn_shape_dict[name])
            nn_pkg.encode(name, param_vector)

    def _encode_state_vector(self, abstract_state, vector, encoding_func):

        for role in abstract_state.get_all_roles():

            index = self.get_role_index(role)
            role_count = abstract_state.get_role_count(role)

            self._set_vector(vector, index, encoding_func(role_count))

    def _encode_role_count(self, nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]

        vector = np.zeros(self._nn_shape_dict["role_count"])

        self._encode_state_vector(abstract_state, vector,
                                  lambda x: x)

        nn_pkg.encode("role_count", vector)

    def _encode_role_count_bool(self, nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]

        vector = np.zeros(self._nn_shape_dict["role_count_bool"])

        self._encode_state_vector(abstract_state, vector,
                                  lambda x: min(2, x))

        nn_pkg.encode("role_count_bool", vector)

    def _encode_nary_vector(self, abstract_state, vector_dict, encoding_func):

        for nary_tuple in abstract_state.get_all_nary_preds():

            name = nary_tuple[0]
            roles = nary_tuple[1:]

            index = ()
            possible_count = 1
            for role in roles:

                possible_count *= abstract_state.get_role_count(role)
                index += (self.get_role_index(role), )

            total_count = abstract_state.get_nary_count(nary_tuple)
            self._set_vector(vector_dict[name], index,
                             encoding_func(total_count, possible_count))

    def _encode_role_count_nary(self, nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]

        vector_dict = {}
        for predicate in self._observed_nary_predicates:

            vector_dict[predicate] = np.zeros(self._get_nary_shape())

        self._encode_nary_vector(abstract_state, vector_dict,
                                 lambda x, _: x)

        for predicate in self._observed_nary_predicates:

            name = "role_count_nary_%s" % (predicate)
            nn_pkg.encode(name, vector_dict[predicate])

        pass

    def _encode_role_count_nary_bool(self, nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]

        vector_dict = {}
        for predicate in self._observed_nary_predicates:

            vector_dict[predicate] = np.zeros(self._get_nary_shape())

        self._encode_nary_vector(abstract_state, vector_dict,
                                 lambda x, y: 1 if x == y else 0.5)

        for predicate in self._observed_nary_predicates:

            name = "role_count_nary_bool_%s" % (predicate)
            nn_pkg.encode(name, vector_dict[predicate])

    def _encode_action(self, nn_pkg, **kwargs):

        action = kwargs["action"]

        action_vector = np.zeros(self._nn_shape_dict["action"])
        index = self.get_action_index(action.get_name())
        self._set_vector(action_vector, index, 1)

        nn_pkg.encode("action", action_vector)

    def _encode_action_params(self, nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]
        action = kwargs["action"]

        param_list = action.get_param_list()
        for i in range(len(param_list)):

            name = "action_param_%u" % (i)
            param_vector = np.zeros(self._nn_shape_dict[name])

            obj = param_list[i]
            obj_role = abstract_state.get_role(obj)
            for unary_pred in obj_role.get_unary_predicates():

                index = self.get_unary_pred_index(unary_pred)
                self._set_vector(param_vector, index, 1)

            nn_pkg.encode(name, param_vector)

        for i in range(len(param_list), self._max_action_params):

            name = "action_param_%u" % (i)
            param_vector = np.zeros(self._nn_shape_dict[name])
            nn_pkg.encode(name, param_vector)

    def get_correct_param_count(self, abstract_state,
                                obj_name,
                                param_vector,
                                threshold=0.5):

        total_correct = 0

        if obj_name is None:

            unary_predicates = set()
        else:

            obj_role = abstract_state.get_role(obj_name)
            unary_predicates = obj_role.get_unary_predicates()

        for i in range(len(param_vector)):

            unary_predicate = self.get_index_unary_pred(i)

            if param_vector[i] > threshold:

                total_correct += unary_predicate in unary_predicates
            else:

                total_correct += unary_predicate not in unary_predicates

        return total_correct

    def _encode_q_s_a(self, nn_pkg, **kwargs):

        q_s_a = kwargs["q_s_a"]

        assert self._nn_shape_dict["q_s_a"] == 1
        vector = np.zeros(1)
        vector[0] = q_s_a

        nn_pkg.encode("q_s_a", vector)

    def encode_nn_input(self, nn_input_call_set, **kwargs):

        nn_pkg = NNPkg()

        for call_func in nn_input_call_set:

            self._encoding_funcs[call_func](nn_pkg,
                                            **kwargs)

        return nn_pkg

    def encode_nn_training_data(self, nn_input_call_set, nn_output_call_set,
                                **kwargs):

        nn_pkg = self.encode_nn_input(nn_input_call_set, **kwargs)

        for call_func in nn_output_call_set:

            self._encoding_funcs[call_func](nn_pkg,
                                            **kwargs)

        return nn_pkg


class CanonicalAbstractionState:

    def __init__(self, problem, state_atom_set, unsupported_predicates_set):

        obj_unary_preds_dict, higher_arity_preds_set = \
            self._process_predicates(state_atom_set,
                                     unsupported_predicates_set)

        self._obj_role_dict, self._role_obj_dict = self._update_object_roles(
            obj_unary_preds_dict,
            problem.get_typed_objects() + [CanonicalAbstraction._0ARY_OBJ])

        self._higher_arity_dict, self._higher_arity_obj_dict = \
            self._update_higher_arity_dicts(higher_arity_preds_set)

    def _process_predicates(self, state_atom_set, unsupported_predicates_set):

        obj_unary_preds_dict = {}
        higher_arity_preds_set = set()
        for atom in state_atom_set:

            if atom.predicate in unsupported_predicates_set:

                continue

            # 0-ary/unary predicates are abstraction predicates.
            if len(atom.args) < 2:
                if len(atom.args) == 0:

                    obj_name = CanonicalAbstraction._0ARY_OBJ.name
                else:

                    obj_name = atom.args[0]

                obj_unary_preds_set = obj_unary_preds_dict.setdefault(obj_name,
                                                                      set())
                obj_unary_preds_set.add(atom.predicate)
            else:

                higher_arity_preds_set.add(atom)

        return obj_unary_preds_dict, higher_arity_preds_set

    def _update_object_roles(self, obj_unary_preds_dict, objects):

        obj_role_dict = {}
        role_obj_dict = {}

        for obj in objects:

            obj_name = obj.name
            unary_preds = obj_unary_preds_dict.get(
                obj_name,
                set())
            obj_role = AbstractRole(unary_preds)

            obj_role_dict[obj_name] = obj_role
            role_obj_set = role_obj_dict.setdefault(obj_role, set())
            role_obj_set.add(obj_name)

        return obj_role_dict, role_obj_dict

    def _create_higher_arity_dict_key(self, predicate):

        assert len(predicate.args) > 1

        dict_key = (predicate.predicate, )
        for obj in predicate.args:

            dict_key += (self.get_role(obj), )

        return dict_key

    def _update_higher_arity_dicts(self, higher_arity_preds_set):

        higher_arity_dict = {}
        higher_arity_obj_dict = {}

        for predicate in higher_arity_preds_set:

            dict_key = self._create_higher_arity_dict_key(predicate)

            higher_arity_pred_set = higher_arity_dict.setdefault(dict_key,
                                                                 set())
            higher_arity_pred_set.add(predicate)

            higher_arity_obj_set = higher_arity_obj_dict.setdefault(dict_key,
                                                                    set())

            higher_arity_obj_set.update(predicate.args)

        return higher_arity_dict, higher_arity_obj_dict

    def get_role(self, obj_name):

        return self._obj_role_dict[obj_name]

    def get_role_count(self, role):

        return len(self._role_obj_dict[role])

    def get_nary_count(self, dict_key):

        return len(self._higher_arity_dict[dict_key])

    def get_nary_objects(self, dict_key):

        return self._higher_arity_obj_dict[dict_key]

    def get_all_roles(self):

        return self._role_obj_dict.keys()

    def get_all_nary_preds(self):

        return self._higher_arity_dict.keys()
