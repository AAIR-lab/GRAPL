'''
Created on Jul 26, 2021

@author: anonymous
'''

import pathlib
import sys

from tarski.dl.features import MinDistanceFeature
from tarski.dl.features import NullaryAtomFeature

from generalized_learning.concretized import problem as problem_instance
from generalized_learning.concretized.domain import Domain
from generalized_learning.neural_net.nn import NNPkg
from generalized_learning.util import constants
import numpy as np
from sltp import features
from sltp import language
from sltp import models
from sltp.util import tools


class DescriptionLogic:

    @staticmethod
    def _d2l_no_goal_param_generator(_):

        return []

    @staticmethod
    def get_feature_filepath(feature_file):

        if feature_file in ["gripper_sat_features.io",
                            "gripper_full_features.io"]:

            return constants.BENCHMARKS_DIR / "gripper" / feature_file
        elif feature_file in ["miconic_sat_features.io",
                              "miconic_full_features.io"]:

            return constants.BENCHMARKS_DIR / "miconic" / feature_file
        elif feature_file in ["on_a_b_sat_features.io",
                              "on_a_b_full_features.io",
                              "on_all_full_features.io"]:

            return constants.BENCHMARKS_DIR / "blocksworld" / feature_file
        elif feature_file in ["visitall_sat_features.io",
                              "visitall_full_features.io"]:

            return constants.BENCHMARKS_DIR / "visitall" / feature_file
        elif feature_file in ["hallway_sat_features.io",
                              "hallway_full_features.io"]:

            return constants.BENCHMARKS_DIR / "hallway" / feature_file
        elif feature_file in ["academic_advising_full_features.io"]:

            return constants.BENCHMARKS_DIR / "academic_advising" / feature_file
        elif feature_file in ["sysadmin_full_features.io"]:

            return constants.BENCHMARKS_DIR / "sysadmin" / feature_file
        elif feature_file in ["game_of_life_full_features.io"]:

            return constants.BENCHMARKS_DIR / "game_of_life" / feature_file
        elif feature_file in ["navigation_full_features.io"]:

            return constants.BENCHMARKS_DIR / "navigation" / feature_file
        elif feature_file in ["crossing_traffic_full_features.io"]:

            return constants.BENCHMARKS_DIR / "crossing_traffic" / feature_file
        elif feature_file in ["triangle_tireworld_full_features.io"]:

            return constants.BENCHMARKS_DIR / "triangle_tireworld" / feature_file
        elif feature_file in ["wildfire_full_features.io"]:

            return constants.BENCHMARKS_DIR / "wildfire" / feature_file
        else:

            # The feature file must be an absolute path.
            assert False
            return feature_file

    def __init__(self, domain_filepath, problem_list,
                 unsupported_predicates_set=set(
                     ["=",
                      problem_instance.Problem.DONE_ATOM.predicate]),
                 **kwargs):

        self._domain = Domain(domain_filepath)

        feature_filepath = DescriptionLogic.get_feature_filepath(
            kwargs["feature_file"])

        _, pddl_language, _ = language.parse_pddl(self._domain.get_filepath())
        self._features = tools.unserialize_features(pddl_language,
                                                    feature_filepath,
                                                    None)
        self._unsupported_predicates = unsupported_predicates_set
        self._problem = None

    def _cache_problem(self, problem):

        if id(self._problem) != id(problem):

            goal_generator = None
            if not problem.has_goals():

                goal_generator = DescriptionLogic._d2l_no_goal_param_generator

            self._problem = problem
            dl_problem, self._model_factory = features.create_model_factory(
                problem.get_domain_filepath(),
                problem.get_problem_filepath(),
                goal_generator)

            self._static_atoms, _ = features.compute_static_atoms(dl_problem)

    def clear(self):

        self._problem = None
        self._static_atoms = None
        self._model_factory = None

    def _translate_state(self, state, static_atoms):

        translated_state = list(static_atoms)

        for atom in state.get_atom_set():

            if atom.predicate not in self._unsupported_predicates:

                translated_atom = [atom.predicate]
                translated_atom += atom.args
                translated_state.append(translated_atom)

        return translated_state

    def create_abstract_state(self, problem, state):

        assert problem is not None
        self._cache_problem(problem)

        translated_state = self._translate_state(state, self._static_atoms)
        dl_model = models.FeatureModel(
            self._model_factory.create_model(translated_state))

        return DescriptionLogicAbstractState(dl_model)


class DescriptionLogicMLP(DescriptionLogic):

    @staticmethod
    def create(domain_filepath, problem_list, **kwargs):

        return DescriptionLogicMLP(domain_filepath, problem_list, **kwargs)

    def __init__(self, domain_filepath, problem_list, **kwargs):

        super(DescriptionLogicMLP, self).__init__(domain_filepath,
                                                  problem_list,
                                                  **kwargs)

        self._action_index_map, self._index_action_map = \
            self._generate_action_maps(self._domain)

        self._max_action_params = self._domain.get_max_action_params()

        self._encoding_funcs = {

            "state": self._encode_state,
            "state_bool": self._encode_state_bool,
            "action": self._encode_action,
            "action_params": self._encode_action_params,
            "q_s_a": self._encode_q_s_a,
        }

        self._nn_shape_dict = self._initialize_nn_shape_dict()

    def initialize_nn(self, nn_inputs=["state", "action", "action_params"],
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

    def _initialize_nn_shape_dict(self):

        nn_shape_dict = {

            "state": len(self._features),
            "state_bool": len(self._features),
            "action": len(self._action_index_map),
            "q_s_a": 1,
        }

        for i in range(self._max_action_params):

            name = "action_param_%u" % (i)
            nn_shape_dict[name] = len(self._features)

        return nn_shape_dict

    def _initialize_nn_shapes(self, nn_names, layer_list, call_set):

        for nn_name in nn_names:

            if "state" == nn_name:

                call_set.add(nn_name)
                layer_list.append(nn_name)
            elif "state_bool" == nn_name:

                call_set.add(nn_name)
                layer_list.append(nn_name)
            elif "action" == nn_name:

                call_set.add(nn_name)
                layer_list.append(nn_name)
            elif "action_params" == nn_name:

                call_set.add("action_params")

                for i in range(self._max_action_params):

                    name = "action_param_%u" % (i)
                    layer_list.append(name)
            elif "q_s_a" == nn_name:

                call_set.add(nn_name)
                layer_list.append(nn_name)
            else:

                raise Exception("Unsupported nn_shape: %s" % (nn_name))

        pass

    def get_nn_shape(self, name):

        return self._nn_shape_dict[name]

    def get_max_action_params(self):

        return self._max_action_params

    def _encode_state_vector(self, abstract_state, vector, encoding_func):

        dl_model = abstract_state.get_dl_model()

        assert len(vector) == len(self._features)

        for i in range(len(self._features)):

            feature = self._features[i]
            value = dl_model.denotation(feature)

            if value == sys.maxsize:

                value = 0

            vector[i] = encoding_func(value)

    def _encode_state(self, nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]

        abs_vector = np.zeros(self._nn_shape_dict["state"])

        self._encode_state_vector(abstract_state, abs_vector,
                                  lambda x: x)

        nn_pkg.encode("state", abs_vector)

    def _encode_state_bool(self, nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]

        abs_vector = np.zeros(self._nn_shape_dict["state_bool"])

        self._encode_state_vector(abstract_state, abs_vector,
                                  lambda x: x > 0)

        nn_pkg.encode("state_bool", abs_vector)

    def get_action_index(self, action_name):

        try:

            return self._action_index_map[action_name]
        except KeyError:

            return float("inf")

    def get_index_action(self, action_index):

        return self._index_action_map[action_index]

    def get_correct_param_count(self, abstract_state,
                                obj_name,
                                param_vector,
                                threshold=0.5):

        assert len(param_vector) == len(self._features)
        total_correct = 0
        dl_model = abstract_state.get_dl_model()

        if obj_name is None:

            obj_idx = float("inf")
        else:
            obj_idx = dl_model.concept_model.universe_idx._index[obj_name]

        for i in range(len(self._features)):

            feature = self._features[i]
            if obj_name is None \
                or isinstance(feature,
                              (NullaryAtomFeature, MinDistanceFeature)):

                total_correct += param_vector[i] < threshold
            elif obj_idx in dl_model.concept_model.uncompressed_denotation(
                    feature.c):

                total_correct += param_vector[i] >= threshold
            else:

                total_correct += param_vector[i] < threshold

        return total_correct

    def _encode_action(self, nn_pkg, **kwargs):

        action = kwargs["action"]

        action_vector = np.zeros(self._nn_shape_dict["action"])
        index = self.get_action_index(action.get_name())
        action_vector[index] = 1

        nn_pkg.encode("action", action_vector)

    def _encode_action_params(self, nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]
        dl_model = abstract_state.get_dl_model()

        action = kwargs["action"]

        param_vectors = np.zeros((self._max_action_params,
                                  len(self._features)))

        param_list = action.get_param_list()
        for i in range(len(param_list)):

            obj = param_list[i]
            obj_idx = dl_model.concept_model.universe_idx._index[obj]

            for j in range(len(self._features)):

                feature = self._features[j]

                if isinstance(feature, (NullaryAtomFeature, MinDistanceFeature)):

                    continue
                objs_in_feature = dl_model.concept_model.uncompressed_denotation(
                    feature.c)
                param_vectors[i][j] = int(obj_idx in objs_in_feature)

        for i in range(self._max_action_params):

            name = "action_param_%u" % (i)
            nn_pkg.encode(name, param_vectors[i])

    def _encode_q_s_a(self, nn_pkg, **kwargs):

        q_s_a = kwargs["q_s_a"]

        assert self._nn_shape_dict["q_s_a"] == 1
        vector = np.zeros(1)
        vector[0] = q_s_a

        nn_pkg.encode("q_s_a", vector)

    def encode_nn_input(self, nn_input_call_set, **kwargs):

        nn_pkg = NNPkg()

        for call_func in nn_input_call_set:

            self._encoding_funcs[call_func](nn_pkg, **kwargs)

        return nn_pkg

    def encode_nn_training_data(self, nn_input_call_set, nn_output_call_set,
                                **kwargs):

        nn_pkg = self.encode_nn_input(nn_input_call_set, **kwargs)

        for call_func in nn_output_call_set:

            self._encoding_funcs[call_func](nn_pkg, **kwargs)

        return nn_pkg


class DescriptionLogicAbstractState:

    def __init__(self, dl_model):

        self._dl_model = dl_model

    def get_dl_model(self):

        return self._dl_model
