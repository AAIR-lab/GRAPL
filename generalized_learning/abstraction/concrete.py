

from generalized_learning.concretized.domain import Domain
from generalized_learning.concretized.problem import Problem
from generalized_learning.neural_net.nn import NNPkg
import numpy as np


class ConcreteMLP:

    @staticmethod
    def create(domain_filepath, problem_list, **kwargs):

        return ConcreteMLP(domain_filepath, problem_list, **kwargs)

    @staticmethod
    def _extract_atoms_and_actions(domain_filepath, problem_filepath):

        problem = Problem(domain_filepath.name, problem_filepath.name,
                          problem_filepath.parent)

        atom_set = set(problem._task.init)
        atom_set.update(problem._atoms)

        action_names = set()
        stochastic_action_dict = problem.get_stochastic_actions_dict()
        for action_name in stochastic_action_dict:

            action_names.add(action_name)

            stochastic_action = stochastic_action_dict[action_name]
            atom_set.update(stochastic_action.get_all_atoms())

        return atom_set, action_names

    def __init__(self, domain_filepath, problem_list, **kwargs):

        self._domain = Domain(domain_filepath)
        self._unsupported_predicates = set(["="])

        self._atom_index_dict = {}
        self._index_atom_dict = {}

        self._action_index_dict = {}
        self._index_action_dict = {}

        for problem_filepath in problem_list:

            atom_set, action_names = ConcreteMLP._extract_atoms_and_actions(
                domain_filepath,
                problem_filepath)

            for atom in atom_set:

                if atom.predicate not in self._unsupported_predicates:

                    self._atom_index_dict.setdefault(
                        atom,
                        len(self._atom_index_dict))
                    self._index_atom_dict.setdefault(
                        len(self._atom_index_dict) - 1,
                        atom)

            for action_name in action_names:

                self._action_index_dict.setdefault(
                    action_name,
                    len(self._action_index_dict))

                self._index_action_dict.setdefault(
                    len(self._action_index_dict) - 1,
                    action_name)

        self._encoding_funcs = {

            "state": self._encode_state,
            "action": self._encode_action,
            "q_s_a": self._encode_q_s_a,
        }

        self._nn_shape_dict = self._initialize_nn_shape_dict()

    def clear(self):

        pass

    def _cache_problem(self, problem):

        pass

    def create_abstract_state(self, problem, state):

        assert problem is not None
        return state

    def _initialize_nn_shape_dict(self):

        nn_shape_dict = {

            "state": len(self._atom_index_dict),
            "action": len(self._action_index_dict),
            "q_s_a": 1,
        }

        return nn_shape_dict

    def _initialize_nn_shapes(self, nn_names, layer_list, call_set):

        for nn_name in nn_names:

            if "state" == nn_name:

                call_set.add(nn_name)
                layer_list.append(nn_name)
            elif "action" == nn_name:

                call_set.add(nn_name)
                layer_list.append(nn_name)
            elif "q_s_a" == nn_name:

                call_set.add(nn_name)
                layer_list.append(nn_name)
            else:

                raise Exception("Unsupported nn_shape: %s" % (nn_name))

        pass

    def get_nn_shape(self, name):

        return self._nn_shape_dict[name]

    def get_max_action_params(self):

        return 0

    def initialize_nn(self, nn_inputs=["state", "action"],
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

    def _encode_state(self, nn_pkg, **kwargs):

        abstract_state = kwargs["abstract_state"]

        abs_vector = np.zeros(self._nn_shape_dict["state"])

        for atom in abstract_state.get_atom_set():

            if atom.predicate in self._unsupported_predicates:

                continue

            index = self._atom_index_dict[atom]
            abs_vector[index] = 1

        nn_pkg.encode("state", abs_vector)

    def get_action_index(self, action_name):

        try:

            return self._action_index_dict[action_name]
        except KeyError:

            return float("inf")

    def get_index_action(self, index):

        return self._index_action_dict[index]

    def _encode_action(self, nn_pkg, **kwargs):

        action = kwargs["action"]

        action_vector = np.zeros(self._nn_shape_dict["action"])
        index = self.get_action_index(str(action))
        action_vector[index] = 1

        nn_pkg.encode("action", action_vector)

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
