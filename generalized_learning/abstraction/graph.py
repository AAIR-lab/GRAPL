'''
Created on Jul 27, 2021

@author: anonymous
'''
import torch
from torch_geometric.data import Data

from generalized_learning.concretized.domain import Domain
from generalized_learning.neural_net.nn import NNPkg
import numpy as np

from pddl.pddl_types import TypedObject


class GraphAbstraction:

    _0ARY_OBJ = TypedObject("null", "0ary")

    @staticmethod
    def create(self, domain_filepath, problem_list, **kwargs):

        return GraphAbstraction(domain_filepath, problem_list, **kwargs)

    def __init__(self, domain_filepath, problem_list, **kwargs):

        domain = Domain(domain_filepath)
        self._domain = domain

        self._unsupported_predicates = set(["="])
        self._problem = None
        self._goals = None

        self._max_action_params = domain.get_max_action_params()

        self._node_features = {}
        self._edge_features = {}

        for action in domain.get_action_templates():

            self._node_features[action.name] = len(self._node_features)

        for i in range(self._max_action_params):

            name = "action_param_%u" % (i)
            self._node_features[name] = len(self._node_features)

        for predicate in self._domain.predicates:

            if predicate.name in self._unsupported_predicates:

                continue

            if len(predicate.arguments) < 2:

                data_dict = self._node_features
            else:
                data_dict = self._edge_features

            data_dict.setdefault(predicate.name, len(data_dict))
            data_dict.setdefault("g_%s" % (predicate.name), len(data_dict))

        self._encoding_funcs = {

            "state": self._encode_state,
            "q_s_a": self._encode_q_s_a,
        }

    def _initialize_nn_shapes(self, nn_shape_dict, nn_names, call_set):

        for nn_name in nn_names:

            if "nodes" == nn_name:

                call_set.add("state")
                nn_shape_dict[nn_name] = len(self._node_features)
            elif "edges" == nn_name:

                call_set.add("state")
                nn_shape_dict[nn_name] = len(self._edge_features)
            elif "q_s_a" == nn_name:

                call_set.add("q_s_a")
                nn_shape_dict[nn_name] = 1
            else:

                raise Exception("Unsupported nn_shape: %s" % (nn_name))

        pass

    def initialize_nn(self, nn_inputs=["nodes", "edges"],
                      nn_outputs=["q_s_a"]):

        self._nn_input_shape_dict = {}
        self._nn_input_call_set = set()
        self._initialize_nn_shapes(self._nn_input_shape_dict,
                                   nn_inputs,
                                   self._nn_input_call_set)

        self._nn_output_shape_dict = {}
        self._nn_output_call_set = set()
        self._initialize_nn_shapes(self._nn_output_shape_dict,
                                   nn_outputs,
                                   self._nn_output_call_set)

        # Create a fixed order list of the inputs and outputs.
        # This can help in concatenation across runs.
        self._nn_inputs = list(self._nn_input_shape_dict.keys())
        self._nn_outputs = list(self._nn_output_shape_dict.keys())

        self._is_nn_initialized = True

    def get_nn_inputs(self):

        return self._nn_inputs

    def get_nn_outputs(self):

        return self._nn_outputs

    def get_nn_input_shape(self, name):

        return self._nn_input_shape_dict[name]

    def get_nn_output_shape(self, name):

        return self._nn_output_shape_dict[name]

    def _cache_problem(self, problem):

        if id(self._problem) != id(problem):

            self._problem = problem
            self._goals = self._problem.get_goal_literals()

    def clear(self):

        self._problem = None
        self._static_atoms = None
        self._model_factory = None

    def _encode_state(self, nn_pkg, **kwargs):

        state = kwargs["abstract_state"]

        # Create all node indices.
        node_index = {}
        nodes = []

        for typed_obj in [GraphAbstraction._0ARY_OBJ] + self._problem.get_typed_objects():

            name = typed_obj.name
            node_index[name] = len(node_index)
            nodes.append(np.zeros(len(self._node_features)))

        edges = {}
        for atom in state.get_atom_set():

            name = atom.predicate

            if name in self._unsupported_predicates:

                continue

            g_name = "g_%s" % (name)

            if len(atom.args) < 2:

                if len(atom.args) == 0:

                    obj_name = "null"
                else:

                    obj_name = atom.args[0]

                idx = node_index[obj_name]
                nodes[idx][self._node_features[name]] = 1

            else:

                o1 = node_index[atom.args[0]]
                o2 = node_index[atom.args[1]]

                edge_vector = edges.setdefault(
                    (o1, o2),
                    np.zeros(len(self._edge_features)))

                edge_vector[self._edge_features[name]] = 1

        for atom in self._goals:

            g_name = "g_%s" % (atom.predicate)

            if len(atom.args) < 2:

                if len(atom.args) == 0:

                    obj_name = "null"
                else:

                    obj_name = atom.args[0]

                idx = node_index[obj_name]
                nodes[idx][self._node_features[g_name]] = 1
            else:

                o1 = node_index[atom.args[0]]
                o2 = node_index[atom.args[1]]

                edge_vector = edges.setdefault(
                    (o1, o2),
                    np.zeros(len(self._edge_features)))

                edge_vector[self._edge_features[g_name]] = 1

        action = kwargs["action"]
        param_list = action.get_param_list()

        for i in range(len(param_list)):

            obj = param_list[i]
            idx = node_index[obj]
            nodes[idx][self._node_features[action.get_name()]] = 1

            name = "action_param_%u" % (i)
            nodes[idx][self._node_features[name]] = 1

        ei = []
        ef = []

        for edge_index in edges:

            ei.append(edge_index)
            ef.append(edges[edge_index])

        data = Data(x=torch.tensor(nodes).float(),
                    edge_index=torch.tensor(ei).t().contiguous(),
                    edge_attr=torch.tensor(ef).float())

        nn_pkg.encode("data", data)

    def _encode_q_s_a(self, nn_pkg, **kwargs):

        q_s_a = kwargs["q_s_a"]

        assert self._nn_output_shape_dict["q_s_a"] == 1
        vector = np.zeros(1)
        vector[0] = q_s_a

        nn_pkg.encode("q_s_a", vector)

    def create_abstract_state(self, problem, state):

        assert problem is not None
        self._cache_problem(problem)

        return state

    def encode_nn_input(self, **kwargs):

        assert self._is_nn_initialized

        nn_pkg = NNPkg()

        for call_func in self._nn_input_call_set:

            self._encoding_funcs[call_func](nn_pkg, **kwargs)

        return nn_pkg

    def encode_nn_training_data(self, **kwargs):

        assert self._is_nn_initialized

        nn_pkg = self.encode_nn_input(**kwargs)

        for call_func in self._nn_output_call_set:

            self._encoding_funcs[call_func](nn_pkg, **kwargs)

        return nn_pkg
