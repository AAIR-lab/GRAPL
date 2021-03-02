
from abstraction.state import AbstractState
from neural_net.nn import NN

from .heuristic import Heuristic


class NNPLACT(Heuristic):

    NON_ZERO_DIVIDER = 1e-9

    def __init__(self, problem, model_dir, model_name, param_threshold=0.5):

        super(NNPLACT, self).__init__("nn_plact", problem)
        self._visited = set()
        self._param_threshold = param_threshold

        self._nn = NN.load(model_dir, model_name)
        self._abstract_domain = self._nn.get_abstract_domain()
        self._cached_current_state = None

    def get_properties(self):

        return {
            "name": "nn_plact",
            "nn": self._nn.get_properties()
        }

    def flatten(self, nn_output_pkg):

        action_vector = nn_output_pkg.decode("action")
        action_vector = action_vector.reshape(
            self._abstract_domain.get_nn_output_shape("action"))

        param_vectors = []
        for i in range(self._abstract_domain.get_max_action_params()):

            param_vector_name = "action_param_%u_preds" % (i)
            param_vector = nn_output_pkg.decode(param_vector_name)
            param_vector = param_vector.reshape(
                self._abstract_domain.get_nn_output_shape(param_vector_name))
            param_vectors.append(param_vector)

        return action_vector, param_vectors

    def update_candidates(self, current_node, candidates):

        current_abstract_state = AbstractState(
            self._problem, current_node.get_concrete_state())
        current_output_pkg = self._nn.predict(current_abstract_state)

        action_vector, param_vectors = self.flatten(current_output_pkg)

        abstract_states = []
        for candidate in candidates:

            concrete_state = candidate.get_concrete_state()
            abstract_states.append(AbstractState(
                self._problem, concrete_state))

        current_output_pkgs = self._nn.predict(abstract_states)

        if not isinstance(current_output_pkgs, list):

            current_output_pkgs = [current_output_pkgs]

        for i in range(len(candidates)):

            output_pkg = current_output_pkgs[i]
            candidate = candidates[i]

            try:

                action_score = 1.0 - self._compute_ascore(
                    self._abstract_domain,
                    current_abstract_state,
                    action_vector,
                    param_vectors,
                    candidate.get_action())
            except Exception:

                action_score = 0.5

            candidate.artificial_g = current_node.artificial_g + action_score

            if self._problem.is_goal_satisfied(candidate.get_concrete_state()):

                candidate.h = 0
            else:

                candidate.h = float(output_pkg.decode("plan_length"))

            candidate._fscore = candidate.artificial_g + candidate.h

    def _cache_current_state(self, current_state):

        self._c_abstract_state = AbstractState(self._problem, current_state)
        self._c_nn_output_pkg = self._nn.predict(self._c_abstract_state)

        self._c_plan_length = self._c_nn_output_pkg.decode("plan_length")
        self._c_plan_length = float(self._c_plan_length)

        self._c_action_vector = self._c_nn_output_pkg.decode("action")
        self._c_action_vector = self._c_action_vector.reshape(
            self._abstract_domain.get_nn_output_shape("action"))

        self._c_param_vectors = []
        for i in range(self._abstract_domain.get_max_action_params()):

            param_vector_name = "action_param_%u_preds" % (i)
            param_vector = self._c_nn_output_pkg.decode(param_vector_name)
            param_vector = param_vector.reshape(
                self._abstract_domain.get_nn_output_shape(param_vector_name))
            self._c_param_vectors.append(param_vector)

        self._cached_current_state = current_state

    def _is_cached(self, current_state):

        return self._cached_current_state is current_state

    def _get_cached_values(self):

        return self._c_abstract_state, self._c_nn_output_pkg, \
            self._c_plan_length, self._c_action_vector, self._c_param_vectors

    def compute_d(self, current_state, unused_next_state, action):

        if not self._is_cached(current_state):

            self._cache_current_state(current_state)

        abstract_state, unused_nn_output_pkg, plan_length, action_vector, \
            param_vectors = self._get_cached_values()

        try:
            action_score = self._compute_ascore(self._abstract_domain,
                                                abstract_state,
                                                action_vector,
                                                param_vectors,
                                                action)
        except Exception:

            action_score = 0.5

        assert action_score <= 1.0
        return 1.0 - action_score

    def compute_h(self, unused_current_state, next_state, action):

        if self._problem.is_goal_satisfied(next_state):

            return 0.0

        else:
            if not self._is_cached(next_state):

                self._cache_current_state(next_state)

            abstract_state, unused_nn_output_pkg, plan_length, action_vector, \
                param_vectors = self._get_cached_values()

            # / (action_score + NNPLACT.NON_ZERO_DIVIDER)
            h = plan_length

            return h

    def _compute_ascore(self, abstract_domain, abstract_state,
                        action_vector, param_vectors, action):

        action_score = 1.0

        action_index = abstract_domain.get_action_index(action.get_name())
        assert action_index != float("inf")

        action_score *= action_vector[action_index]

        param_list = action.get_param_list()
        max_params = abstract_domain.get_max_action_params()
        assert len(param_list) <= max_params

        param_score = 0.0
        for i in range(len(param_list)):

            action_param = param_list[i]
            role = abstract_state.get_role(action_param)
            unary_predicates = role.get_unary_predicates()

            # Get the total number of correct parameter predictions.
            num_correct = 0
            for j in range(len(param_vectors[i])):

                unary_predicate = abstract_domain.get_index_action_param(j)
                if param_vectors[i][j] > self._param_threshold:

                    num_correct += unary_predicate in unary_predicates
                else:

                    num_correct += unary_predicate not in unary_predicates

            param_score += num_correct / len(param_vectors[i])

        for i in range(len(param_list), max_params):

            for j in range(len(param_vectors[i])):

                num_correct += param_vectors[i][i] < self._param_threshold

        param_score /= max_params
        action_score *= param_score

        assert action_score >= 0 and action_score <= 1.0
        return action_score

    def expand(self, parent):

        raise NotImplementedError
