
from generalized_learning.abstraction.canonical import CanonicalAbstractionMLP
from generalized_learning.abstraction.concrete import ConcreteMLP
from generalized_learning.abstraction.description_logic import DescriptionLogicMLP
from neural_net.nn import NN


class RAN:

    @staticmethod
    def load(model_dir, nn_name):

        ran = RAN()
        ran.q_nn = NN.load(model_dir,
                           "%s_q" % (nn_name))

        ran.policy_nn = NN.load(model_dir,
                                "%s_action" % (nn_name))

        # Set both the policy and Q network with the same abstraction
        ran.policy_nn._abstract_domain = ran.q_nn._abstract_domain

        return ran

    @staticmethod
    def _get_abstraction(abstraction_type, domain_filepath, problem_list,
                         **kwargs):

        if abstraction_type == "description_logic_mlp":

            abstraction = DescriptionLogicMLP.create(
                domain_filepath,
                problem_list,
                feature_file=kwargs["feature_file"])

        elif abstraction_type == "canonical_abstraction_mlp":

            abstraction = CanonicalAbstractionMLP.create(domain_filepath,
                                                         None)
        elif abstraction_type == "concrete":

            abstraction = ConcreteMLP.create(domain_filepath, problem_list)
        else:

            raise Exception("Unknown abstraction function")

        return abstraction

    @staticmethod
    def _get_q_network_params(abstraction_type, abstraction):

        if abstraction_type == "description_logic_mlp":

            nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set = \
                abstraction.initialize_nn(
                    nn_inputs=["state", "action", "action_params"],
                    nn_outputs=["q_s_a"])

        elif abstraction_type == "canonical_abstraction_mlp":

            nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set = \
                abstraction.initialize_nn(
                    nn_inputs=["features", "action", "action_params_features"],
                    nn_outputs=["q_s_a"])
        elif abstraction_type == "concrete":

            nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set = \
                abstraction.initialize_nn(
                    nn_inputs=["state", "action"],
                    nn_outputs=["q_s_a"])
        else:

            raise Exception("Unknown abstraction function")
            return None, None, None, None

        return nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set

    @staticmethod
    def _get_policy_network_params(abstraction_type, abstraction):

        if abstraction_type == "description_logic_mlp":

            nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set = \
                abstraction.initialize_nn(
                    nn_inputs=["state_bool"],
                    nn_outputs=["action", "action_params"])

        elif abstraction_type == "canonical_abstraction_mlp":

            nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set = \
                abstraction.initialize_nn(
                    nn_inputs=["features", "action", "action_params_features"],
                    nn_outputs=["action", "action_params"])
        elif abstraction_type == "concrete":

            nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set = \
                abstraction.initialize_nn(
                    nn_inputs=["state"],
                    nn_outputs=["action"])
        else:

            raise Exception("Unknown abstraction function")
            return None, None, None, None

        return nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set

    @staticmethod
    def get_instance(abstraction_type, domain_filepath, problem_list,
                     nn_type, nn_name, **kwargs):

        ran = RAN()

        abstraction = RAN._get_abstraction(abstraction_type, domain_filepath,
                                           problem_list, **kwargs)

        nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set = \
            RAN._get_q_network_params(abstraction_type, abstraction)

        ran.q_nn = NN.get_instance(abstraction,
                                   "%s_q" % (nn_type),
                                   "%s_q" % (nn_name),
                                   nn_inputs, nn_input_call_set,
                                   nn_outputs, nn_output_call_set)

        nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set = \
            RAN._get_policy_network_params(abstraction_type, abstraction)

        ran.policy_nn = NN.get_instance(abstraction,
                                        "%s_action" % (nn_type),
                                        "%s_action" % (nn_name),
                                        nn_inputs, nn_input_call_set,
                                        nn_outputs, nn_output_call_set)

        return ran

    def __init__(self):

        self.policy_nn = None
        self.q_nn = None

    def soft_save(self, output_dir):

        self.q_nn.soft_save(output_dir)
        self.policy_nn.soft_save(output_dir)

    def get_abstraction(self):

        return self.q_nn._abstract_domain
