
from tensorflow import keras

import numpy as np


class BlocksworldNN:

    @staticmethod
    def get_2p_7h(abstract_domain):

        state_unary_preds_shape = abstract_domain.get_nn_input_shape(
            "state_unary_preds")
        role_counts_shape = abstract_domain.get_nn_input_shape(
            "state_unary_preds")

        action_shape = abstract_domain.get_nn_output_shape("action")
        action_param_preds_shape = abstract_domain.get_nn_output_shape(
            "action_param_0_preds")

        inputs = []
        input_1 = keras.Input(shape=state_unary_preds_shape,
                              name="state_unary_preds")

        input_2 = keras.Input(shape=role_counts_shape,
                              name="role_counts")
        inputs.append(input_1)
        inputs.append(input_2)

        concatenated_binned = []
        concatenated_raw = []

        concatenated_binned.append(input_1)
        concatenated_raw.append(input_2)

        for binary_predicate in abstract_domain.get_binary_predicates():

            output_shape = abstract_domain.get_nn_input_shape(binary_predicate)

            binary_pred = keras.Input(
                shape=output_shape,
                name=binary_predicate)

            role_count_binary_pred = keras.Input(
                shape=output_shape,
                name="role_count_%s" % (binary_predicate))

            inputs.append(binary_pred)
            inputs.append(role_count_binary_pred)

            concatenated_binned.append(keras.layers.Reshape(
                (1, np.prod(output_shape)))(binary_pred))
            concatenated_raw.append(keras.layers.Reshape(
                (1, np.prod(output_shape)))(role_count_binary_pred))

        binned_input = keras.layers.concatenate(concatenated_binned)
        raw_input = keras.layers.concatenate(concatenated_raw)

        h_0 = keras.layers.Dense(64, activation="relu")(binned_input)
        h_1 = keras.layers.Dense(64, activation="relu")(h_0)

        h_2 = keras.layers.Dense(64, activation="sigmoid")(binned_input)
        h_3 = keras.layers.Dense(64, activation="sigmoid")(h_2)

        h_4 = keras.layers.Dense(64, activation="sigmoid")(binned_input)
        h_5 = keras.layers.Dense(64, activation="sigmoid")(h_4)

        h_6 = keras.layers.Dense(64, activation="relu")(raw_input)
        h_7 = keras.layers.Dense(64, activation="relu")(h_6)

        output_1 = keras.layers.Dense(action_shape, activation="softmax",
                                      name="action")(h_1)

        output_2 = keras.layers.Dense(
            action_param_preds_shape,
            activation="sigmoid",
            name="action_param_0_preds")(h_3)

        output_3 = keras.layers.Dense(
            action_param_preds_shape,
            activation="sigmoid",
            name="action_param_1_preds")(h_5)

        output_4 = keras.layers.Dense(
            abstract_domain.get_nn_output_shape("plan_length"),
            activation="relu",
            name="plan_length")(h_7)

        model = keras.Model(inputs=inputs,
                            outputs=[output_1, output_2, output_3, output_4])

        loss_func_dict = {
            "action": "categorical_crossentropy",
            "action_param_0_preds": "binary_crossentropy",
            "action_param_1_preds": "binary_crossentropy",
            "plan_length": "mae",
        }

        loss_weights_dict = {
            "action": 1.0,
            "action_param_0_preds": 1.0,
            "action_param_1_preds": 1.0,
            "plan_length": 1.0,
        }

        metrics_dict = {
            "action": "categorical_accuracy",
            "action_param_0_preds": "binary_accuracy",
            "action_param_1_preds": "binary_accuracy",
            "plan_length": "mae",
        }

        model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                      loss=loss_func_dict,
                      loss_weights=loss_weights_dict,
                      metrics=metrics_dict)

        output_index_dict = {}
        for i in range(len(model.output_names)):

            output_index_dict[model.output_names[i]] = i

        return model, output_index_dict, "keras"

    @staticmethod
    def get_nn(abstract_domain, nn_name):

        if "2p_7h" == nn_name:

            model, output_index_dict, arch = BlocksworldNN.get_2p_7h(
                abstract_domain)
        else:

            raise Exception("Unknown nn_name={}".format(nn_name))

        return model, output_index_dict, arch
