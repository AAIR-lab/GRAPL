import re

from tensorflow import keras

import numpy as np


class DeliveryNN:

    @staticmethod
    def get_delivery_nh(abstract_domain):

        state_unary_preds_shape = abstract_domain.get_nn_input_shape(
            "state_unary_preds")

        action_shape = abstract_domain.get_nn_output_shape("action")
        action_param_preds_shape = abstract_domain.get_nn_output_shape(
            "action_param_0_preds")

        input_1 = keras.Input(shape=state_unary_preds_shape,
                              name="state_unary_preds")

        output_1 = keras.layers.Dense(action_shape, activation="softmax",
                                      name="action")(input_1)

        concatenated_inputs = keras.layers.concatenate([input_1, output_1])

        output_2 = keras.layers.Dense(
            action_param_preds_shape,
            activation="sigmoid",
            name="action_param_0_preds")(concatenated_inputs)

        output_3 = keras.layers.Dense(
            action_param_preds_shape,
            activation="sigmoid",
            name="action_param_1_preds")(concatenated_inputs)

        model = keras.Model(inputs=[input_1],
                            outputs=[output_1, output_2, output_3])

        loss_func_dict = {
            "action": "categorical_crossentropy",
            "action_param_0_preds": "binary_crossentropy",
            "action_param_1_preds": "binary_crossentropy",
        }

        loss_weights_dict = {
            "action": 1.0,
            "action_param_0_preds": 1.0,
            "action_param_1_preds": 1.0,
        }

        metrics_dict = {
            "action": "categorical_accuracy",
            "action_param_0_preds": "binary_accuracy",
            "action_param_1_preds": "binary_accuracy",
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
    def get_delivery_7h(abstract_domain):

        state_unary_preds_shape = abstract_domain.get_nn_input_shape(
            "state_unary_preds")
        role_counts_shape = abstract_domain.get_nn_input_shape(
            "state_unary_preds")

        action_shape = abstract_domain.get_nn_output_shape("action")
        action_param_preds_shape = abstract_domain.get_nn_output_shape(
            "action_param_0_preds")

        input_1 = keras.Input(shape=state_unary_preds_shape,
                              name="state_unary_preds")

        input_2 = keras.Input(shape=role_counts_shape,
                              name="role_counts")

        h_0 = keras.layers.Dense(32, activation="relu")(input_1)
        h_1 = keras.layers.Dense(32, activation="relu")(h_0)

        h_2 = keras.layers.Dense(32, activation="sigmoid")(input_1)
        h_3 = keras.layers.Dense(32, activation="sigmoid")(h_2)

        h_4 = keras.layers.Dense(32, activation="sigmoid")(input_1)
        h_5 = keras.layers.Dense(32, activation="sigmoid")(h_4)

        h_6 = keras.layers.Dense(32, activation="relu")(input_2)
        h_7 = keras.layers.Dense(32, activation="relu")(h_6)

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

        model = keras.Model(inputs=[input_1, input_2],
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

        if "delivery_nh" == nn_name:

            model, output_index_dict, arch = DeliveryNN.get_delivery_nh(
                abstract_domain)
        elif "delivery_7h" == nn_name:

            model, output_index_dict, arch = DeliveryNN.get_delivery_7h(
                abstract_domain)
        else:

            raise Exception("Unknown nn_name={}".format(nn_name))

        return model, output_index_dict, arch
