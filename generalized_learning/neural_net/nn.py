import pathlib
import pickle

from tensorflow import keras

from abstraction.state import AbstractState
from benchmarks.barman.nn import BarmanNN
from benchmarks.blocksworld.nn import BlocksworldNN
from benchmarks.childsnack.nn import ChildsnackNN
from benchmarks.delivery.nn import DeliveryNN
from benchmarks.depots.nn import DepotsNN
from benchmarks.ferry.nn import FerryNN
from benchmarks.goldminer.nn import GoldminerNN
from benchmarks.grid.nn import GridNN
from benchmarks.gripper.nn import GripperNN
from benchmarks.grippers.nn import GrippersNN
from benchmarks.hanoi.nn import HanoiNN
from benchmarks.logistics.nn import LogisticsNN
from benchmarks.miconic.nn import MiconicNN
from benchmarks.npuzzle.nn import NPuzzleNN
from benchmarks.parking.nn import ParkingNN
from benchmarks.spanner.nn import SpannerNN
from benchmarks.tyreworld.nn import TyreworldNN
from benchmarks.visitall.nn import VisitAllNN
from neural_net.dataloader import get_loader
import numpy as np
import tensorflow as tf


class NNPkg:

    def __init__(self):

        self._layers = {}

    def encode(self, name, data):

        self._layers[name] = data

    def decode(self, name):

        return self._layers[name]


class GenericNN:

    @staticmethod
    def get_dense_layer(neurons, activation, input_layer):

        return keras.layers.Dense(neurons, activation=activation)(input_layer)

    @staticmethod
    def get_generic_merged_skip(abstract_domain, num_hidden_layers=2, neurons=32):

        BINNED_CONFIG = [("state_unary_preds", False)]
        RAW_CONFIG = [("role_counts", False)]

        for arity in abstract_domain.get_arities():

            for predicate in abstract_domain.get_predicates(arity):

                BINNED_CONFIG.append((predicate, True))
                RAW_CONFIG.append(("role_count_%s" % (predicate), True))

        input_layers = []
        concatenate_binned = []
        concatenate_raw = []

        GenericNN._create_input_layers(abstract_domain, BINNED_CONFIG,
                                       input_layers, concatenate_binned)

        GenericNN._create_input_layers(abstract_domain, RAW_CONFIG,
                                       input_layers, concatenate_raw)

        if len(concatenate_binned) == 1:

            binned_layer = concatenate_binned[0]
        else:
            binned_layer = keras.layers.concatenate(concatenate_binned)

        if len(concatenate_raw) == 1:

            raw_layer = concatenate_raw[0]
        else:

            raw_layer = keras.layers.concatenate(concatenate_raw)

        output_layers = []
        loss_func_dict = {}
        loss_weights_dict = {}
        metrics_dict = {}

        action_layer = GenericNN._create_action_output_layer(abstract_domain,
                                                             num_hidden_layers,
                                                             neurons,
                                                             "relu",
                                                             binned_layer,
                                                             loss_func_dict,
                                                             loss_weights_dict,
                                                             metrics_dict)
        output_layers.append(action_layer)

        plan_length_layer = GenericNN._create_plan_length_output_layer(
            abstract_domain,
            num_hidden_layers,
            neurons,
            "relu",
            raw_layer,
            loss_func_dict,
            loss_weights_dict,
            metrics_dict)
        output_layers.append(plan_length_layer)

        action_param_layer = keras.layers.concatenate(
            concatenate_binned + [action_layer])

        for i in range(abstract_domain.get_max_action_params()):

            name = "action_param_%u_preds" % (i)
            param_layer = GenericNN._create_action_param_output_layer(
                abstract_domain,
                name,
                num_hidden_layers,
                neurons,
                "relu",
                action_param_layer,
                loss_func_dict,
                loss_weights_dict,
                metrics_dict)

            output_layers.append(param_layer)

        model = keras.Model(inputs=input_layers, outputs=output_layers)
        model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                      loss=loss_func_dict,
                      loss_weights=loss_weights_dict,
                      metrics=metrics_dict)

        output_index_dict = {}
        for i in range(len(model.output_names)):

            output_index_dict[model.output_names[i]] = i

        return model, output_index_dict, "keras"

    @staticmethod
    def get_input_layer(domain, name, flatten=False):

        shape = domain.get_nn_input_shape(name)
        input_layer = keras.Input(shape=shape,
                                  name=name)

        if flatten:

            flattened_layer = keras.layers.Reshape(
                (1, np.prod(shape)))(input_layer)

            return input_layer, flattened_layer
        else:

            return input_layer, None

    @staticmethod
    def _create_input_layers(domain, config, input_list, concatenate_list):

        for layer_config in config:

            name = layer_config[0]
            flatten = layer_config[1]

            input_layer, flattened_layer = GenericNN.get_input_layer(
                domain, name, flatten)

            input_list.append(input_layer)

            if flattened_layer is None:

                concatenate_list.append(input_layer)
            else:

                concatenate_list.append(flattened_layer)

    @staticmethod
    def _create_hidden_layers(num_layers, neurons, activation, input_layer):

        last_layer = input_layer
        for _ in range(num_layers):

            layer = keras.layers.Dense(
                neurons, activation=activation)(last_layer)
            last_layer = layer

        return last_layer

    @staticmethod
    def _create_action_output_layer(domain, num_hidden_layers, neurons,
                                    hidden_activation, input_layer,
                                    loss_func_dict, loss_weights_dict,
                                    metrics_dict):

        layer = GenericNN._create_hidden_layers(num_hidden_layers, neurons,
                                                hidden_activation, input_layer)
        output_layer = keras.layers.Dense(
            domain.get_nn_output_shape("action"),
            activation="softmax",
            name="action")(layer)

        loss_weights_dict["action"] = 1.0
        if np.prod(domain.get_nn_output_shape("action")) == 1:

            loss_func_dict["action"] = "binary_crossentropy"
            metrics_dict["action"] = "binary_accuracy"
        else:

            loss_func_dict["action"] = "categorical_crossentropy"
            metrics_dict["action"] = "categorical_accuracy"

        return output_layer

    @staticmethod
    def _create_action_param_output_layer(domain, name,
                                          num_hidden_layers, neurons,
                                          hidden_activation, input_layer,
                                          loss_func_dict, loss_weights_dict,
                                          metrics_dict):

        layer = GenericNN._create_hidden_layers(num_hidden_layers, neurons,
                                                hidden_activation, input_layer)
        output_layer = keras.layers.Dense(
            domain.get_nn_output_shape(name),
            activation="sigmoid",
            name=name)(layer)

        loss_weights_dict[name] = 1.0
        loss_func_dict[name] = "binary_crossentropy"
        metrics_dict[name] = "binary_accuracy"

        return output_layer

    @staticmethod
    def _create_plan_length_output_layer(domain, num_hidden_layers, neurons,
                                         hidden_activation, input_layer,
                                         loss_func_dict, loss_weights_dict,
                                         metrics_dict):

        layer = GenericNN._create_hidden_layers(num_hidden_layers, neurons,
                                                hidden_activation, input_layer)
        output_layer = keras.layers.Dense(
            domain.get_nn_output_shape("plan_length"),
            activation="relu",
            name="plan_length")(layer)

        loss_weights_dict["plan_length"] = 1.0
        loss_func_dict["plan_length"] = "mae"
        metrics_dict["plan_length"] = "mae"

        return output_layer

    @staticmethod
    def get_generic(abstract_domain, num_hidden_layers=2, neurons=32):

        BINNED_CONFIG = [("state_unary_preds", False)]
        RAW_CONFIG = [("role_counts", False)]

        for arity in abstract_domain.get_arities():

            for predicate in abstract_domain.get_predicates(arity):

                BINNED_CONFIG.append((predicate, True))
                RAW_CONFIG.append(("role_count_%s" % (predicate), True))

        input_layers = []
        concatenate_binned = []
        concatenate_raw = []

        GenericNN._create_input_layers(abstract_domain, BINNED_CONFIG,
                                       input_layers, concatenate_binned)

        GenericNN._create_input_layers(abstract_domain, RAW_CONFIG,
                                       input_layers, concatenate_raw)

        binned_layer = keras.layers.concatenate(concatenate_binned)
        raw_layer = keras.layers.concatenate(concatenate_raw)

        output_layers = []
        loss_func_dict = {}
        loss_weights_dict = {}
        metrics_dict = {}

        action_layer = GenericNN._create_action_output_layer(abstract_domain,
                                                             num_hidden_layers,
                                                             neurons,
                                                             "relu",
                                                             binned_layer,
                                                             loss_func_dict,
                                                             loss_weights_dict,
                                                             metrics_dict)
        output_layers.append(action_layer)

        plan_length_layer = GenericNN._create_plan_length_output_layer(
            abstract_domain,
            num_hidden_layers,
            neurons,
            "relu",
            raw_layer,
            loss_func_dict,
            loss_weights_dict,
            metrics_dict)
        output_layers.append(plan_length_layer)

        for i in range(abstract_domain.get_max_action_params()):

            name = "action_param_%u_preds" % (i)
            param_layer = GenericNN._create_action_param_output_layer(
                abstract_domain,
                name,
                num_hidden_layers,
                neurons,
                "sigmoid",
                binned_layer,
                loss_func_dict,
                loss_weights_dict,
                metrics_dict)

            output_layers.append(param_layer)

        model = keras.Model(inputs=input_layers, outputs=output_layers)
        model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                      loss=loss_func_dict,
                      loss_weights=loss_weights_dict,
                      metrics=metrics_dict)

        output_index_dict = {}
        for i in range(len(model.output_names)):

            output_index_dict[model.output_names[i]] = i

        return model, output_index_dict, "keras"

    @staticmethod
    def get_generic_merged(abstract_domain, num_hidden_layers=2, neurons=32):

        BINNED_CONFIG = [("state_unary_preds", False)]
        RAW_CONFIG = [("role_counts", False)]

        for arity in abstract_domain.get_arities():

            for predicate in abstract_domain.get_predicates(arity):

                BINNED_CONFIG.append((predicate, True))
                RAW_CONFIG.append(("role_count_%s" % (predicate), True))

        input_layers = []
        concatenate_binned = []
        concatenate_raw = []

        GenericNN._create_input_layers(abstract_domain, BINNED_CONFIG,
                                       input_layers, concatenate_binned)

        GenericNN._create_input_layers(abstract_domain, RAW_CONFIG,
                                       input_layers, concatenate_raw)

        merged = concatenate_binned + concatenate_raw

        binned_layer = keras.layers.concatenate(merged)
        raw_layer = binned_layer

        output_layers = []
        loss_func_dict = {}
        loss_weights_dict = {}
        metrics_dict = {}

        action_layer = GenericNN._create_action_output_layer(abstract_domain,
                                                             num_hidden_layers,
                                                             neurons,
                                                             "relu",
                                                             binned_layer,
                                                             loss_func_dict,
                                                             loss_weights_dict,
                                                             metrics_dict)
        output_layers.append(action_layer)

        plan_length_layer = GenericNN._create_plan_length_output_layer(
            abstract_domain,
            num_hidden_layers,
            neurons,
            "relu",
            raw_layer,
            loss_func_dict,
            loss_weights_dict,
            metrics_dict)
        output_layers.append(plan_length_layer)

        for i in range(abstract_domain.get_max_action_params()):

            name = "action_param_%u_preds" % (i)
            param_layer = GenericNN._create_action_param_output_layer(
                abstract_domain,
                name,
                num_hidden_layers,
                neurons,
                "sigmoid",
                binned_layer,
                loss_func_dict,
                loss_weights_dict,
                metrics_dict)

            output_layers.append(param_layer)

        model = keras.Model(inputs=input_layers, outputs=output_layers)
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

        if "generic" == nn_name:

            model, output_index_dict, arch = GenericNN.get_generic(
                abstract_domain)
        elif "generic_merged" == nn_name:

            model, output_index_dict, arch = GenericNN.get_generic_merged(
                abstract_domain)
        elif "gms" == nn_name:

            model, output_index_dict, arch = GenericNN.get_generic_merged_skip(
                abstract_domain)
        else:

            raise Exception("Unknown nn_name={}".format(nn_name))

        return model, output_index_dict, arch


class NN:

    SUPPORTED_ARCHITECTURES = set(["keras", "pytorch"])
    DYING_RELU_THRESHOLD = 0.1

    PICKLED_MODEL_EXTENSION = ".nn.model"
    KERAS_MODEL_EXTENSION = ".keras.h5"
    PYTORCH_MODEL_EXTENSION = ".pth"

    @staticmethod
    def get_instance(abstract_domain, nn_type, nn_name):

        if "generic" == nn_type:

            model, output_index_dict, arch = GenericNN.get_nn(
                abstract_domain, nn_name)
        elif "delivery" == nn_type:

            model, output_index_dict, arch = DeliveryNN.get_nn(
                abstract_domain, nn_name)
        elif "miconic" == nn_type:

            model, output_index_dict, arch = MiconicNN.get_nn(
                abstract_domain, nn_name)
        elif "visitall" == nn_type:

            model, output_index_dict, arch = VisitAllNN.get_nn(
                abstract_domain, nn_name)
        elif "goldminer" == nn_type:

            model, output_index_dict, arch = GoldminerNN.get_nn(
                abstract_domain, nn_name)
        elif "spanner" == nn_type:

            model, output_index_dict, arch = SpannerNN.get_nn(
                abstract_domain, nn_name)
        elif "childsnack" == nn_type:

            model, output_index_dict, arch = ChildsnackNN.get_nn(
                abstract_domain, nn_name)
        elif "ferry" == nn_type:

            model, output_index_dict, arch = FerryNN.get_nn(
                abstract_domain, nn_name)
        elif "logistics" == nn_type:

            model, output_index_dict, arch = LogisticsNN.get_nn(
                abstract_domain, nn_name)
        elif "grippers" == nn_type:

            model, output_index_dict, arch = GrippersNN.get_nn(
                abstract_domain, nn_name)
        elif "gripper" == nn_type:

            model, output_index_dict, arch = GripperNN.get_nn(
                abstract_domain, nn_name)
        elif "hanoi" == nn_type:

            model, output_index_dict, arch = HanoiNN.get_nn(
                abstract_domain, nn_name)
        elif "grid" == nn_type:

            model, output_index_dict, arch = GridNN.get_nn(
                abstract_domain, nn_name)
        elif "npuzzle" == nn_type:

            model, output_index_dict, arch = NPuzzleNN.get_nn(
                abstract_domain, nn_name)
        elif "blocksworld" == nn_type:

            model, output_index_dict, arch = BlocksworldNN.get_nn(
                abstract_domain, nn_name)
        elif "barman" == nn_type:

            model, output_index_dict, arch = BarmanNN.get_nn(
                abstract_domain, nn_name)
        elif "parking" == nn_type:

            model, output_index_dict, arch = ParkingNN.get_nn(
                abstract_domain, nn_name)
        elif "tyreworld" == nn_type:

            model, output_index_dict, arch = TyreworldNN.get_nn(
                abstract_domain, nn_name)
        elif "depots" == nn_type:

            model, output_index_dict, arch = DepotsNN.get_nn(
                abstract_domain, nn_name)
        else:

            raise Exception("Unknown nn_type={}".format(nn_type))

        return NN(model, output_index_dict, nn_name, arch, abstract_domain)

    def save(self, directory):

        filepath = pathlib.Path(directory, "%s%s" % (self.get_name(),
                                                     NN.PICKLED_MODEL_EXTENSION))

        if "keras" == self._arch:

            model_filepath = "%s/%s%s" % (directory,
                                          self.get_name(),
                                          NN.KERAS_MODEL_EXTENSION)
            self._model.save(model_filepath)
            tf.keras.backend.clear_session()

            self._model = None
            file_handle = open(filepath, "wb")
            pickle.dump(self, file_handle)
        else:

            raise Exception("save: Unknown arch={}".format(self._arch))

    @staticmethod
    def load(model_dir, model_name):

        filepath = "%s/%s%s" % (model_dir, model_name,
                                NN.PICKLED_MODEL_EXTENSION)

        file_handle = open(filepath, "rb")
        nn = pickle.load(file_handle)
        assert isinstance(nn, NN)

        if "keras" == nn._arch:

            model_filepath = "%s/%s%s" % (model_dir, model_name,
                                          NN.KERAS_MODEL_EXTENSION)

            tf.keras.backend.clear_session()
            g = tf.Graph()

            with g.as_default():

                sess = tf.compat.v1.Session(graph=g)
                with sess.as_default():
                    model = keras.models.load_model(model_filepath)

            nn._model = model
            nn._g = g
            nn._sess = sess
        else:
            raise Exception("load: Unknown arch={}".format(nn._arch))

        return nn

    def __init__(self, model, output_index_dict, name, arch, abstract_domain):

        assert arch in NN.SUPPORTED_ARCHITECTURES

        self._model = model
        self._output_index_dict = output_index_dict
        self._name = name
        self._arch = arch
        self._abstract_domain = abstract_domain
        self._train_failure_threshold = NN.DYING_RELU_THRESHOLD

    def get_properties(self):

        return {
            "name": self._name,
            "arch": self._arch
        }

    def get_name(self):

        return self._name

    def get_abstract_domain(self):

        return self._abstract_domain

    def plot_model(self, directory):

        if "keras" == self._arch:

            filepath = "%s/%s.png" % (directory, self._name)
            keras.utils.plot_model(self._model, filepath, show_shapes=True)
        else:

            raise Exception("Cannot plot the model.")

    def _train_keras(self, nn_train_pkgs_list, epochs, batch_size, shuffle, verbose=2):

        dataloader = get_loader(abstract_domain=self._abstract_domain, nn_package_list=nn_train_pkgs_list,
                                batch_size=batch_size, arch=self._arch, shuffle=shuffle)
        history = self._model.fit(dataloader, epochs=epochs, verbose=verbose)

        pl_loss = history.history["plan_length_loss"]
        return (pl_loss[0] - pl_loss[-1]) > self._train_failure_threshold

    def train(self, nn_train_pkgs_list, epochs, batch_size, shuffle):

        if "keras" == self._arch:

            return [self._train_keras(nn_train_pkgs_list,
                                      epochs, batch_size, shuffle)]
        else:

            assert False
            return [False]
            pass

        return []

    def evaluate(self, nn_test_pkgs_list, verbose=0):

        if "keras" == self._arch:
            dataloader = get_loader(abstract_domain=self._abstract_domain,
                                    nn_package_list=nn_test_pkgs_list, batch_size=32, arch=self._arch, shuffle=False)
            self._model.evaluate(dataloader, verbose=verbose)
        else:

            assert False
            pass

        return []

    def encode_nn_output(self, prediction, total_length=1):

        if total_length == 1:

            nn_output_pkg = NNPkg()

            for output_layer_name in self._output_index_dict:
                index = self._output_index_dict[output_layer_name]
                nn_output_pkg.encode(
                    output_layer_name, prediction[index])

            return nn_output_pkg
        else:

            nn_output_pkgs = []

            for i in range(total_length):

                nn_output_pkg = NNPkg()
                for output_layer_name in self._output_index_dict:
                    index = self._output_index_dict[output_layer_name]
                    nn_output_pkg.encode(
                        output_layer_name, prediction[index][i])

                nn_output_pkgs.append(nn_output_pkg)
            return nn_output_pkgs

    def _predict_keras(self, abstract_state):

        total_length = 1
        if isinstance(abstract_state, list):

            total_length = len(abstract_state)
            nn_input_pkg = []
            for state in abstract_state:

                nn_input_pkg.append(
                    self._abstract_domain.encode_nn_input(state))
        else:

            nn_input_pkg = self._abstract_domain.encode_nn_input(
                abstract_state)
            nn_input_pkg = [nn_input_pkg]

        nn_input_dict = self._abstract_domain.get_nn_input_dict(nn_input_pkg)

        with self._g.as_default():
            with self._sess.as_default():
                prediction = self._model.predict(nn_input_dict)

        nn_output_pkg = self.encode_nn_output(prediction, total_length)
        return nn_output_pkg

    def predict(self, abstract_state):

        if "keras" == self._arch:

            return self._predict_keras(abstract_state)
        else:

            assert False
            pass

        return None

    def get_action_score(self, problem, abstract_state, action):

        nn_output_pkg = self.predict(abstract_state)

        return self._abstract_domain.decode_nn_output(nn_output_pkg,
                                                      abstract_state,
                                                      action)
