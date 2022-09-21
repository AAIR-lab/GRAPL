import pathlib
import pickle

from tensorflow import keras
import torch
from torch.utils.data import DataLoader

from generalized_learning.neural_net.dataloader import KerasDataset
from generalized_learning.neural_net.torch import TorchA2C, TorchGenericAction
from generalized_learning.neural_net.torch import TorchGenericQ
from neural_net.dataloader import get_loader
import numpy as np
import tensorflow as tf


tf.config.run_functions_eagerly(False)


class NNPkg:

    def __init__(self):

        self._layers = {}

    def encode(self, name, data):

        self._layers[name] = data

    def decode(self, name):

        return self._layers[name]


class GenericNN:

    @staticmethod
    def get_generic_merged_skip(abstraction):

        input_layers = []
        flattened_layers = []
        for nn_input in abstraction.get_nn_inputs():

            shape = abstraction.get_nn_input_shape(nn_input)
            input_layer = keras.Input(shape=shape, name=nn_input)
            flattened_layer = keras.layers.Reshape(
                (np.prod(shape), ))(input_layer)
            input_layers.append(input_layer)
            flattened_layers.append(flattened_layer)

        flattened_layer = keras.layers.concatenate(flattened_layers)

        loss_func_dict = {}
        loss_weights_dict = {}
        metrics_dict = {}

        leaky_relu = keras.layers.LeakyReLU()(flattened_layer)

        hidden = keras.layers.Dense(32, activation="relu")(leaky_relu)
        hidden = keras.layers.Dense(32, activation="relu")(hidden)

        q_s_a_layer = keras.layers.Dense(
            abstraction.get_nn_output_shape("q_s_a"),
            name="q_s_a")(hidden)

        loss_weights_dict["q_s_a"] = 1.0
        loss_func_dict["q_s_a"] = "mse"
        metrics_dict["q_s_a"] = "mse"

        output_layers = {"q_s_a": q_s_a_layer}

        model = keras.Model(inputs=input_layers, outputs=output_layers)
        model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                      loss=loss_func_dict,
                      loss_weights=loss_weights_dict,
                      metrics=metrics_dict,
                      run_eagerly=False)

        return model, [("q_s_a", float)], "keras"


class NN:

    SUPPORTED_ARCHITECTURES = set(["keras", "pytorch"])
    DYING_RELU_THRESHOLD = 0.1

    PICKLED_MODEL_EXTENSION = ".nn.model"
    KERAS_MODEL_EXTENSION = ".keras.h5"
    PYTORCH_MODEL_EXTENSION = ".pth"

    @staticmethod
    def get_instance(abstract_domain, nn_type, nn_name,
                     nn_inputs, nn_input_call_set, nn_outputs,
                     nn_output_call_set):

        if "gms" == nn_type:

            model, output_list, arch = GenericNN.get_generic_merged_skip(
                abstract_domain)
        elif "torch_generic_q" == nn_type:

            model, output_list, arch = TorchGenericQ.create(
                abstract_domain, nn_inputs, nn_outputs)
        elif "torch_generic_action" == nn_type:

            model, output_list, arch = TorchGenericAction.create(
                abstract_domain, nn_inputs, nn_outputs)
        elif "py_a2c" == nn_type:

            model, output_list, arch = TorchA2C.create(
                abstract_domain)
        else:

            raise Exception("Unknown nn_type={}".format(nn_type))

        return NN(model, output_list, nn_name, arch, abstract_domain, nn_type,
                  nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set)

    def save(self, directory):

        filepath = pathlib.Path(directory, "%s%s" % (self.get_name(),
                                                     NN.PICKLED_MODEL_EXTENSION))

        self._abstract_domain.clear()

        if "keras" == self._arch:

            model_filepath = "%s/%s%s" % (directory,
                                          self.get_name(),
                                          NN.KERAS_MODEL_EXTENSION)
            self._model.save(model_filepath)
            tf.keras.backend.clear_session()

        elif "pytorch" == self._arch:

            model_filepath = "%s/%s%s" % (directory,
                                          self.get_name(),
                                          NN.PYTORCH_MODEL_EXTENSION)
            torch.save({
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
            }, model_filepath)

            pass
        else:
            raise Exception("save: Unknown arch={}".format(self._arch))

        self._model = None
        file_handle = open(filepath, "wb")
        pickle.dump(self, file_handle)

    def soft_save(self, directory):

        filepath = pathlib.Path(directory, "%s%s" % (self.get_name(),
                                                     NN.PICKLED_MODEL_EXTENSION))

        model = self._model

        if "keras" == self._arch:

            model_filepath = "%s/%s%s" % (directory,
                                          self.get_name(),
                                          NN.KERAS_MODEL_EXTENSION)
            self._model.save(model_filepath)
            file_handle = open(filepath, "wb")

            self._model = None
            pickle.dump(self, file_handle)
            self._model = model
        elif "pytorch" == self._arch:

            model = self._model
            self.save(directory)
            self._model = model
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

            model = keras.models.load_model(model_filepath)
            nn._model = model

        elif "pytorch" == nn._arch:

            model_filepath = "%s/%s%s" % (model_dir, model_name,
                                          NN.PYTORCH_MODEL_EXTENSION)
            model_dict = torch.load(model_filepath)

            nn._model = NN.get_instance(nn._abstract_domain, nn.nn_type,
                                        nn._name,
                                        nn.nn_inputs, nn.nn_input_call_set,
                                        nn.nn_outputs,
                                        nn.nn_output_call_set)._model

            nn._optimizer = torch.optim.Adam(
                params=nn._model.parameters())
            nn._loss_fn = torch.nn.MSELoss()

            nn._model.load_state_dict(model_dict['model_state_dict'])
            nn._optimizer.load_state_dict(model_dict['optimizer_state_dict'])

        else:
            raise Exception("load: Unknown arch={}".format(nn._arch))

        return nn

    def __init__(self, model, output_list, name, arch, abstract_domain, nn_type,
                 nn_inputs, nn_input_call_set, nn_outputs, nn_output_call_set):

        assert arch in NN.SUPPORTED_ARCHITECTURES

        self._model = model
        self._output_list = output_list
        self._name = name
        self._arch = arch
        self._abstract_domain = abstract_domain
        self._train_failure_threshold = NN.DYING_RELU_THRESHOLD
        self._g = None

        self.nn_type = nn_type
        self.nn_inputs = nn_inputs
        self.nn_input_call_set = nn_input_call_set
        self.nn_outputs = nn_outputs
        self.nn_output_call_set = nn_output_call_set

        if self._arch == "pytorch":

            self._loss_fn = torch.nn.MSELoss()
            self._optimizer = torch.optim.Adam(
                params=self._model.parameters())

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

    def _train_keras(self, nn_train_pkgs_list, epochs, batch_size, shuffle, verbose=0):

        dataloader = get_loader(abstract_domain=self._abstract_domain, nn_package_list=nn_train_pkgs_list,
                                batch_size=batch_size, arch=self._arch, shuffle=shuffle)
        history = self._model.fit(dataloader, epochs=epochs, verbose=verbose)

        return False

#         pl_loss = history.history["plan_length_loss"]
#         return (pl_loss[0] - pl_loss[-1]) > self._train_failure_threshold

    def train_pytorch(self, nn_train_pkgs_list, epochs, batch_size, shuffle):

        torch_dataset = self._model.get_dataset(self.nn_inputs,
                                                self.nn_outputs,
                                                nn_train_pkgs_list)
        dataloader = DataLoader(torch_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle)

        for _ in range(epochs):

            for _, (X, Y) in enumerate(dataloader):

                prediction = self._model(X)

                loss = 0
                for i in range(len(self._output_list)):

                    name = self._output_list[i][0]
                    loss_func = self._output_list[i][2]
                    loss += loss_func(prediction[name], Y[name])

                # Backpropagation
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

    def train(self, nn_train_pkgs_list, epochs, batch_size, shuffle):

        if "keras" == self._arch:

            return [self._train_keras(nn_train_pkgs_list,
                                      epochs, batch_size, shuffle)]
        elif "pytorch" == self._arch:

            self.train_pytorch(nn_train_pkgs_list, epochs, batch_size, shuffle)
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

        nn_output_pkgs = []

        for i in range(total_length):

            nn_output_pkg = NNPkg()
            for j in range(len(self._output_list)):

                output_name = self._output_list[j][0]
                conversion_func = self._output_list[j][1]

                if conversion_func is None:
                    nn_output_pkg.encode(output_name,
                                         prediction[output_name][i])
                else:
                    value = conversion_func(prediction[output_name][i])
                    nn_output_pkg.encode(output_name,
                                         value)

            nn_output_pkgs.append(nn_output_pkg)
        return nn_output_pkgs

    def _fit_pkgs_pytorch(self, nn_input_pkg_list):

        torch_dataset = self._model.get_dataset(self.nn_inputs,
                                                self.nn_outputs,
                                                nn_input_pkg_list)

        dataloader = DataLoader(torch_dataset,
                                batch_size=len(nn_input_pkg_list),
                                shuffle=False)

        with torch.no_grad():
            for X, _ in dataloader:
                prediction = self._model(X)

        nn_output_pkg = self.encode_nn_output(
            prediction, len(nn_input_pkg_list))
        return nn_output_pkg

    def _fit_pkgs_keras(self, nn_input_pkg_list):

        nn_input_dict = KerasDataset.filter_data_to_np(
            self._abstract_domain.get_nn_inputs(), nn_input_pkg_list)

        prediction = self._model.predict(nn_input_dict)

        nn_output_pkgs = self.encode_nn_output(prediction,
                                               len(nn_input_pkg_list))

        return nn_output_pkgs

    def fit_pkgs(self, nn_input_pkg_list):

        if self._arch == "keras":

            return self._fit_pkgs_keras(nn_input_pkg_list)
        elif self._arch == "pytorch":

            return self._fit_pkgs_pytorch(nn_input_pkg_list)
        else:

            assert False
            raise Exception("Unsupported architecture.")
