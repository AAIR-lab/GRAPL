'''
Created on Jul 13, 2021

@author: anonymous
'''


from torch import nn
import torch

import numpy as np
import torch.nn.functional as F


class TorchDataset:

    def __init__(self, nn_inputs, nn_outputs, nn_pkg_list):

        self.nn_inputs = nn_inputs
        self.nn_outputs = nn_outputs
        self.nn_pkg_list = nn_pkg_list
        pass

    def __len__(self):

        return len(self.nn_pkg_list)

    def __getitem__(self, idx):

        nn_pkg = self.nn_pkg_list[idx]
        x = []

        y = {}
        for layer_name in self.nn_outputs:

            try:
                y[layer_name] = torch.tensor(
                    nn_pkg.decode(layer_name)).float()
            except KeyError:

                pass

        for layer_name in self.nn_inputs:

            vector = nn_pkg.decode(layer_name)
            x = np.concatenate((x, vector), axis=None)

        return torch.tensor(x).float(), y


class TorchA2C(nn.Module):

    @staticmethod
    def create(abstraction):

        state_size = abstraction.get_nn_input_shape("state")
        action_size = abstraction.get_nn_output_shape("action")

        return TorchA2C(state_size, action_size), \
            [("action", None, None),
                ("v_s", None, None)], \
            "pytorch"

    def get_dataset(self, abstraction, nn_pkg_list):

        return TorchDataset(abstraction, nn_pkg_list)

    def __init__(self, state_size, action_size):

        super(TorchA2C, self).__init__()

        self.lin1 = nn.Linear(state_size, 64)
        self.v_s = nn.Linear(64, 1)
        self.action = nn.Linear(64, action_size)
#
#         self.action_param_0 = nn.Linear(32, action_param_size)
#         self.action_param_1 = nn.Linear(32, action_param_size)
#         self.action_param_2 = nn.Linear(32, action_param_size)

    def forward(self, x):

        x = F.relu(self.lin1(x))

        return F.softmax(self.action(x), dim=-1), self.v_s(x)

#         return {
#             "v_s": self.v_s(x),
#             "action": F.softmax(self.action(x), dim=-1)
#         }


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(
            m.weight, nonlinearity='relu')


class TorchGenericAction(nn.Module):

    @staticmethod
    def cross_entropy_one_hot(x, y):

        # https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580
        _, y = y.max(dim=-1)
        return torch.nn.CrossEntropyLoss()(x, y)

    @staticmethod
    def create(abstraction, nn_inputs, nn_outputs):

        total_size = 0
        for input_layer in nn_inputs:

            total_size += np.prod(
                abstraction.get_nn_shape(input_layer))

        action_size = abstraction.get_nn_shape("action")

        output_list = [
            ("action", None, TorchGenericAction.cross_entropy_one_hot)]

        num_action_params = abstraction.get_max_action_params()
        action_param_size = None
        for i in range(abstraction.get_max_action_params()):

            name = "action_param_%u" % (i)
            assert name in nn_outputs

            action_param_size = abstraction.get_nn_shape(name)

            output_list.append(
                (name, None, torch.nn.BCEWithLogitsLoss()))

        return TorchGenericAction(total_size, action_size,
                                  num_action_params, action_param_size), \
            output_list, \
            "pytorch"

    def get_dataset(self, nn_inputs, nn_outputs, nn_pkg_list):

        return TorchDataset(nn_inputs, nn_outputs, nn_pkg_list)

    def __init__(self, input_size, action_size, num_action_params,
                 action_param_size):

        super(TorchGenericAction, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.linear_relu_stack.apply(init_weights)

        self.action = nn.Linear(32, action_size)
        self.action.apply(init_weights)

        self.action_params = nn.ModuleList()
        for i in range(num_action_params):

            self.action_params.append(nn.Linear(32, action_param_size))
            self.action_params[i].apply(init_weights)

    def forward(self, x):

        x = self.linear_relu_stack(x)

        return_dict = {
            "action": self.action(x)
        }

        for i in range(len(self.action_params)):

            name = "action_param_%u" % (i)
            return_dict[name] = self.action_params[i](x)

        return return_dict


class TorchGenericQ(nn.Module):

    @staticmethod
    def create(abstraction, nn_inputs, nn_outputs):

        assert nn_outputs == ["q_s_a"]

        total_size = 0
        for input_layer in nn_inputs:

            total_size += np.prod(
                abstraction.get_nn_shape(input_layer))

        return TorchGenericQ(total_size), \
            [("q_s_a", float, torch.nn.MSELoss())], \
            "pytorch"

    def get_dataset(self, nn_inputs, nn_outputs, nn_pkg_list):

        return TorchDataset(nn_inputs, nn_outputs, nn_pkg_list)

    def __init__(self, input_size):

        super(TorchGenericQ, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

        self.linear_relu_stack.apply(init_weights)

    def forward(self, x):

        q_s_a = self.linear_relu_stack(x)
        return {
            "q_s_a": q_s_a
        }
