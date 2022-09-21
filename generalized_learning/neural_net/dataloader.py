from tensorflow import keras
from tqdm import tqdm

import numpy as np


class KerasDataset(keras.utils.Sequence):

    @staticmethod
    def filter_data_to_np(layer_names, nn_pkg_list):

        filtered_dict = {}

        for layer_name in layer_names:

            data = map(lambda nn_pkg: nn_pkg.decode(layer_name), nn_pkg_list)
            data = np.asarray(list(data))

            filtered_dict[layer_name] = data

        return filtered_dict

    def __init__(self, abstract_domain, nn_package_list, batch_size, shuffle=True):

        self.nn_input_dict = KerasDataset.filter_data_to_np(
            abstract_domain.get_nn_inputs(), nn_package_list)

        self.nn_label_dict = KerasDataset.filter_data_to_np(
            abstract_domain.get_nn_outputs(), nn_package_list)

        self.batch_size = batch_size
        self.len = len(nn_package_list)

        if shuffle:
            indices = list(
                range(len(self.nn_input_dict[list(self.nn_input_dict)[0]])))
            self.total_length = len(indices)
            np.random.shuffle(indices)
            for k in self.nn_input_dict:
                self.nn_input_dict[k] = self.nn_input_dict[k][indices]
            for k in self.nn_label_dict:
                self.nn_label_dict[k] = self.nn_label_dict[k][indices]

    def __len__(self):
        return int(np.ceil(self.len / float(self.batch_size)))

    def __getitem__(self, idx):
        x, y = {}, {}
        for k in self.nn_input_dict:
            x[k] = self.nn_input_dict[k][idx *
                                         self.batch_size:(idx + 1) * self.batch_size]
        for k in self.nn_label_dict:
            y[k] = self.nn_label_dict[k][idx *
                                         self.batch_size:(idx + 1) * self.batch_size]
        return x, y


def get_loader(abstract_domain, nn_package_list, batch_size, arch,
               shuffle=True):

    dataloader = None

    if arch == 'keras':
        dataloader = KerasDataset(
            abstract_domain, nn_package_list, batch_size, shuffle)
    else:

        raise Exception("Unsupported neural net architecture.")

    return dataloader
