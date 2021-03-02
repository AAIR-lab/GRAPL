from tensorflow import keras
from tqdm import tqdm

import numpy as np


class KerasDataset(keras.utils.Sequence):

    def __init__(self, abstract_domain, nn_package_list, batch_size, shuffle=True):
        self.nn_input_dict, self.nn_label_dict = abstract_domain.get_nn_train_dict(
            nn_package_list)
        self.batch_size = batch_size
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
        return int(np.ceil(len(self.nn_input_dict['state_unary_preds']) / float(self.batch_size)))

    def __getitem__(self, idx):
        x, y = {}, {}
        for k in self.nn_input_dict:
            x[k] = self.nn_input_dict[k][idx *
                                         self.batch_size:(idx + 1) * self.batch_size]
        for k in self.nn_label_dict:
            y[k] = self.nn_label_dict[k][idx *
                                         self.batch_size:(idx + 1) * self.batch_size]
        return x, y


def get_loader(abstract_domain, nn_package_list, batch_size, arch, root="/tmp", shuffle=True, phase="train"):

    dataloader = None

    if arch == 'keras':
        dataloader = KerasDataset(
            abstract_domain, nn_package_list, batch_size, shuffle)
    else:

        raise Exception("Unsupported neural net architecture.")

    return dataloader
