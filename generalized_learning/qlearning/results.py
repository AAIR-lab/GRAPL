'''
Created on Aug 2, 2021

@author: anonymous
'''

import ast
import csv
import os
import pathlib
import statistics

import matplotlib.pyplot as plt
import numpy as np


class CSVResults:

    def __init__(self, input_dir, fieldnames, output_prefix,
                 write_mode="a"):

        self.output_prefix = output_prefix
        self.output_filepath = "%s/%s.csv" % (input_dir, self.output_prefix)
        self._file_handle = open(self.output_filepath,
                                 write_mode)

        self.fieldnames = fieldnames

        self._csv_writer = csv.DictWriter(
            self._file_handle,
            fieldnames=fieldnames)

        # Only write the header once.
        first_line = open(self.output_filepath, "r").readline()

        if write_mode == "w" or first_line == "":

            self._csv_writer.writeheader()

    def add_data(self, data):

        self._csv_writer.writerow(data)

    def close(self):

        self._file_handle.close()


class QLearningResults:

    @staticmethod
    def _plot(data, ax, x_label, y_label, x_func, y_func):

        # Don't support multiple problem names yet since the only consumer is
        # the external gridspec script.
        assert len(data) == 1

        for dict_key in data:

            problem_name = dict_key
            problem_data = data[problem_name]

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            for experiment_name in problem_data:

                experiment_data = sorted(problem_data[experiment_name],
                                         key=x_func)
                x = [x_func(x) for x in experiment_data]
                y = [y_func(x) for x in experiment_data]

                new_x, y_mean, y_stdev = QLearningResults.get_mean(x, y)

                p = ax.plot(new_x, y_mean, label=experiment_name)

                ax.fill_between(new_x, y_mean - y_stdev,
                                y_mean + y_stdev,
                                color=p[-1].get_color(),
                                alpha=0.2)

    @staticmethod
    def get_mean(x, y):

        new_x = []
        y_mean = []
        y_stdev = []

        prev_x = x[0]
        entries = [y[0]]
        for i in range(1, len(x)):

            if prev_x == x[i]:

                entries.append(y[i])
                if i < len(x) - 1:

                    continue

            new_x.append(prev_x)
            y_mean.append(statistics.mean(entries))

            if len(entries) == 1:

                y_stdev.append(0)
            else:
                y_stdev.append(statistics.stdev(entries))

            prev_x = x[i]
            entries = [y[i]]

        return np.array(new_x), np.array(y_mean), np.array(y_stdev)

    @staticmethod
    def _add_data(data_dict, problem_name, experiment_name, data):

        dict_key = problem_name

        problem_data = data_dict.setdefault(dict_key, {})
        data_list = problem_data.setdefault(experiment_name, [])

        data_list.append(data)

    @staticmethod
    def read_data(filepath, fieldnames):

        file_handle = open(filepath, "r")

        data_dict = {}

        reader = csv.DictReader(file_handle)
        for row in reader:

            problem_name = row["problem_name"]
            experiment_name = row["experiment_name"]
            data = ast.literal_eval(row[fieldnames[-1]])

            QLearningResults._add_data(data_dict, problem_name,
                                       experiment_name, data)

        return data_dict

    def __init__(self, input_dir, fieldnames, output_prefix,
                 write_mode="a"):

        self.output_prefix = output_prefix
        output_filepath = "%s/%s.csv" % (input_dir, self.output_prefix)
        self._file_handle = open(output_filepath,
                                 write_mode)

        self.fieldnames = fieldnames

        self._csv_writer = csv.DictWriter(
            self._file_handle,
            fieldnames=fieldnames)

        # Only write the header once.
        first_line = open(output_filepath, "r").readline()

        if write_mode == "w" or first_line == "":

            self._csv_writer.writeheader()

        self.data = QLearningResults.read_data(output_filepath,
                                               self.fieldnames)

    def write_data(self, problem_name, experiment_name, data):

        self._csv_writer.writerow({
            "problem_name": problem_name,
            "experiment_name": experiment_name,
            self.fieldnames[-1]: data
        })

    def add_data(self, problem, experiment_name, data):

        problem_filepath = pathlib.Path(problem.get_problem_filepath())

        QLearningResults._add_data(self.data, problem_filepath.name,
                                   experiment_name, data)

        self.write_data(problem_filepath.name, experiment_name, data)

        # Immediately flush.
        self._file_handle.flush()
        os.fsync(self._file_handle)

    def plot(self, output_dir, x_label, y_label, x_func, y_func,
             output_prefix=None):

        if output_prefix is None:

            output_prefix = self.output_prefix

        for dict_key in self.data:

            problem_name = dict_key
            problem_data = self.data[problem_name]

            fig, ax = plt.subplots()

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            for experiment_name in problem_data:

                experiment_data = sorted(problem_data[experiment_name],
                                         key=x_func)
                x = [x_func(x) for x in experiment_data]
                y = [y_func(x) for x in experiment_data]

                new_x, y_mean, y_stdev = QLearningResults.get_mean(x, y)

                p = ax.plot(new_x, y_mean, label=experiment_name)

                ax.fill_between(new_x, y_mean - y_stdev,
                                y_mean + y_stdev,
                                color=p[-1].get_color(),
                                alpha=0.2)

            ax.legend()

            if len(self.data) > 0:

                output_filepath = "%s/%s.%s.png" % (output_dir,
                                                    output_prefix,
                                                    problem_name)
                fig.savefig(output_filepath)
