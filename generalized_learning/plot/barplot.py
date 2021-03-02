'''
Created on Apr 5, 2020

@author: rkaria
'''

from matplotlib import pyplot
import numpy as np


class BarPlot:

    _DEFAULT_WIDTH = 0.25

    def __init__(self, title=None, xlabel=None, ylabel=None):

        self._fig, self._ax = pyplot.subplots()
        pyplot.setp(self._ax.get_xticklabels(), ha="right", rotation=45)

        self._ax.set_title(title)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

        self._labels = {}
        self._colors = {}

        self._width = BarPlot._DEFAULT_WIDTH

    def add_data(self, label, x, y, yerr=None, color=None):

        try:

            _x, _y, _yerr = self._labels[label]
        except KeyError:

            _x, _y, _yerr = [], [], []
            self._labels[label] = _x, _y, _yerr

        self._colors[label] = color
        _x.append(x)
        _y.append(y)
        _yerr.append(yerr)

    def add_range(self, label, x, y, yerr=None, color=None):

        assert label not in self._labels
        assert isinstance(x, list)
        assert isinstance(y, list)
        assert yerr is None or isinstance(yerr, list)

        self._colors[label] = color
        self._labels[label] = x, y, yerr

    def _plot_stacked(self):

        bottom = None

        for label in self._labels:

            x, y, yerr = self._labels[label]
            self._ax.bar(x, y, bottom=bottom, label=label,
                         color=self._colors[label], width=self._width)
            bottom = y

    def _plot_grouped(self):

        # Must have at least one label.
        assert len(self._labels) >= 1

        label = next(iter(self._labels))

        # First we need all the x-points.
        x = set()
        for label in self._labels:

            _x, _y, _yerr = self._labels[label]
            x.update(_x)

        # Convert it into an ordered list.
        x = sorted(x)
        group_size = len(x)
        label_locations = np.arange(group_size)
        label_width = self._width / group_size

        for label in self._labels:

            _x, _y, _yerr = self._labels[label]

            y = []
            yerr = []
            for i in range(len(x)):

                try:

                    index = _x.index(x[i])
                    y.append(_y[index])
                    yerr.append(_yerr[index])
                except ValueError:

                    y.append(0)
                    yerr.append(0)

            self._ax.bar(label_locations, y, label=label, yerr=yerr,
                         color=self._colors[label], width=label_width)

            # Update the label_locations for the next value.
            label_locations = label_locations + label_width

        self._ax.set_xticks(np.arange(group_size) + self._width)
        self._ax.set_xticklabels(np.arange(group_size), rotation=45)

    def plot(self, stacked=True):

        if stacked:

            self._plot_stacked()
        else:

            self._plot_grouped()

        self._ax.legend(loc="upper right")

    def save(self, output_filepath):

        self._fig.savefig(output_filepath)
