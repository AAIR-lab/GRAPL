'''
Created on Apr 5, 2020

@author: rkaria
'''

from matplotlib import pyplot
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm


class LinePlot:

    def __init__(self, title=None, xlabel=None, ylabel=None):

        self._fig, self._ax = pyplot.subplots()
        pyplot.setp(self._ax.get_xticklabels(), ha="right", rotation=45)

        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel

        self._labels = {}
        self._fmt = {}

    def add_data(self, label, x, y, yerr=None, fmt="-o"):

        assert label not in self._labels
        assert isinstance(x, list)
        assert isinstance(y, list)
        assert len(x) == len(y)

        self._labels[label] = x, y, yerr
        self._fmt[label] = fmt

    def _plot_split_labels(self, ignore_yerr=False, ylim=None):

        total_labels = len(self._labels)
        fig = pyplot.figure(figsize=(16, 12), constrained_layout=True)
        gs = GridSpec(total_labels + 1, total_labels, figure=fig)

        base_plot = fig.add_subplot(gs[0:-1, :int(total_labels / 2) + 1])
        base_plot.set_ylabel(self._ylabel)

        aliases = {}
        alias_x = []
        alias_col = []
        i = 0
        for label in self._labels:

            x, y, yerr = self._labels[label]
            for _x in x:

                if _x not in aliases:

                    alias_col.append([_x])
                    alias_x.append(i)
                    aliases[_x] = i
                    i = i + 1

        i = 0
        for label in self._labels:

            x, y, yerr = self._labels[label]
            yerr = None if ignore_yerr else yerr

            base_plot.errorbar(alias_x, y, yerr=yerr, label=label,
                               fmt=self._fmt[label],
                               uplims=True,
                               lolims=True)
            base_plot.legend(loc="lower right")
            base_plot.set_xticks(alias_x)
            base_plot.set_ylim(ylim)

            label_plot = fig.add_subplot(gs[i, int(total_labels / 2) + 1:-1])

            label_plot.errorbar(alias_x, y, yerr=yerr, label=label,
                                fmt=self._fmt[label],
                                uplims=True,
                                lolims=True,
                                color="C%u" % (i))

            label_plot.legend(loc="lower right")
            label_plot.set_xticks(alias_x)
            label_plot.set_ylim(ylim)

            i = i + 1

        table_plot = fig.add_subplot(gs[-1, :-1])
        table_plot.set_axis_off()
        table_plot.table(cellText=alias_col,
                         rowLabels=alias_x,
                         loc='top',
                         cellLoc="left",
                         rowColours=["palegreen"] * len(alias_x),
                         colColours=["palegreen"] * 1,
                         colLabels=["x-axis (%s) alias" % self._xlabel])

        fig.suptitle(self._title)
        self._fig = fig

    def plot(self, ignore_yerr=True, split_labels=False, ylim=None):

        if split_labels:

            self._plot_split_labels(ignore_yerr, ylim)
        else:

            self._ax.set_ylim(ylim)

            for label in self._labels:

                x, y, yerr = self._labels[label]
                yerr = None if ignore_yerr else yerr

                self._ax.errorbar(x, y, yerr=yerr, label=label,
                                  fmt=self._fmt[label],
                                  uplims=True,
                                  lolims=True)

                self._ax.legend(loc="lower right")

            for ax in self._fig.get_axes():
                ax.label_outer()

        self._fig.suptitle(self._title)
        self._fig.align_labels()

    def save(self, output_filepath):

        self._fig.savefig(output_filepath)
