'''
Created on Apr 4, 2020

@author: rkaria
'''

from matplotlib import pyplot


class BoxPlot:

    def __init__(self, title=None, xlabel=None, ylabel=None):

        self._fig, self._ax = pyplot.subplots()
        pyplot.setp(self._ax.get_xticklabels(), ha="right", rotation=45)

        self._ax.set_title(title)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

        self._x_data = []
        self._y_data = []

    def add_data(self, x_data, y_data):

        self._x_data.append(x_data)
        self._y_data.append(y_data)

    def plot(self):

        self._ax.boxplot(self._y_data, labels=self._x_data, showmeans=True,
                         notch=False)

    def save(self, output_filepath):

        self._fig.savefig(output_filepath)
