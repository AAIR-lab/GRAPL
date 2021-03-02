'''
Created on Sep 13, 2020

@author: rkaria
'''

from matplotlib import pyplot
from matplotlib.gridspec import GridSpec


class CDPManager:

    def __init__(self):

        self.domain_dict = {}

    def add_data(self, document):

        domain = document["problem"]["domain"]

        cdp = self.domain_dict.setdefault(domain, CDP())
        cdp.add_data(document)

    def plot(self, output_dir):

        fig = pyplot.figure(figsize=(8, 6), constrained_layout=True)
        gs = GridSpec(4, 4, figure=fig)

        row = 0
        column = 0
        for domain in self.domain_dict:

            cdp = self.domain_dict[domain]
            ne_axis = fig.add_subplot(gs[row, column])
            tt_axis = fig.add_subplot(gs[row + 1, column])

            ne_axis.set(title=domain)

            if column != 0:

                ne_axis.set(yticks=[])
                tt_axis.set(yticks=[])

            cdp.plot_nodes_expanded(ne_axis)
            cdp.plot_time_taken(tt_axis)

            column += 1
            if column == 4:

                row += 2
                column = 0

        fig.savefig("%s/cdp.png" % (output_dir))


class CDP:

    def __init__(self):

        self.solution_dict = {}
        self.total_problems = set()
        self.domain = None
        self.ignore_list = set(["fd_astar"])
        self.max_nodes_expanded = float("-inf")
        self.max_time_taken = float("-inf")

    def add_data(self, document):

        solution_name = document["solution"]["name"]
        problem_name = document["problem"]["filepath"]

        if self.domain is None:

            self.domain = document["problem"]["domain"]
        else:

            assert self.domain == document["problem"]["domain"]

        self.total_problems.add(problem_name)

        if document["solution"]["is_plan_found"] \
                and solution_name not in self.ignore_list:

            nodes_expanded, time_taken = self.solution_dict.setdefault(
                solution_name,
                ([], []))

            ne = int(document["solution"]["nodes_expanded"])
            tt = round(float(document["solution"]["time_taken"]), 2)

            self.max_nodes_expanded = max(ne, self.max_nodes_expanded)
            self.max_time_taken = max(tt, self.max_time_taken)

            nodes_expanded.append(ne)
            time_taken.append(tt)

    def _compute_cumulative_data(self, data_list):

        x = []
        y = []
        total = 0

        total_problems = len(self.total_problems)
        assert total_problems > 0

        data_list = sorted(data_list)
        for data in data_list:

            total += 1
            percent = round((total * 100.0) / total_problems, 2)

            y.append(percent)
            x.append(data)

        return x, y

    def plot_nodes_expanded(self, plot_axis):

        for solution_name in self.solution_dict:

            nodes_expanded, _ = self.solution_dict[solution_name]

            x, y = self._compute_cumulative_data(nodes_expanded)
            x.append(self.max_nodes_expanded)
            y.append(max(y))
            plot_axis.plot(x, y, label=solution_name)

    def plot_time_taken(self, plot_axis):

        for solution_name in self.solution_dict:

            _, time_taken = self.solution_dict[solution_name]

            x, y = self._compute_cumulative_data(time_taken)
            x.append(self.max_time_taken)
            y.append(max(y))
            plot_axis.plot(x, y, label=solution_name)

    def plot_data(self, output_dir):

        ne_fig, ne_ax = pyplot.subplots()
        tt_fig, tt_ax = pyplot.subplots()

        ne_ax.set_xlabel("Nodes expanded")
        ne_ax.set_ylabel("Percent solved")

        tt_ax.set_xlabel("Time taken")
        tt_ax.set_ylabel("Percent solved")

        for solution_name in self.solution_dict:

            nodes_expanded, time_taken = self.solution_dict[solution_name]

            x, y = self._compute_cumulative_data(nodes_expanded)
            x.append(self.max_nodes_expanded)
            y.append(max(y))
            ne_ax.plot(x, y, label=solution_name)

            x, y = self._compute_cumulative_data(time_taken)
            x.append(self.max_time_taken)
            y.append(max(y))
            tt_ax.plot(x, y, label=solution_name)

        ne_ax.legend()
        tt_ax.legend()

        ne_fig.savefig("%s/%s.cdp.nodes_expanded.png" %
                       (output_dir, self.domain))
        tt_fig.savefig("%s/%s.cdp.time_taken.png" % (output_dir, self.domain))
