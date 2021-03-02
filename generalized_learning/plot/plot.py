'''
Created on Mar 31, 2020

@author: rkaria
'''


import itertools
import logging
import math
import multiprocessing
import statistics

from tinydb import Query
from tinydb import TinyDB
from tinydb.storages import MemoryStorage

from concretized.solution import Solution
from plot.barplot import BarPlot
from plot.bins import Bins
from plot.boxplot import BoxPlot
from plot.cdp import CDP
from plot.cdp import CDPManager
from plot.lineplot import LinePlot
from plot.series import Series
from util import constants
from util import executor
from util import file
from util.phase import Phase


logger = logging.getLogger(__name__)


class Plot(Phase):

    REQUIRED_KEYS = set(["input_dir", "num_bins"]).union(Phase.REQUIRED_KEYS)

    #: The default phase dict for this phase.
    DEFAULT_PHASE_DICT = {

        **Phase.DEFAULT_PHASE_DICT,

        "use_mpi": False,
        "max_workers": multiprocessing.cpu_count(),
        "chunk_size": 25,
        "force_single_core": False
    }

    @staticmethod
    def get_instance(parent, parent_dir, global_dict, user_phase_dict,
                     failfast):

        return Plot(parent, parent_dir, global_dict, user_phase_dict,
                    failfast)

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(Plot, self).__init__(parent, parent_dir, global_dict,
                                   user_phase_dict, failfast)

    def _generate_args(self, chunk_size, solutions):

        assert chunk_size > 0

        total_problems = len(solutions)
        assert total_problems > 0

        total_chunks = math.ceil(total_problems / chunk_size)
        logger.debug("Generating total_chunks=%u" % (total_chunks))

        for chunk_no in range(total_chunks):

            start = chunk_no * chunk_size
            end = min(total_problems, start + chunk_size)

            yield (solutions[start:end], )

    def _extract_properties(self, solutions):

        properties = []

        for solution in solutions:

            properties.append(Solution.parse(solution).get_properties())

        return properties

    def get_bin_plots(self, prefix, titleprefix, bins, nodes_expanded_dict,
                      plan_length_dict, solved_dict):

        ne_lineplot = LinePlot(title="%s.%s.nodes_expanded" % (prefix, titleprefix),
                               xlabel="bins",
                               ylabel="avg. nodes expanded")

        pl_lineplot = LinePlot(title="%s.%s.plan_length" % (prefix, titleprefix),
                               xlabel="bins",
                               ylabel="avg. plan length")

        solved_lineplot = LinePlot(title="%s.%s.solved" % (prefix, titleprefix),
                                   xlabel="bins",
                                   ylabel="% succeeded")

        solution_names = solved_dict.keys()

        x = []
        for i in range(len(bins)):

            x.append(bins.get_bin_name(i))

        for solution_name in solution_names:

            y = []
            for _x in x:

                _y = solved_dict[solution_name].get_y(_x)
                total_failed = len(list(itertools.filterfalse(None, _y)))
                total_succeeded = len(_y) - total_failed
                percent_succeeded = (total_succeeded * 100.0) / len(_y)

                y.append(percent_succeeded)

            solved_lineplot.add_data(solution_name, x, y)

            if solution_name in nodes_expanded_dict:

                nodes_expanded_series = nodes_expanded_dict[solution_name]

                # If the solution_name is in the nodes_expanded, then by
                # default it must also have an associated entry in the plan
                # length.
                plan_length_series = plan_length_dict[solution_name]

                ne_u = []
                ne_stdev = []
                pl_u = []
                pl_stdev = []
                for _x in x:

                    try:
                        ne_y = nodes_expanded_series.get_y(_x)
                        pl_y = plan_length_series.get_y(_x)

                        ne_u.append(statistics.mean(ne_y))
                        pl_u.append(statistics.mean(pl_y))

                        # ne_u[-1], pl_u[-1] is the mean that we computed
                        # corresponding to x.
                        #
                        # Use pstdev() instead of stdev() since the former can
                        # work with just one data point (which is always
                        # guaranteed when executing this code).
                        ne_stdev.append(statistics.pstdev(ne_y, mu=ne_u[-1]))
                        pl_stdev.append(statistics.pstdev(pl_y, mu=pl_u[-1]))
                    except KeyError:

                        # Fill missing data with zeroes.
                        ne_u.append(0)
                        pl_u.append(0)
                        ne_stdev.append(0)
                        pl_stdev.append(0)
                        pass

                # The x[], y[] data is generated.
                ne_lineplot.add_data(solution_name, x, ne_u, yerr=ne_stdev)
                pl_lineplot.add_data(solution_name, x, pl_u, yerr=pl_stdev)

        ne_lineplot.plot(split_labels=True)
        pl_lineplot.plot(split_labels=True)
        solved_lineplot.plot(split_labels=True)

        return ne_lineplot, pl_lineplot, solved_lineplot

    def get_bin_reference_plots(self, prefix, titleprefix, bins, nodes_expanded_dict,
                                plan_length_dict, solved_dict, ref_source="ff"):

        ne_lineplot = LinePlot(title="%s.%s.ref.nodes_expanded" % (prefix, titleprefix),
                               xlabel="bins",
                               ylabel="avg. nodes expanded")

        pl_lineplot = LinePlot(title="%s.%s.ref.plan_length" % (prefix, titleprefix),
                               xlabel="bins",
                               ylabel="avg. plan length")

        solution_names = solved_dict.keys()

        x = []
        for i in range(len(bins)):

            x.append(bins.get_bin_name(i))

        for solution_name in solution_names:

            if solution_name in nodes_expanded_dict:

                nodes_expanded_series = nodes_expanded_dict[solution_name]

                # If the solution_name is in the nodes_expanded, then by
                # default it must also have an associated entry in the plan
                # length.
                plan_length_series = plan_length_dict[solution_name]

                diff_ne = []
                diff_pl = []
                for _x in x:

                    try:

                        ref_ne = statistics.mean(
                            nodes_expanded_dict[ref_source].get_y(_x))
                        ref_pl = statistics.mean(
                            plan_length_dict[ref_source].get_y(_x))

                        ne = statistics.mean(nodes_expanded_series.get_y(_x))
                        pl = statistics.mean(plan_length_series.get_y(_x))

                        diff_ne.append(((ne - ref_ne) * 100.0) / ref_ne)
                        diff_pl.append(((pl - ref_pl) * 100.0) / ref_pl)

                    except KeyError:

                        diff_ne.append(200.0)
                        diff_pl.append(200.0)
                        pass

                # The x[], y[] data is generated.
                ne_lineplot.add_data(solution_name, x, diff_ne)
                pl_lineplot.add_data(solution_name, x, diff_pl)

        ne_lineplot.plot(split_labels=True, ylim=(-50, 75))
        pl_lineplot.plot(split_labels=True, ylim=(-50, 75))

        return ne_lineplot, pl_lineplot

    def _plot_bins(self, prefix, fileprefix, titleprefix, output_dir, bins,
                   nodes_expanded_dict, plan_length_dict, solved_dict):

        ne_lineplot, pl_lineplot, solved_lineplot = self.get_bin_plots(
            prefix, titleprefix,
            bins,
            nodes_expanded_dict,
            plan_length_dict,
            solved_dict)

        ne_lineplot.save("%s/%s.%s.nodes_expanded.png" % (output_dir,
                                                          prefix,
                                                          fileprefix))

        pl_lineplot.save("%s/%s.%s.plan_length.png" % (output_dir,
                                                       prefix,
                                                       fileprefix))

        solved_lineplot.save("%s/%s.%s.solved.png" % (output_dir,
                                                      prefix,
                                                      fileprefix))

        ne_lineplot, pl_lineplot = self.get_bin_reference_plots(
            prefix, titleprefix,
            bins,
            nodes_expanded_dict,
            plan_length_dict,
            solved_dict)

        ne_lineplot.save("%s/%s.%s.ref.nodes_expanded.png" % (output_dir,
                                                              prefix,
                                                              fileprefix))

        pl_lineplot.save("%s/%s.%s.ref.plan_length.png" % (output_dir,
                                                           prefix,
                                                           fileprefix))

    def get_box_bar_plots(self, prefix, titleprefix, nodes_expanded,
                          plan_length, solved):

        # The solved series contains all solution names since no matter pass or
        # fail, that information is always stored.
        solution_names = sorted(solved.get_x())

        nodes_expanded_str = "%s.%s.nodes_expanded" % (prefix, titleprefix)
        ne_boxplot = BoxPlot(title=nodes_expanded_str,
                             xlabel="algorithm",
                             ylabel="nodes expanded")

        plan_length_str = "%s.%s.plan_length" % (prefix, titleprefix)
        pl_boxplot = BoxPlot(title=plan_length_str,
                             xlabel="algorithm",
                             ylabel="plan length")

        solved_str = "%s.%s.solved" % (prefix, titleprefix)
        solved_barplot = BarPlot(title=solved_str,
                                 xlabel="algorithm",
                                 ylabel="success/failure")

        for solution_name in solution_names:

            if solution_name in nodes_expanded.get_x():

                ne_boxplot.add_data(solution_name,
                                    nodes_expanded.get_y(solution_name))

                # If the solution_name is in the nodes_expanded, then by
                # default it must also have an associated entry in the plan
                # length.
                pl_boxplot.add_data(solution_name,
                                    plan_length.get_y(solution_name))

            solved_data = solved.get_y(solution_name)
            total_solved = len(solved_data)
            total_failed = len(list(itertools.filterfalse(None, solved_data)))

            solved_barplot.add_data("failure", solution_name, total_failed,
                                    "r")
            solved_barplot.add_data("success", solution_name,
                                    total_solved - total_failed,
                                    "b")

        ne_boxplot.plot()
        pl_boxplot.plot()
        solved_barplot.plot()

        return ne_boxplot, pl_boxplot, solved_barplot

    def _plot_box_bar(self, prefix, fileprefix, titleprefix, output_dir,
                      nodes_expanded, plan_length, solved):

        ne_boxplot, pl_boxplot, solved_barplot = self.get_box_bar_plots(
            prefix,
            titleprefix,
            nodes_expanded,
            plan_length,
            solved)

        nodes_expanded_str = "%s.%s.nodes_expanded" % (prefix, fileprefix)
        plan_length_str = "%s.%s.plan_length" % (prefix, fileprefix)
        solved_str = "%s.%s.solved" % (prefix, fileprefix)

        ne_boxplot.save("%s/%s.png" % (output_dir, nodes_expanded_str))
        pl_boxplot.save("%s/%s.png" % (output_dir, plan_length_str))
        solved_barplot.save("%s/%s.png" % (output_dir, solved_str))

    def get_documents(self, input_dir):

        # Get the solution file list.
        solutions = file.get_file_list(input_dir,
                                       constants.SOLUTION_FILE_REGEX)

        # Get the list of solution properties.
        if self.get_value("force_single_core"):

            properties = executor.singlecore_execute(self._extract_properties,
                                                     (solutions))
        else:

            properties = executor.multicore_execute(
                self._extract_properties,
                (solutions, ),
                self._generate_args,
                self.get_value("max_workers"),
                self.get_value("chunk_size"),
                self.get_value("use_mpi"))

        # Create the database.
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple(properties)

        # Try applying any specified filter.
        try:

            query_str = self.get_value("filter")
            query = Query()
            (query)
            documents = db.search(eval(query_str))

        except KeyError:

            # If no filter, then all documents are a part of the data.
            documents = db.all()

        return documents

    def execute_main(self, input_dir):

        domains = ["blocksworld", "childsnack", "ferry", "goldminer",
                   "spanner", "miconic", "visitall", "logistics"]

        cdp_manager = CDPManager()
        for domain in domains:

            data_dir = "%s/%s/test_data" % (input_dir, domain)
            documents = self.get_documents(data_dir)

            for document in documents:

                cdp_manager.add_data(document)

        cdp_manager.plot(input_dir)

    def execute(self):

        # Get the documents.
        input_dir = file.get_relative_path(self.get_value("input_dir"),
                                           self._parent_dir)

        # self.execute_main(input_dir)

        documents = self.get_documents(input_dir)

        # Plot the cumulative plots.
        cdp = CDP()
        for document in documents:

            cdp.add_data(document)

        cdp.plot_data(input_dir)

        # Get the generic test set data.
        nodes_expanded_series, plan_length_series, solved_series, domain, \
            problem_params = Plot.get_wholistic_data(documents)

        # Create the bins.
        bins = Bins(problem_params,
                    self.get_value("num_bins"),
                    self._phase_dict.get("bin_filters", {}))

        bins_nodes_expanded, bins_plan_length, bins_solved, \
            solutions_nodes_expanded, solutions_plan_length, solutions_solved = \
            Plot.get_problem_dependent_data(bins, documents)

        # Plot the problem size dependent data.
        self._plot_box_bar(domain, "aggregate", "aggregate", input_dir,
                           nodes_expanded_series,
                           plan_length_series, solved_series)

        for bin_name in bins_nodes_expanded:

            bin_index = bins.get_index_from_name(bin_name)
            self._plot_box_bar(domain, bin_index, bin_name, input_dir,
                               bins_nodes_expanded[bin_name],
                               bins_plan_length[bin_name],
                               bins_solved[bin_name])

        self._plot_bins(domain, "avg", "avg", input_dir,
                        bins,
                        solutions_nodes_expanded,
                        solutions_plan_length,
                        solutions_solved)

    @staticmethod
    def get_problem_dependent_data(bins, documents):

        bins_nodes_expanded = {}
        bins_plan_length = {}
        bins_solved = {}

        solutions_nodes_expanded = {}
        solutions_plan_length = {}
        solutions_solved = {}

        for document in documents:

            bin_index = bins.get_bin_index(document)
            if bin_index == float("inf"):

                continue

            # Get the keys.
            bin_name = bins.get_bin_name(bin_index)
            solution_name = document["solution"]["name"]

            # Get the series.
            bin_nodes_expanded = bins_nodes_expanded.setdefault(
                bin_name,
                Series())
            bin_plan_length = bins_plan_length.setdefault(
                bin_name,
                Series())
            bin_solved = bins_solved.setdefault(
                bin_name,
                Series())

            solution_nodes_expanded = solutions_nodes_expanded.setdefault(
                solution_name,
                Series())
            solution_plan_length = solutions_plan_length.setdefault(
                solution_name,
                Series())
            solution_solved = solutions_solved.setdefault(
                solution_name,
                Series())

            if document["solution"]["is_plan_found"]:

                # Add the data point to the nodes expanded, plan_length iff the
                # instance was solved.
                nodes_expanded = document["solution"]["nodes_expanded"]
                plan_length = document["solution"]["plan_length"]

                bin_nodes_expanded.add_point(solution_name, nodes_expanded)
                bin_plan_length.add_point(solution_name, plan_length)
                bin_solved.add_point(solution_name, True)

                solution_nodes_expanded.add_point(bin_name, nodes_expanded)
                solution_plan_length.add_point(bin_name, plan_length)
                solution_solved.add_point(bin_name, True)
            else:

                bin_solved.add_point(solution_name, False)
                solution_solved.add_point(bin_name, False)

        return bins_nodes_expanded, bins_plan_length, bins_solved, \
            solutions_nodes_expanded, solutions_plan_length, solutions_solved

    @staticmethod
    def get_wholistic_data(documents):

        # The plots for the wholistic view of the instance data.
        # The x-axis is the solution name used to solve them.
        nodes_expanded_series = Series()
        plan_length_series = Series()
        solved_series = Series()

        domains = set()

        # The discovered problem params.
        problem_params = Series()

        # First pass, get data for the nodes expanded, plan length and the solved
        # instances.
        for document in documents:

            solution_name = document["solution"]["name"]

            # Ensure only one domain.
            domain = document["problem"]["domain"]
            domains.add(domain)
            assert len(domains) == 1

            if document["solution"]["is_plan_found"]:

                solved_series.add_point(solution_name, True)

                # Add the data point to the nodes expanded, plan_length iff the
                # instance was solved.
                nodes_expanded = document["solution"]["nodes_expanded"]
                plan_length = document["solution"]["plan_length"]

                nodes_expanded_series.add_point(solution_name, nodes_expanded)
                plan_length_series.add_point(solution_name, plan_length)
            else:

                solved_series.add_point(solution_name, False)

            # Also, add values for the problem params.
            for problem_param in document["problem"]["bin_params"]:

                problem_params.add_point(problem_param,
                                         document["problem"][problem_param])

        return nodes_expanded_series, \
            plan_length_series, \
            solved_series, \
            next(iter(domains)), \
            problem_params
