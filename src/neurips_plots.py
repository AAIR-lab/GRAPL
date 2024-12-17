'''
Created on Jan 18, 2023

@author: rkaria
'''

import matplotlib
import csv
import ast
import matplotlib
import matplotlib.pyplot as plt
import statistics
import numpy as np
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "text.usetex": True,
})

plt.rcParams["mathtext.fontset"]  = "dejavuserif"
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })

def get_interaction_adjustment(key, alg, value, prev_value):
    if "Interaction" in key:

        if "GLIB" in alg:
            value = 1 if value == 0 else value
        # elif value == 0:
        #     value = prev_value + 1
        #     return value, value
        # else:
        #     value = prev_value + 5
        #     return value, value

    return value, prev_value


def get_data(filepath, xkey, ykey, alg, metric):
    fh = open(filepath, "r")
    reader = csv.DictReader(fh, delimiter=";")

    x = []
    y = []
    ipml_sample = 0

    for row in reader:

        x_value = ast.literal_eval(row[xkey])
        y_value = ast.literal_eval(row[ykey])
        if isinstance(y_value, tuple):
            y_value = y_value[0]

        x_value, ipml_sample = get_interaction_adjustment(xkey,
                                                          alg,
                                                          x_value,
                                                          ipml_sample)

        y_value, ipml_sample = get_interaction_adjustment(ykey,
                                                          alg,
                                                          y_value,
                                                          ipml_sample)

        x.append(x_value)
        y.append(y_value)

    return x, y


def convert_to_bins(num_bins, bin_size, x, y):
    new_x = [i for i in range(num_bins)]
    new_y = [[] for i in range(num_bins)]

    for i in range(len(y)):

        bin_no = int(x[i] // bin_size)

        if bin_no >= num_bins:
            bin_no = num_bins - 1

        if new_y[bin_no] == 0:
            new_y[bin_no] = []

        new_y[bin_no].append(y[i])

    return new_x, new_y


def average_out(x):
    for i in range(len(x)):

        if len(x[i]) > 0:
            x[i] = statistics.mean(x[i])
        else:
            x[i] = float("inf")

    return x


def sum_out(x):
    for i in range(len(x)):

        if len(x[i]) > 0:
            x[i] = sum(x[i])
        else:
            x[i] = float("inf")

    return x


def no_change(x, default_value):
    x[0] = default_value if x[0] == float("inf") else x[0]
    for i in range(1, len(x)):

        if x[i] == float("inf"):
            x[i] = x[i - 1]

    return x


def max_out(x):
    for i in range(len(x)):

        if len(x[i]) > 0:
            x[i] = max(x[i])
        else:
            x[i] = float("inf")

    return x


def plot_vd(filename, ax, label, color):
    pass


def cumulative(x):
    for i in range(1, len(x)):
        x[i] += x[i - 1]


def compute_max_for_matric(data_dict, algs, metric, runs):
    x = []
    runs_x = []
    for run_no in runs:
        for alg in algs:
            x += data_dict[alg][metric][run_no]
        runs_x.append(max(x))

    return statistics.mean(runs_x)


def organize_into_bins(bin_data, algs, x_metric, metric, data, runs,
                       bin_size, num_bins):
    for run_no in runs:
        for alg in algs:

            data_points = data[alg][metric][run_no]
            x_points = data[alg][x_metric][run_no]
            for i in range(len(x_points)):

                data_point = data_points[i]
                x_point = x_points[i]

                bin_no = int(x_point // bin_size)
                if bin_no >= num_bins:
                    continue

                bin_data[alg][metric]["data"][bin_no].append(data_point)


def standardize(bin_data, algs, metric, data):

    for alg in algs:

        x = bin_data[alg][metric]["data"]
        stds = bin_data[alg][metric]["std"]
        for bin_no in range(len(x)):

            data_points = x[bin_no]

            if len(data_points) == 0:
                mean = float("inf")
                std = float("inf")
            else:
                mean = statistics.mean(data_points)
                std = statistics.stdev(data_points) if len(data_points) > 1 else 0

            x[bin_no] = mean
            stds[bin_no] = std


def get_last_range(x):
    idx = 0
    for i in range(len(x)):

        if x[i] != float("inf"):
            idx = i + 1

    return idx


def fill_gaps(x, default_value):
    idx = len(x) - 1
    while x[idx] == float("inf") \
            and idx >= 0:
        idx = idx - 1

    if idx < 0:
        return

    value = x[idx]
    for i in range(idx - 1, -1, -1):

        if x[i] == float("inf"):
            x[i] = value
        else:
            value = x[i]


def get_filename(domain, alg):
    if alg == "IPML-R":

        suffix = "ipml_randomized.csv"
    elif alg == "IPML-S":

        suffix = "ipml_sequential.csv"
    elif alg == "GLIB_G1":

        suffix = "glib_g1.csv"
    elif alg == "GLIB_L2":

        suffix = "glib_l2.csv"
    else:
        assert False

    return "%s_%s" % (domain.lower(), suffix)


def write_file(output_dir, domain, bin_data, algs, x_metric, metrics, num_bins):
    for alg in algs:
        filepath = output_dir + "/" + get_filename(domain, alg)
        fh = open(filepath, "w")

        for i in range(num_bins):

            string = "%s" % (bin_data[alg][x_metric][i])
            for metric in metrics:
                string += ",%s" % (bin_data[alg][metric]["data"][i])
                string += ",%s" % (bin_data[alg][metric]["std"][i])

            string += "\n"
            fh.write(string)

        fh.close()


def get_title_for_domain(domain):
    if domain == "Tireworld":

        return "Driver Agent"
    elif domain == "Explodingblocks":

        return "Warehouse Robot"
    elif domain == "Probabilistic_elevators":

        return "Elevator Control Agent"
    elif domain == "First_responders":

        return "First Responder Robot"
    else:

        assert False


def get_ground_truth_cost_sr_for_domain(domain):
    if domain == "Tireworld":

        return (7.725, 1.0)
    elif domain == "Explodingblocks":

        return (12.25, 0.92)
    elif domain == "Probabilistic_elevators":

        return (18.75, 1.0)
    elif domain == "First_responders":

        return (9.35, 1.0)
    else:

        assert False


def plot_supplemental(results_dir, domains, domain_data, num_bins):
    fig = plt.figure(constrained_layout=False, figsize=(20, 4))
    gs = gridspec.GridSpec(ncols=4, nrows=1, figure=fig, wspace=0.04)

    metrics = ["Variational Difference (Samples)"]

    PLOTTERS = [
        ("IPML-R", "IPML-R (Ours)", "blue", "-"),
        # ("IPML-S", "IPML-S (Ours)", "green", "-."),
        ("GLIB_G1", "GLIB (G1)", "black", "--"),
        ("GLIB_L2", "GLIB (L2)", "red", ":"),
    ]

    TITLE_FONTSIZE = 24
    YLABEL_FONTSIZE = 24
    XLABEL_FONTSIZE = 24
    LEGEND_FONTSIZE = 20

    XTICKLABEL_FONTSIZE = 16
    YTICKLABEL_FONTSIZE = 16

    for column, domain in enumerate(domains):

        title = get_title_for_domain(domain)
        r2 = fig.add_subplot(gs[0, column])

        axes = [r2]

        r2.set_title(title, fontsize=TITLE_FONTSIZE)

        if column == 0:

            r2.set_ylabel("Variational Distance", fontsize=YLABEL_FONTSIZE)

        else:
            # pass
            for ax in axes:
                ax.set_yticklabels([])

        bin_data = domain_data[domain]

        for ax, metric in zip(axes, metrics):

            for alg, label, color, linestyle in PLOTTERS:

                x = [i for i in range(1, num_bins + 1)]
                y = np.asarray(bin_data[alg][metric]["data"])
                yerr = np.asarray(bin_data[alg][metric]["std"])

                marker_point = len(x) - 1
                for i in range(len(x) - 2, -1, -1):
                    if y[i] != y[marker_point]:
                        break
                    else:
                        marker_point = i

                for i in range(marker_point + 1, len(yerr)):
                    yerr[i] = 0

                if marker_point != len(x) - 1 and alg == "IPML-R":
                    ax.plot(x, y, label=label, color=color, linestyle=linestyle, markevery=[marker_point], marker="x",
                            markersize=14)
                    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.09)
                else:
                    ax.plot(x, y, label=label, color=color, linestyle=linestyle)
                    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.09)




                ax.set_xticklabels(ax.get_xticklabels(), fontsize=XTICKLABEL_FONTSIZE)

                ax.set_ylim([-0.1, 1.05])

                if column == 0:
                    yticklabels = ax.get_yticklabels()
                    yticklabels[1] = "$\mathcal{T}'$"
                    ax.set_yticklabels(yticklabels, fontsize=YTICKLABEL_FONTSIZE)
                # if column > 0:
                #     for ax in axes:
                #         ax.set_yticklabels([])
                # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    fig.legend(
        labels=["QACE (Ours)", "_b",
                "GLIB$-$G", "_b",
                "GLIB$-$L", "_b"],
        loc="center", ncols=4,
        bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=LEGEND_FONTSIZE)

    fig.text(0.43, -0.06, "Learning Time (minutes)", fontsize=XLABEL_FONTSIZE)

    fig.savefig("%s/ace_plots.pdf" % (results_dir), bbox_inches="tight")
    fig.savefig("%s/ace_plots.png" % (results_dir), bbox_inches="tight")


def get_vd_bin_index(vd, vd_ranges):
    for idx, vd_range in enumerate(vd_ranges.categories):

        if vd in vd_range:
            return len(vd_ranges.categories) - 1 - idx

    return None


def process_metric(bin_data, alg, metric, X_METRIC, data, run_no, num_bins, bin_size):
    x_values = data[alg][X_METRIC][run_no]
    y_values = data[alg][metric][run_no]
    assert len(x_values) == len(y_values)

    vd = 1.0

    last_bin_no = 0
    for x_value, y_value in zip(y_values, x_values):

        next_bin_no = int(y_value // bin_size)
        if next_bin_no >= num_bins:

            for bin_idx in range(last_bin_no, num_bins):

                bin_data[alg][metric]["data"][bin_idx].append(vd)

            return
        elif next_bin_no == last_bin_no:

            vd = x_value
        else:
            for bin_idx in range(last_bin_no, next_bin_no):

                bin_data[alg][metric]["data"][bin_idx].append(vd)

            vd = x_value
            last_bin_no = next_bin_no

    for bin_idx in range(last_bin_no, num_bins):
        bin_data[alg][metric]["data"][bin_idx].append(vd)

    pass


if __name__ == "__main__":

    BASE_DIR = "/tmp/results"
    RESULTS_DIR = BASE_DIR
    START_RUN = 0
    END_RUN = 9

    algs = ["IPML-R", "GLIB_G1", "GLIB_L2"]

    SECONDARY_Y_METRIC = "Interaction"
    X_METRIC = 'Variational Difference (Samples)'
    X_AXIS_LABEL = "Time (s)"
    SECONDARY_Y_AXIS_LABEL = "# of Samples"
    SECONDARY_Y_AXIS_LEGEND_LABEL = "Sample Count"

    SECONDARY_Y_METRIC = "Elapsed Time"
    X_METRIC = 'Elapsed Time'
    X_AXIS_LABEL = "# of Samples (s, a, s') Used for Learning"
    SECONDARY_Y_AXIS_LABEL = "Learning Time (s)"
    SECONDARY_Y_AXIS_LEGEND_LABEL = "Learning Time"

    metrics = ["Variational Difference (Samples)"]
    metrics_default_values = [1.0, 0, 0]

    fig = plt.figure(constrained_layout=False, figsize=(20, 5))
    gs = gridspec.GridSpec(ncols=4, nrows=2, figure=fig,
                           hspace=0.1, wspace=0.06)

    PLOTTERS = [
        ("IPML-R", "IPML-R (Ours)", "blue", "-", "--"),
        # ("IPML-S", "IPML-S (Ours)", "red", "-", "--"),
        ("GLIB_G1", "GLIB (G1)", "grey", "-", "--"),
        ("GLIB_L2", "GLIB (L2)", "green", "-", "--"),
    ]

    RUNS = list(range(30))
    RUNS.remove(24)
    DOMAINS = ["Explodingblocks", "Tireworld",
               "First_responders", "Probabilistic_elevators", ]
    domain_data = {}

    MAX_TIME = 14400
    for column, DOMAIN in enumerate(DOMAINS):
        data = {}
        for alg in algs:
            data[alg] = {}
            for metric in metrics:
                data[alg][metric] = {}
                data[alg][X_METRIC] = {}

        for run_no in RUNS:

            ipml_r_file = "%s/run%u/%s/ipml_randomized_sdm_counted.csv" % (BASE_DIR, run_no,
                                                                           DOMAIN)

            # ipml_s_file = "%s/run%u/%s/ipml_sequential.csv" % (BASE_DIR, run_no,
            #                                                      DOMAIN)

            g1_file = "%s/run%u/%s/glib_g1_lndr.csv" % (BASE_DIR, run_no,
                                                        DOMAIN)
            g2_file = "%s/run%u/%s/glib_l2_lndr.csv" % (BASE_DIR, run_no,
                                                        DOMAIN)

            for metric in metrics:
                data["IPML-R"][X_METRIC][run_no], data["IPML-R"][metric][run_no] = get_data(ipml_r_file,
                                                                                            X_METRIC, metric, "IPML-R",
                                                                                            metric)

                # data["IPML-S"][X_METRIC][run_no], data["IPML-S"][metric][run_no] = get_data(ipml_s_file,
                #                                           X_METRIC, metric, "IPML-S", metric)

                data["GLIB_G1"][X_METRIC][run_no], data["GLIB_G1"][metric][run_no] = get_data(g1_file,
                                                                                              X_METRIC, metric, alg,
                                                                                              metric)

                data["GLIB_L2"][X_METRIC][run_no], data["GLIB_L2"][metric][run_no] = get_data(g2_file,
                                                                                              X_METRIC, metric, alg,
                                                                                              metric)

        TIME_INTERVAL_IN_SEC = 60
        NUM_BINS = MAX_TIME // TIME_INTERVAL_IN_SEC


        bin_data = {}
        for alg in algs:
            bin_data[alg] = {}
            bin_data[alg][X_METRIC] = [i for i in range(1, NUM_BINS + 1)]

            for metric in metrics:

                bin_data[alg][metric] = {}
                bin_data[alg][metric]["data"] = [[] for i in range(NUM_BINS)]
                bin_data[alg][metric]["std"] = [0] * NUM_BINS

                for run_no in RUNS:
                    process_metric(bin_data, alg, metric, X_METRIC, data, run_no, NUM_BINS, TIME_INTERVAL_IN_SEC)

        for alg in algs:
            for metric in metrics:
                for bin_idx in range(NUM_BINS):
                    assert len(bin_data[alg][metric]["data"][bin_idx]) == len(RUNS)

        for metric in metrics:
            standardize(bin_data, algs, metric, data)
    #
    #     for alg in algs:
    #         for metric in metrics:
    #             fill_gaps(bin_data[alg][metric]["data"], 0)
    #             fill_gaps(bin_data[alg][metric]["std"], 0)
    #
    #     write_file(RESULTS_DIR, DOMAIN, bin_data, algs, X_METRIC, metrics, NUM_BINS)
        domain_data[DOMAIN] = bin_data
    #
    plot_supplemental(RESULTS_DIR, DOMAINS, domain_data, NUM_BINS)
