'''
Created on Jan 18, 2023

@author: rkaria
'''
import os.path

import matplotlib
import csv
import ast
import matplotlib
import matplotlib.pyplot as plt
import statistics
import numpy as np
import matplotlib.gridspec as gridspec
import itertools
import pickle
import matplotlib.transforms as transforms
import matplotlib.lines as lines

from matplotlib.legend_handler import HandlerTuple
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })

def get_data(results_dir, domain_name, alg, run,
                  data, xkey, ykeys=None):

    filepath = "%s/%s/run%s/%s_steps_vs_tasks.csv" % (results_dir,
                                                      domain_name,
                                                      run,
                                                      alg)


    fh = open(filepath, "r")
    reader = csv.DictReader(fh, delimiter=";")

    if ykeys is None:
        ykeys = reader.fieldnames

    domain_dict = data.setdefault(domain_name, {})
    alg_dict = domain_dict.setdefault(alg, {})
    for row in reader:

        x_value = ast.literal_eval(row[xkey])
        x_dict = alg_dict.setdefault(x_value, {})
        for ykey in ykeys:
            y_list = x_dict.setdefault(ykey, [])
            try:
                y_list.append(ast.literal_eval(row[ykey]))
            except (ValueError, SyntaxError):

                assert isinstance(row[ykey], str)
                y_list.append(row[ykey])


def ensure_data_consistent(data, runs):
    x_keys = None
    for domain in data:
        for alg in data[domain]:
            for xkey in data[domain][alg]:

                if x_keys is None:
                    x_keys = data[domain][alg].keys()

                else:
                    assert x_keys == data[domain][alg].keys()

                for ykey, ylist in data[domain][alg][xkey].items():
                    assert len(ylist) == len(runs)

def compute_means(data):

    for domain in data:
        for alg in data[domain]:
            for xkey in data[domain][alg]:
                for ykey, ylist in list(data[domain][alg][xkey].items()):

                    try:
                        data[domain][alg][xkey][ykey] = statistics.mean(ylist)
                        err_ykey = "%s-err" % (ykey)
                        assert err_ykey not in data[domain][alg][xkey]

                        if len(ylist) > 0:
                            data[domain][alg][xkey][err_ykey] = \
                                statistics.stdev(ylist)
                        else:
                            data[domain][alg][xkey][err_ykey] = 0
                    except Exception:

                        assert isinstance(ylist[0], (tuple, str))

def get_all_data(results_dir, domains, algs, runs, xkey, ykeys=None):

    data = {}
    for domain, alg, run in itertools.product(domains, algs, runs):

        get_data(results_dir, domain, alg, run, data, xkey, ykeys)

    ensure_data_consistent(data, runs)

    return data

def compute_adaptive_efficiency(data, horizon=40,
                        key_name="Adaptive Efficiency",
                        steps_per_task=100 * 1000):

    epsilon=0.6
    for domain in data:

        for alg in data[domain]:

            if alg == "oracle":
                continue

            xkeys = list(data[domain][alg].keys())
            assert len(xkeys) > 0
            total_runs = len(data[domain][alg][xkeys[0]]["Oracle Diff"])
            adaptive_efficiency = np.full(total_runs, steps_per_task)
            task_no = 0
            total_bad_values = 0
            values_per_task = steps_per_task // 100
            task_start = task_no * values_per_task
            task_end = task_start + values_per_task
            oracle_reward = [data[domain]["oracle"][x]["Avg. Reward"] for x in xkeys[task_start:task_end]]
            oracle_avg = np.mean(oracle_reward)
            oracle_std = np.std(oracle_reward)

            for xkey in xkeys:

                if xkey % steps_per_task == 0:
                    adaptive_efficiency_dict = data[domain][alg].setdefault("Adaptive Efficiency", {})
                    task_name = "t-%u" % (task_no)
                    adaptive_efficiency_dict[task_name] = adaptive_efficiency
                    task_no += 1
                    adaptive_efficiency = np.full(total_runs, steps_per_task)

                    task_start = task_no * values_per_task
                    task_end = task_start + values_per_task
                    oracle_reward = [data[domain]["oracle"][x]["Avg. Reward"] for x in xkeys[task_start:task_end]]
                    oracle_avg = np.mean(oracle_reward)
                    oracle_std = np.std(oracle_reward)

                alg_reward = np.asarray(data[domain][alg][xkey]["Avg. Reward"])

                curr_step = xkey - task_no * steps_per_task
                for i, alg_r in enumerate(alg_reward):

                    if abs(alg_r - oracle_avg) <= oracle_std * 4:
                        if adaptive_efficiency[i] == steps_per_task:
                            adaptive_efficiency[i] = curr_step
                            total_bad_values -= 1
                            total_bad_values = max(0, total_bad_values)
                    elif total_bad_values == 100:
                        adaptive_efficiency[i] = steps_per_task
                        total_bad_values = 0
                    else:
                        total_bad_values += 1

            # print(domain, alg)
            # print(adaptive_efficiency_dict)


def compute_oracle_diff(data, oracle="oracle", horizon=40,
                        key_name="Oracle Diff"):

    for domain in data:

        assert oracle in data[domain]
        for alg in data[domain]:
            for xkey in list(data[domain][alg].keys()):

                assert "Avg. Reward" in data[domain][alg][xkey]
                oracle_reward = np.asarray(data[domain][oracle][xkey]["Avg. Reward"])
                alg_reward = np.asarray(data[domain][alg][xkey]["Avg. Reward"])
                assert len(oracle_reward) == len(alg_reward)

                oracle_diff = []
                for or_r, alg_r in zip(oracle_reward, alg_reward):

                    assert or_r != -horizon
                    if alg_r == -horizon:

                        oracle_diff.append(0)
                    else:
                        oracle_diff.append(or_r / alg_r)
                    pass

                data[domain][alg][xkey][key_name] = oracle_diff
                pass

def get_x_y(data, domain, alg, ykey):

    x = [x for x in data[domain][alg].keys() if not isinstance(x, str)]

    err_ykey = "%s-err" % (ykey)

    y = [data[domain][alg][_x][ykey] for _x in x]
    yerr = [data[domain][alg][_x][err_ykey] for _x in x]

    return np.asarray(x), np.asarray(y), np.asarray(yerr)

def transform_ticklabels(ax, axis="x", fontsize=12):

    if axis == "x":
        ticks = ax.get_xticks()
    else:
        ticks = ax.get_yticks()

    ticklabels = []
    for tick in ticks:

        value = float(tick)

        if value < 0:
            value = ""
        elif value  > 1000:
            value = value / 1000
            value = "%.0fk" % (value)
        else:
            value = "%.0f" % (value)
        ticklabels.append(value)

    if axis == "x":
        ax.set_xticklabels(ticklabels, fontsize=fontsize)
    else:
        ax.set_yticklabels(ticklabels, fontsize=fontsize)

def get_splice(x, y, yerr, start_idx, end_idx):

    return x[start_idx:end_idx], y[start_idx:end_idx], yerr[start_idx:end_idx]

def plot_old_style():

    # 5 tasks separated
    fig = plt.figure(constrained_layout=False, figsize=(20, 8))
    fig.text(0.47, 0.56, "Simulator Steps $|\Delta|$", fontsize=16)
    fig.text(0.47, 0.045, "Simulator Steps $|\Delta|$", fontsize=16)
    fig.text(0.1, 0.15, "Avg. Reward w.r.t. Oracle", rotation=90, fontsize=16)
    gs = gridspec.GridSpec(ncols=4, nrows=9, figure=fig,
                           hspace=0.1, wspace=0.06)

    for column, domain in enumerate(DOMAINS):
        domain_ax = fig.add_subplot(gs[0:3, column])
        domain_ax.set_title(domain["name"], fontsize=18)
        for alg in ALGS:
            x, y, yerr = get_x_y(data, domain["source"], alg["name"],
                                 "Total Goals Reached")
            lh = domain_ax.plot(x, y,
                           label=alg["label"],
                           color=alg["color"],
                           linestyle=alg["linestyle"])
            alg["handle"] = lh
            domain_ax.fill_between(x, y - yerr, y + yerr, color=alg["color"],
                                   alpha=AREA_ALPHA)

        transform_ticklabels(domain_ax)
        transform_ticklabels(domain_ax, axis="y")

        for task_no in range(TOTAL_TASKS):

            t_ax = fig.add_subplot(gs[TASK_START_GS_ROW + task_no, column])
            t_ax.text(0.90, 0.23, "$M_%s, \delta_%s$" % (task_no, task_no),
                      horizontalalignment="center",
                      verticalalignment="center",
                      transform=t_ax.transAxes,
                      fontsize=12)

            # https://stackoverflow.com/questions/33707162/zigzag-or-wavy-lines-in-matplotlib
            with plt.rc_context({'path.sketch': (1, 20, 1)}):
                domain_ax.axvline(task_no * STEPS_PER_TASK,
                                  linestyle="solid",
                                  color="grey",
                                  alpha=0.3)

            blended_transform = transforms.blended_transform_factory(
                domain_ax.transData, domain_ax.transAxes)

            domain_ax.text((task_no * STEPS_PER_TASK) + 3000, 0.92, "$\delta_%s$" % (task_no),
                           verticalalignment="center",
                           transform=blended_transform,
                           fontsize=12)

            for alg in ALGS:

                if alg["name"].lower() == "oracle":
                    continue

                x, y, yerr = get_x_y(data, domain["source"], alg["name"],
                                     "Oracle Diff")

                t_x, t_y, t_yerr = get_splice(x, y, yerr,
                                              start_idx=task_no * LOG_EVENTS_PER_TASK,
                                              end_idx=LOG_EVENTS_PER_TASK * (task_no + 1))

                t_ax.plot(t_x, t_y, label=alg["label"],
                           color=alg["color"],
                           linestyle="solid")

                # t_ax.fill_between(t_x, t_y - t_yerr, t_y + t_yerr, color=alg["color"],
                #                        alpha=AREA_ALPHA)

            transform_ticklabels(t_ax)
            if task_no == 0:
                first_xticklabels = t_ax.get_xticklabels()

            t_ax.set_ylim([-0.1, 1.3])

            if column != 0:
                t_ax.set_yticklabels([])

        if task_no != TOTAL_TASKS - 1:
            t_ax.set_xticklabel([])
        else:
            t_ax.set_xticklabels(first_xticklabels)

        if column != 0:
            domain_ax.set_yticklabels([])
        else:
            domain_ax.set_ylabel("Total Tasks\nAccomplished", fontsize=16)

        domain_ax.set_ylim([-2.5 * 1000, 120 * 1000])

    legend_order = ["oracle", "drift", "qace", "qace-stateless", "qlearning"]
    legend_labels = []
    legend_handles = []
    for legend in legend_order:
        for alg in ALGS:
            if legend == alg["name"]:
                legend_handles.append(alg["handle"][0])
                legend_labels.append(alg["label"])
                break
    fig.legend(handles=legend_handles,
               labels=legend_labels,
               loc="center", ncols=5,
                bbox_to_anchor=(0.5, 0.95), frameon=False, fontsize=16)

    fig.savefig("%s/plots.png" % (RESULTS_DIR),  bbox_inches="tight")

def plot_adaptive_efficiency(ae_ax, data, domain, TOTAL_TASKS, ALGS):

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    width = 0.1
    multiplier = 0

    x = np.arange(TOTAL_TASKS)
    task_names = ["t-%s" % (task_no) for task_no in range(TOTAL_TASKS)]
    label_names = ["$M_%s$" % (task_no) for task_no in range(TOTAL_TASKS)]
    for alg in ALGS:

        if alg["name"] == "oracle":
            continue

        ae_means = [data[domain["source"]][alg["name"]]["Adaptive Efficiency"][name]
                    for name in task_names]
        ae_errs = [data[domain["source"]][alg["name"]]["Adaptive Efficiency"]["%s-err" % (name)]
                    for name in task_names]

        offset = width * multiplier
        rects = ae_ax.bar(x + offset, ae_means, width, label=alg["name"], color=alg["color"])
        # ae_ax.errorbar(x + offset, yerr=ae_errs)
        alg["bar_handle"] = rects
        multiplier += 1

    ae_ax.set_ylim([0, 50 * 1000])

    ae_ax.set_xticks(x + width, label_names, fontsize=12)


if __name__ == "__main__":

    RESULTS_DIR = "/tmp/results/"
    RUNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ALGS = ["qace", "qace-stateless", "drift", "oracle"]
    XKEY = "Total Steps"
    YKEYS = ["Total Goals Reached"]
    DOMAINS = [
        {"name": "2-Bandit", "source": "directed_gof"},

    ]

    ALGS = [

        {"name": "oracle",
         "color": "black",
         "linestyle": "dotted",
         "label": "Oracle"
         },

        {"name": "qace",
         "color": "red",
         "linestyle": "solid",
         "label": "Adaptive + Comprehensive (A+C-Learner)"
         },

        # {"name": "qace-stateless",
        #  "color": "green",
        #  "linestyle": "dashdot",
        #  "label": "Unadaptive + Comprehensive (U+C-Learner)"
        #  },

        {"name": "qlearning",
         "color": "grey",
         "linestyle": "dashed",
         "label": "Q-Learning"
         },

        {"name": "drift",
         "color": "blue",
         "linestyle": "solid",
         "label": "Adaptive + Need-based (CLaP (Ours))"
         },
    ]

    DATA_PATH = "/tmp/data.pkl"
    DELETE = False
    STEPS_PER_TASK = 1000
    STEP_LOG_INTERVAL = 100
    LOG_EVENTS_PER_TASK = STEPS_PER_TASK // STEP_LOG_INTERVAL
    AREA_ALPHA = 0.05
    ALG_START_ROW = 3
    TOTAL_TASKS = 2
    ORACLE = ALGS[0]
    HORIZON = 40

    if DELETE and os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)

    if False and os.path.exists(DATA_PATH):
        with open(DATA_PATH, "rb") as fh:
            data = pickle.load(fh)
    else:
        data = get_all_data(RESULTS_DIR, [x["source"] for x in DOMAINS],
                            [x["name"] for x in ALGS],
                            RUNS, XKEY)
        compute_oracle_diff(data)
        # compute_adaptive_efficiency(data)
        compute_means(data)
        # compute_means(data)

        # with open(DATA_PATH, "wb") as fh:
        #     pickle.dump(data, fh)

    # compute_adaptive_efficiency(data)
    # compute_means(data)

    fig = plt.figure(constrained_layout=False, figsize=(10, 8))
    fig.text(0.40, 0.050, "Simulator Steps $|\Delta|$", fontsize=16)
    fig.text(0.065, 0.19, "(b) Avg. Reward", rotation=90, fontsize=16)
    # fig.text(0.498, 0.05, "Tasks", fontsize=16)

    # https://matplotlib.org/stable/gallery/misc/fig_x.html
    fig.add_artist(lines.Line2D([0.086, 0.086], [0.11, 0.48], color="black"))

    # Here is the label and arrow code of interest
    # fig.annotate('SDL', xy=(0.5, 0.90), xytext=(0.5, 1.00), xycoords='axes fraction',
    #             fontsize=16, ha='center', va='bottom',
    #             bbox=dict(boxstyle='square', fc='white', color='k'),
    #             arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=2.0, color='k'))

    gs = gridspec.GridSpec(ncols=1, nrows=6, figure=fig,
                           hspace=0.1, wspace=0.06)

    for column, domain in enumerate(DOMAINS):
        domain_ax = fig.add_subplot(gs[0:3, column])
        domain_ax.set_title(domain["name"], fontsize=18)
        for alg in ALGS:
            x, y, yerr = get_x_y(data, domain["source"], alg["name"],
                                 "Total Goals Reached")
            lh = domain_ax.plot(x, y,
                           label=alg["label"],
                           color=alg["color"],
                           linestyle=alg["linestyle"])
            alg["handle"] = lh
            domain_ax.fill_between(x, y - yerr, y + yerr, color=alg["color"],
                                   alpha=AREA_ALPHA)

        # transform_ticklabels(domain_ax)
        # transform_ticklabels(domain_ax, axis="y")


        for task_no in range(TOTAL_TASKS):
            with plt.rc_context({'path.sketch': (1, 20, 1)}):
                domain_ax.axvline(task_no * STEPS_PER_TASK,
                                  linestyle="solid",
                                  color="grey",
                                  alpha=0.3)

            blended_transform = transforms.blended_transform_factory(
                domain_ax.transData, domain_ax.transAxes)
            domain_ax.text((task_no * STEPS_PER_TASK), 0.92, "$M_%s, \delta_%s$" % (task_no, task_no),
                           verticalalignment="center",
                           transform=blended_transform,
                           fontsize=12)


        alg_no = 0
        for alg in ALGS:
            if alg["name"] == "oracle":
                continue

            alg_ax = fig.add_subplot(gs[ALG_START_ROW + alg_no, column])
            for task_no in range(TOTAL_TASKS):
                with plt.rc_context({'path.sketch': (1, 20, 1)}):
                    alg_ax.axvline(task_no * STEPS_PER_TASK,
                                      linestyle="solid",
                                      color="grey",
                                      alpha=0.3)

            o_x, o_y, o_yerr = get_x_y(data, domain["source"], "oracle",
                                 "Avg. Reward")

            markers_on = []
            for i in range(0, (TOTAL_TASKS * STEPS_PER_TASK) // STEP_LOG_INTERVAL, 5):
                markers_on.append(i)
            markers_on = np.asarray(markers_on)
            alg_ax.plot(o_x, o_y, label=ORACLE["label"],
                      color=ORACLE["color"],
                      marker="x",
                      markevery=markers_on,
                      markersize=8,
                      linestyle="solid")
            alg_ax.fill_between(o_x, o_y - o_yerr, o_y + o_yerr, color=ORACLE["color"],
                                                     alpha=AREA_ALPHA)

            alg_x, alg_y, alg_yerr = get_x_y(data, domain["source"], alg["name"],
                                 "Avg. Reward")
            alg_ax.plot(alg_x, alg_y, label=alg["label"],
                      color=alg["color"],
                      linestyle=alg["linestyle"])
            alg_ax.fill_between(alg_x, alg_y - alg_yerr, alg_y + alg_yerr, color=alg["color"],
                                alpha=AREA_ALPHA)

            alg_no += 1
                # t_ax.fill_between(t_x, t_y - t_yerr, t_y + t_yerr, color=alg["color"],
                #                        alpha=AREA_ALPHA)

            # transform_ticklabels(alg_ax)

            alg_ax.set_yticks([-HORIZON, 0, 5])
            alg_ax.set_yticklabels([str(-HORIZON), str(0), ""])
            # alg_ax.set_ylim([-40, 5])
            # alg_ax.invert_yaxis()

            if column != 0:
                alg_ax.set_yticklabels([])

            if alg_no != len(ALGS) - 1:
                alg_ax.set_xticklabels([])

        if column != 0:
            domain_ax.set_yticklabels([])
        else:
            domain_ax.set_ylabel("(a) Total Tasks\nAccomplished", fontsize=16)

        # domain_ax.set_ylim([-2.5 * 1000, 120 * 1000])

    legend_order = ["oracle", "qlearning", "drift", "qace",]
    legend_labels = []
    legend_handles = []
    handler_map = {}
    for legend in legend_order:
        for alg in ALGS:
            if legend == alg["name"]:

                if "bar_handle" not in alg:
                    legend_handles.append((alg["handle"][0],))
                else:
                    legend_handles.append((alg["bar_handle"], alg["handle"][0]))
                handler_map[legend_handles[-1]] = HandlerTuple(ndivide=None, pad=0.3)
                legend_labels.append(alg["label"])
                break
    fig.legend(handles=legend_handles,
               labels=legend_labels,
               ncols=2,
               handler_map=handler_map,
               handlelength=4,
               loc="center",
                bbox_to_anchor=(0.53, 0.96), frameon=False, fontsize=14)

    fig.savefig("%s/plots.png" % (RESULTS_DIR),  bbox_inches="tight")

    pass
    pass

def get_interaction_adjustment(key, alg, value, prev_value):

    if "Interaction" in key:

        if "GLIB" in alg:

            value = 1 if value == 0 else value
        elif value == 0:
            value = prev_value + 1
            return value, value
        else:
            value = prev_value + 5
            return value, value

    return value, prev_value




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

    idx = get_last_range(x)

    x[0] = default_value if x[0] == float("inf") else x[0]
    for i in range(idx):

        if x[i] == float("inf"):
            x[i] = x[i - 1]

def get_filename(domain, alg):

    if alg == "oracle":

        suffix = "oracle_steps_vs_tasks.csv"
    elif alg == "drift":

        suffix = "odrift_steps_vs_tasks.csv"
    elif alg == "qlearning":

        suffix = "qlearning_steps_vs_tasks.csv"
    elif alg == "qace":

        suffix = "qace_steps_vs_tasks.csv"
    elif alg == "qace-stateless":

        suffix = "qace-stateless_steps_vs_tasks.csv"
    else:
        assert False

    return "%s_%s" % (domain.lower(), suffix)

def write_file(output_dir, domain, bin_data, algs, x_metric, metrics, num_bins):

    for alg in algs:
        filepath = output_dir + "/" +  get_filename(domain, alg)
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

    if domain == "tireworld":

        return "TireWorld"
    elif domain == "first_responders":

        return "FirstResponders"
    elif domain == "elevators":

        return "Elevators"
    elif domain == "blocks":

        return "Blocks"
    else:

        assert False

def get_ground_truth_cost_sr_for_domain(domain):

    if domain == "tireworld":

        return (7.725, 1.0)
    elif domain == "first_responders":

        return (12.25, 0.92)
    elif domain == "elevators":

        return (18.75, 1.0)
    elif domain == "blocks":

        return (9.35, 1.0)
    else:

        assert False

def plot_supplemental(results_dir, domains, domain_data, num_bins):

    fig = plt.figure(constrained_layout=False, figsize=(20, 10))
    gs = gridspec.GridSpec(ncols=4, nrows=3, figure=fig,
                           hspace=0.1,wspace=0.06)

    metrics =  ["Total Goals Reached",
                "Avg. Reward",
                "Avg. Success Rate"]

    PLOTTERS = [
            ("oracle", "Oracle", "blue", "-", 1, "--"),
            ("qlearning", "Q Learning", "grey", "-", 1, "--"),
            ("qace", "QACE", "green", "-", 1, "--"),
            ("qace-stateless", "QACE-S", "yellow", "-", 1, "--"),
            ("drift", "Drift (Ours)", "red", "-", 1, "--"),
        ]

    for column, domain in enumerate(domains):

        title = get_title_for_domain(domain)
        r1 = fig.add_subplot(gs[0, column])
        r2 = fig.add_subplot(gs[1, column])
        r3 = fig.add_subplot(gs[2, column])

        axes = [r1, r2, r3]

        r1.set_title(title, fontsize=15)

        for axis in [r1, r2, r3]:

            axis.set_xticklabels([])

        if column == 0:

            r1.set_ylabel("Total Goals Reached")
            r2.set_ylabel("Average Reward")
            r3.set_ylabel("Average\nSuccess\nRate")
        else:
            for ax in axes:
                ax.set_yticklabels([])

        r1.set_ylim([0, 120000])
        r2.set_ylim([-45, 3])
        r3.set_ylim([-0.05, 1.1])


        bin_data = domain_data[domain]
        cost, sr = get_ground_truth_cost_sr_for_domain(domain)

        for ax, metric in zip(axes, metrics):
            for alg, label, color, linestyle, opacity, _ in PLOTTERS:

                x = np.asarray(bin_data[alg][X_METRIC])
                y = np.asarray(bin_data[alg][metric]["data"])
                yerr = np.asarray(bin_data[alg][metric]["std"])

                # if ax == r4:
                #
                #     ax.scatter([-50], [cost], marker="*", color="black")


                ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.0, alpha=opacity)
                ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.09)


    fig.legend(
        labels=["Oracle", "_b",
                "Qlearning", "_b",
                "QACE", "_b",
                "QACE-S", "_b",
                "Drift (Ours)", "_b"],
               loc="center", ncols=5,
                bbox_to_anchor=(0.5, 0.93), frameon=False, fontsize=13)

    fig.text(0.4, 0.063, "Total Steps", fontsize=13.0)

    fig.savefig("%s/supplemental.pdf" % (results_dir), bbox_inches="tight")
    fig.savefig("%s/supplemental.png" % (results_dir), bbox_inches="tight")


if __name__ == "main_old":

    BASE_DIR = "/Users/pulkitverma/Code/differential-learning-private/results/"
    RESULTS_DIR = "/Users/pulkitverma/Code/differential-learning-private/results"
    START_RUN = 0
    END_RUN = 9

    algs = ["drift", "oracle", "qace", "qace-stateless", "qlearning"]

    SECONDARY_Y_METRIC = "Interaction"
    X_METRIC = "Total Steps"
    X_AXIS_LABEL ="Total Steps"
    SECONDARY_Y_AXIS_LABEL = "Total Goals Reached"
    SECONDARY_Y_AXIS_LEGEND_LABEL = "Total Goals Reached"

    SECONDARY_Y_METRIC = "Task Number"
    X_METRIC = "Total Steps"
    X_AXIS_LABEL = "Total Steps"
    SECONDARY_Y_AXIS_LABEL = ""
    SECONDARY_Y_AXIS_LEGEND_LABEL = ""

    metrics =  ["Total Goals Reached",
                "Avg. Reward",
                "Avg. Success Rate"]
    metrics_default_values = [0, 0, 0]


    fig = plt.figure(constrained_layout=False, figsize=(20, 5))
    gs = gridspec.GridSpec(ncols=4, nrows=2, figure=fig,
                           hspace=0.1,wspace=0.06)

    PLOTTERS = [
            ("oracle", "Oracle", "blue", "-", "--"),
            ("qlearning", "Q Learning", "grey", "-", "--"),
            ("qace", "QACE", "green", "-", "--"),
            ("qace-stateless", "QACE-S", "yellow", "-", 1, "--"),
            ("drift", "Drift (Ours)", "red", "-", "--"),
        ]

    RUNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    DOMAINS = ["tireworld", "first_responders", "elevators", "blocks"]
            #    , "Probabilistic_elevators",
            #         "First_responders"
                    # ]
    domain_data = {}

    for column, DOMAIN  in enumerate(DOMAINS):
        data = {}
        for alg in algs:
            data[alg] = {}
            for metric in metrics:

                data[alg][metric] = {}
                data[alg][X_METRIC] = {}

        for run_no in RUNS:

            oracle_file = "%s/%s/run%u/oracle_steps_vs_tasks.csv" % (BASE_DIR,
                                                                 DOMAIN, run_no)

            drift_file = "%s/%s/run%u/drift_steps_vs_tasks.csv" % (BASE_DIR,
                                                                 DOMAIN, run_no)

            qlearning_file = "%s/%s/run%u/qlearning_steps_vs_tasks.csv" % (BASE_DIR,
                                                                 DOMAIN, run_no)
            qace_file = "%s/%s/run%u/qace_steps_vs_tasks.csv" % (BASE_DIR,
                                                                 DOMAIN, run_no)
            qace_stateless_file = "%s/%s/run%u/qace-stateless_steps_vs_tasks.csv" % (BASE_DIR,
                                                                 DOMAIN, run_no)

            for metric in metrics:

                data["oracle"][X_METRIC][run_no], data["oracle"][metric][run_no] = get_data(oracle_file,
                                                          X_METRIC, metric, "oracle", metric)

                data["qlearning"][X_METRIC][run_no], data["qlearning"][metric][run_no] = get_data(qlearning_file,
                                                          X_METRIC, metric, alg, metric)

                data["qace"][X_METRIC][run_no], data["qace"][metric][run_no] = get_data(qace_file,
                                                          X_METRIC, metric, alg, metric)

                data["qace-stateless"][X_METRIC][run_no], data["qace-stateless"][metric][run_no] = get_data(qace_stateless_file,
                                                          X_METRIC, metric, alg, metric)

                data["drift"][X_METRIC][run_no], data["drift"][metric][run_no] = get_data(drift_file,
                                                          X_METRIC, metric, "drift", metric)

        max_x = compute_max_for_matric(data, algs, X_METRIC, RUNS)
        print(max_x)
        NUM_BINS = 100000

        BIN_SIZE = int(max_x // NUM_BINS)
        BIN_SIZE = max(1, BIN_SIZE)

        print(BIN_SIZE)
        print([i * BIN_SIZE for i in range(NUM_BINS)])

        bin_data = {}
        for alg in algs:
            bin_data[alg] = {}
            bin_data[alg][X_METRIC] = [i * BIN_SIZE for i in range(NUM_BINS)]


            for metric in metrics:

                bin_data[alg][metric] = {}
                bin_data[alg][metric]["data"] = [[] for i in range(NUM_BINS)]
                bin_data[alg][metric]["std"] = [0] * NUM_BINS

        for metric in metrics:
            organize_into_bins(bin_data, algs, X_METRIC, metric, data, RUNS,
                               BIN_SIZE, NUM_BINS)

            standardize(bin_data, algs, metric, data)


        for alg in algs:
            for metric, default_value in zip(metrics, metrics_default_values):

                fill_gaps(bin_data[alg][metric]["data"], default_value)
                fill_gaps(bin_data[alg][metric]["std"], 0)


        write_file(RESULTS_DIR, DOMAIN, bin_data, algs, X_METRIC, metrics, NUM_BINS)
        domain_data[DOMAIN] = bin_data

    plot_supplemental(RESULTS_DIR, DOMAINS, domain_data, NUM_BINS)



