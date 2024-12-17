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
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
})

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
                        data[domain][alg][xkey][ykey] = round(statistics.mean(ylist), 2)
                        err_ykey = "%s-err" % (ykey)
                        assert err_ykey not in data[domain][alg][xkey]

                        if len(ylist) > 0:
                            data[domain][alg][xkey][err_ykey] = \
                                round(statistics.stdev(ylist), 2)
                        else:
                            data[domain][alg][xkey][err_ykey] = 0
                    except Exception:

                        assert isinstance(ylist[0], str)

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
        elif value  >= 1000:
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

def plot_adaptive_efficiency(ae_ax, data, domain, TOTAL_TASKS, ALGS,
                             YLIM=[0, 50 * 1000]):

    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    width = 0.18
    multiplier = 0

    x = np.arange(TOTAL_TASKS)
    task_names = ["t-%s" % (task_no) for task_no in range(TOTAL_TASKS)]
    label_names = ["$M_0$"] \
      + ["$M_%s\\rightarrow M_%s$" % (task_no - 1, task_no) for task_no in range(1, TOTAL_TASKS)]
    for alg in ALGS:

        if alg["name"] == "oracle":
            continue

        ae_means = [data[domain["source"]][alg["name"]]["Adaptive Efficiency"][name]
                    for name in task_names]
        ae_errs = [data[domain["source"]][alg["name"]]["Adaptive Efficiency"]["%s-err" % (name)]
                    for name in task_names]

        offset = width * multiplier
        rects = ae_ax.bar(x + offset, ae_means, width,
                          label=alg["name"],
                          color=alg["color"],
                          alpha=0.8,
                          hatch=alg["hatch"],
                          edgecolor="dimgrey")
        bar_labels = []
        for i, mean in enumerate(ae_means):

            if mean < 10:
                bar_labels.append(str(int(mean)))
            else:
                bar_labels.append("")

        ae_ax.bar_label(rects, labels=bar_labels, padding=1.5,
                        color=alg["color"], fontsize=11)
        # ae_ax.errorbar(x + offset, yerr=ae_errs)
        alg["bar_handle"] = rects
        multiplier += 1

    ae_ax.set_ylim(YLIM)

    ae_ax.set_xticks(x + width, label_names, fontsize=11)


if __name__ == "__main__":

    RESULTS_DIR = "/tmp/results/"
    RUNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ALGS = ["qace", "qace-stateless", "drift", "oracle"]
    XKEY = "Total Steps"
    YKEYS = ["Total Goals Reached"]
    DOMAINS = [
        {"name": "Tireworld", "source": "tireworld"},
        {"name": "Blocksworld", "source": "blocks"},
        {"name": "FirstResponders", "source": "first_responders"},
        {"name": "Elevators", "source": "elevators"},
    ]

    ALGS = [

        {"name": "oracle",
         "color": "black",
         "linestyle": (0, (1, 20)),
         "hatch": "",
         "label": "Oracle"
         },

        {"name": "drift",
         "color": "tab:blue",
         "linestyle": "solid",
         "hatch": "",
         "label": "Differential Learning: CLaP (Ours)"
         },

        {"name": "qace",
         "color": "tab:olive",
         "linestyle": "dashdot",
         "hatch": "////",
         "label": "Relearning: Adaptive QACE"
         },

        # {"name": "qace-stateless",
        #  "color": "green",
        #  "linestyle": "dashdot",
        #  "hatch": "",
        #  "label": "Non-adaptive + Comprehensive (U+C-Learner)"
        #  },

        {"name": "qlearning",
         "color": "tab:grey",
         "linestyle": "dashed",
         "hatch": "\\\\\\\\",
         "label": "Q-Learning"
         },
    ]

    DATA_PATH = "/tmp/data.pkl"
    DELETE = False
    STEPS_PER_TASK = 100 * 1000
    STEP_LOG_INTERVAL = 100
    LOG_EVENTS_PER_TASK = STEPS_PER_TASK // STEP_LOG_INTERVAL
    AREA_ALPHA = 0.05
    ALG_START_ROW = 3
    TOTAL_TASKS = 5
    ORACLE = ALGS[0]
    HORIZON = 40

    if DELETE and os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)

    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "rb") as fh:
            data = pickle.load(fh)
    else:
        data = get_all_data(RESULTS_DIR, [x["source"] for x in DOMAINS],
                            [x["name"] for x in ALGS],
                            RUNS, XKEY)
        compute_oracle_diff(data)
        compute_adaptive_efficiency(data)
        compute_means(data)
        # compute_means(data)

        with open(DATA_PATH, "wb") as fh:
            pickle.dump(data, fh)

    # compute_adaptive_efficiency(data)
    # compute_means(data)

    fig = plt.figure(constrained_layout=False, figsize=(20, 8))
    fig.text(0.47, 0.360, "Simulator Steps $|\Delta|$", fontsize=16)
    fig.text(0.095, 0.44, "(b) Avg. Reward", rotation=90,
             fontsize=16, ha="center")
    fig.text(0.498, 0.05, "Tasks", fontsize=16)
    fig.text(0.095, 0.15, "(c) Adaptive\nDelay", rotation=90,
             fontsize=16, ha="center")

    # https://matplotlib.org/stable/gallery/misc/fig_x.html
    fig.add_artist(lines.Line2D([0.106, 0.106], [0.42, 0.64], color="black"))

    fig.text(0.911, 0.52, "Higher Values Better", rotation=270,
             fontsize=16, ha="center", color="tab:brown")
    # fig.add_artist(lines.Line2D([0.905, 0.905], [0.425, 0.88],
    #                             color="tab:green", alpha=0.8))

    fig.text(0.911, 0.115, "Shorter Bars Better", rotation=270,
             fontsize=16, ha="center", color="tab:brown")

    # Here is the label and arrow code of interest
    # fig.annotate('SDL', xy=(0.5, 0.90), xytext=(0.5, 1.00), xycoords='axes fraction',
    #             fontsize=16, ha='center', va='bottom',
    #             bbox=dict(boxstyle='square', fc='white', color='k'),
    #             arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=2.0, color='k'))

    gs = gridspec.GridSpec(ncols=4, nrows=10, figure=fig,
                           hspace=0.1, wspace=0.06)

    for column, domain in enumerate(DOMAINS):
        domain_ax = fig.add_subplot(gs[0:3, column])
        domain_ax.set_title(domain["name"], fontsize=18)

        markers_on = []
        for i in range(0, (TOTAL_TASKS * STEPS_PER_TASK) // 100, 400):
            markers_on.append(i)
        markers_on.append((TOTAL_TASKS * STEPS_PER_TASK) // 100 - 1)
        markers_on = np.asarray(markers_on)
        ORACLE_MARKERSIZE = 20
        for alg in ALGS:
            x, y, yerr = get_x_y(data, domain["source"], alg["name"],
                                 "Total Goals Reached")

            if alg["name"] == "oracle":

                lh = domain_ax.scatter(markers_on * 100,
                    [y[i] for i in markers_on],
                                    color=alg["color"],
                                    label=alg["label"],
                                    marker="x",
                                   linewidths=1,
                                       zorder=10,
                                    s=ORACLE_MARKERSIZE)
                lh = [lh, None]
            else:
                lh = domain_ax.plot(x, y,
                               label=alg["label"],
                               color=alg["color"],
                               linestyle=alg["linestyle"])

                domain_ax.fill_between(x, y - yerr, y + yerr, color=alg["color"],
                                       alpha=AREA_ALPHA)

            alg["handle"] = lh

        transform_ticklabels(domain_ax)
        transform_ticklabels(domain_ax, axis="y")

        ae_ax = fig.add_subplot(gs[8:11, column])
        plot_adaptive_efficiency(ae_ax, data, domain, TOTAL_TASKS, ALGS,
                                 YLIM=[0, 55 * 1000])
        transform_ticklabels(ae_ax, axis="y")

        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
        ae_ax2 = fig.add_subplot(gs[7:8, column])
        plot_adaptive_efficiency(ae_ax2, data, domain, TOTAL_TASKS, ALGS,
                                 YLIM=[65 * 1000, 105 * 1000])
        transform_ticklabels(ae_ax2, axis="y")
        ae_ax2.set_xticklabels([])
        ae_ax2.set_xticks([])

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ae_ax.plot([0, 1], [1, 1], transform=ae_ax.transAxes, **kwargs)
        ae_ax2.plot([0, 1], [0, 0], transform=ae_ax2.transAxes, **kwargs)

        ae_ax.spines["top"].set_visible(False)
        ae_ax2.spines["bottom"].set_visible(False)
        if column != 0:
            ae_ax.set_yticklabels([])
            ae_ax2.set_yticklabels([])

        for task_no in range(TOTAL_TASKS):
            with plt.rc_context({'path.sketch': (1, 20, 1)}):
                domain_ax.axvline(task_no * STEPS_PER_TASK,
                                  linestyle="solid",
                                  color="grey",
                                  alpha=0.3)

            blended_transform = transforms.blended_transform_factory(
                domain_ax.transData, domain_ax.transAxes)
            domain_ax.text((task_no * STEPS_PER_TASK) + 3000, 0.92, "$M_%s$" % (task_no),
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

            alg_x, alg_y, alg_yerr = get_x_y(data, domain["source"], alg["name"],
                                 "Avg. Reward")
            alg_ax.plot(alg_x, alg_y, label=alg["label"],
                      color=alg["color"],
                      linestyle=alg["linestyle"],
                        alpha=0.8)
            alg_ax.fill_between(alg_x, alg_y - alg_yerr, alg_y + alg_yerr, color=alg["color"],
                                alpha=AREA_ALPHA)

            o_x, o_y, o_yerr = get_x_y(data, domain["source"], "oracle",
                                 "Avg. Reward")

            alg_ax.scatter(markers_on * 100,
                [np.mean(o_y[i // 1000 * 1000:i // 1000 * 1000 + 1000]) for i in markers_on],
                           color=ORACLE["color"],
                           marker="x",
                           linewidths=1,
                           zorder=10,
                           s=ORACLE_MARKERSIZE)

            alg_no += 1

            transform_ticklabels(alg_ax)

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
            domain_ax.set_ylabel("(a) $\#$ of Tasks\nAccomplished", fontsize=16)

        domain_ax.set_ylim([-2.5 * 1000, 120 * 1000])

    legend_order = ["oracle", "qlearning", "drift", "qace"]
    legend_labels = []
    legend_handles = []
    handler_map = {}
    for legend in legend_order:
        for alg in ALGS:
            if legend == alg["name"]:

                if "bar_handle" not in alg:
                    legend_handles.append((alg["handle"][0], ))
                else:
                    legend_handles.append((alg["bar_handle"], alg["handle"][0]))

                if legend == "oracle":
                    handler_map[legend_handles[-1]] = HandlerTuple(ndivide=2,
                                                                   pad=-10)
                else:
                    handler_map[legend_handles[-1]] = HandlerTuple(ndivide=None,
                                                                   pad=0.3)
                legend_labels.append(alg["label"])
                break
    fig.legend(handles=legend_handles,
               labels=legend_labels,
               ncols=4,
               handler_map=handler_map,
               handlelength=4,
               loc="center",
                bbox_to_anchor=(0.51, 0.96), frameon=False, fontsize=14)

    fig.savefig("%s/plots.png" % (RESULTS_DIR),  bbox_inches="tight")

    pass
    pass