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

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })

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

def get_data(filepath, xkey, ykey, alg, metric):
    

    fh = open(filepath, "r")
    reader = csv.DictReader(fh, delimiter=";")

    x = []
    y = []
    drift_sample = 0
    
    for row in reader:

        x_value = ast.literal_eval(row[xkey])
        y_value = ast.literal_eval(row[ykey])
        if isinstance(y_value, tuple):

            y_value = y_value[0]

        # x_value, drift_sample = get_interaction_adjustment(xkey,
        #                                                   alg,
        #                                                   x_value, 
        #                                                   drift_sample)

        # y_value, drift_sample = get_interaction_adjustment(ykey,
        #                                                   alg,
        #                                                   y_value, 
        #                                                   drift_sample)

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
    
    fig = plt.figure(constrained_layout=False, figsize=(20, 8))
    gs = gridspec.GridSpec(ncols=4, nrows=2, figure=fig,
                           hspace=0.1,wspace=0.06)
    
    metrics =  ["Total Goals Reached",
                "Avg. Reward",
                # "Avg. Success Rate"
                ]
    
    PLOTTERS = [
            ("oracle", "Oracle (Best Possible Performance)", "grey", "dashdot", 1, "--"),
            ("qlearning", "Q Learning", "grey", "dashed", 1, "--"),
            ("qace", "Unknown Drift + Comprehensive", "green", "dashed", 1, "--"),
            ("qace-stateless", "Known Drift + Comprehensive", "red", "dashed", 1, "--"),
            ("drift", "Unknown Drift + Need-based (Ours)", "blue", "solid", 1, "--"),
        ]
    
    for column, domain in enumerate(domains):
        
        title = get_title_for_domain(domain)
        r1 = fig.add_subplot(gs[0, column])
        r2 = fig.add_subplot(gs[1, column])
        # r3 = fig.add_subplot(gs[2, column])
        
        axes = [r1, r2]
        
        r1.set_title(title, fontsize=16)

        r1.tick_params(axis='both', labelsize=14)
        r2.tick_params(axis='both', labelsize=14)
        # r3.tick_params(axis='both', labelsize=14)

        for axis in [r1]:
            
            axis.set_xticklabels([])

        # for axis in [r1, r2]:
            
            r2.set_yticklabels([])
            r2.set_yticks([-45,0,18,60,78,120,138,180,198,240], [-45,0,-45,0,-45,0,-45,0,-45,0], fontsize=10)
            # r3.set_yticklabels([])
            # r3.set_yticks([0,1,2,3,4,5,6,7,8,9], [0,1,0,1,0,1,0,1,0,1])

        if column == 0:
            
            r1.set_ylabel("Total Tasks\nAccomplished", fontsize=18)
            r2.set_ylabel("Avg. Reward", fontsize=18)
            # r3.set_ylabel("Avg. Success\nRate", fontsize=18)
        else:
            for ax in axes:
                ax.set_yticklabels([])
            
        r1.set_ylim([0, 120000])
        r2.set_ylim([-50, 260])
        # r3.set_ylim([-0.5, 9.5])

            
        bin_data = domain_data[domain]
        cost, sr = get_ground_truth_cost_sr_for_domain(domain)
        
        for ax, metric in zip(axes, metrics):
            i=0
            for alg, label, color, linestyle, opacity, _ in PLOTTERS:
                i+=1
                x = np.asarray(bin_data[alg][X_METRIC])
                y = np.asarray(bin_data[alg][metric]["data"])
                yerr = np.asarray(bin_data[alg][metric]["std"])

                # if "goals reached" not in metric.lower():
                #     point = int(i * 1000 * 100 / 2)
                #     # i+= 1
                #     label = bin_data[alg][metric]["data"][point]
                    

                if "reward" in metric.lower():
                    if alg == "oracle":
                        y += 240
                        # ax.annotate(label, (point, 0), fontsize=8,
                        #         xytext=(4, 8), textcoords="offset points",
                        #         ha='center', va='top')
                        # yerr += 240
                    elif alg == "drift":
                        
                        y+= 180
                        # ax.annotate(label, (point, 0), fontsize=8,
                        #         xytext=(4, 12), textcoords="offset points",
                        #         ha='center', va='top')
                        # yerr += 180
                    elif alg == "qlearning":
                        
                        y+= 120
                        # yerr += 120
                    elif alg == "qace":
                        
                        y+= 60
                        # yerr += 60
                    elif alg == "qace-stateless":
                        
                        # suffix = "qace-stateless_steps_vs_tasks.csv"
                        y+= 0
                        # yerr += 0
                    else:
                        assert False

                if "success" in metric.lower():
                    if alg == "oracle":
                        y += 8
                        # yerr += 8
                    elif alg == "drift":
                        
                        y+= 6
                        # yerr += 6
                    elif alg == "qlearning":
                        
                        y+= 4
                        # yerr += 4
                    elif alg == "qace":
                        
                        y+= 2
                        # yerr += 2
                    elif alg == "qace-stateless":
                        
                        # suffix = "qace-stateless_steps_vs_tasks.csv"
                        y+= 0
                        # yerr += 0
                    else:
                        assert False
                
                # if ax == r4:
                #
                #     ax.scatter([-50], [cost], marker="*", color="black")
                               

                # if "goals reached" in metric.lower():
                ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.0, alpha=opacity)
                # else:
                #     ax.scatter(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.0, alpha=opacity)

                # if "goals reached" in metric.lower():
                ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.09)

        if column == 0:

            ylabels = r1.get_yticks()
            new_yticklabels = []
            for ylabel in ylabels:
                try:
                    value = float(ylabel)
                    if value > 1000:
                        value = value / 1000
                        value = str("%sk" % (int(value)))
                    else:
                        value = str(value)
                    new_yticklabels.append(value)
                except Exception:

                    new_yticklabels.append(ylabel)

            r1.set_yticklabels(new_yticklabels)

        xlabels = r2.get_xticks()
        new_xticklabels = []
        for xlabel in xlabels:
            try:
                value = float(xlabel)
                if abs(value) > 1000:
                    value = value / 1000
                    value = str("%sk" % (int(value)))
                else:
                    value = str(value)
                new_xticklabels.append(value)
            except Exception:

                new_xticklabels.append(xlabel)

        r2.set_xticklabels(new_xticklabels)

        for ax in [r1, r2]:
            for x in range(0, 5):
                ax.axvline(x * 1000 * 100, linestyle="dotted",
                           alpha=0.1)
                
            

    fig.legend(
        labels=["Oracle (Best Possible Performance)", "_b",
                "Q-Learning", "_b",
                "UC-Learner", "_b",
                "AC-Learner", "_b",
                "CPL (Ours)", "_b"],
               loc="center", ncols=5,
                bbox_to_anchor=(0.5, 0.95), frameon=False, fontsize=16)

    fig.text(0.5, 0.043, "Total Steps", fontsize=18)
    
    # fig.savefig("%s/plots.pdf" % (results_dir), bbox_inches="tight")
    fig.savefig("%s/supplemental.png" % (results_dir), bbox_inches="tight")
            

if __name__ == "__main__":
    
    BASE_DIR = "/tmp/results/"
    RESULTS_DIR = "/tmp/results/"
    START_RUN = 0
    END_RUN = 9
    NUM_BINS = 500000
    
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
                # "Avg. Success Rate"
                ]
    metrics_default_values = [0, 0]
    
    
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
    

    import pickle
    import os
    if os.path.exists("/tmp/data.pkl"):
        with open("/tmp/data.pkl", "rb") as fh:
            domain_data = pickle.load(fh)
    else:
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
                                                              X_METRIC, metric, "qlearning", metric)

                    data["qace"][X_METRIC][run_no], data["qace"][metric][run_no] = get_data(qace_file,
                                                              X_METRIC, metric, "qace", metric)

                    data["qace-stateless"][X_METRIC][run_no], data["qace-stateless"][metric][run_no] = get_data(qace_stateless_file,
                                                              X_METRIC, metric, "qace-stateless", metric)

                    data["drift"][X_METRIC][run_no], data["drift"][metric][run_no] = get_data(drift_file,
                                                              X_METRIC, metric, "drift", metric)

            max_x = compute_max_for_matric(data, algs, X_METRIC, RUNS)
            print(max_x)
            NUM_BINS = max_x
            BIN_SIZE = int(max_x // NUM_BINS)
            BIN_SIZE = max(1, max_x)
            BIN_SIZE = 1


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

            with open("/tmp/data.pkl", "wb") as fh:
                pickle.dump(domain_data, fh)


    plot_supplemental(RESULTS_DIR, DOMAINS, domain_data, NUM_BINS)
    

        
    