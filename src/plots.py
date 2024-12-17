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
    
    idx = get_last_range(x)
    
    x[0] = default_value if x[0] == float("inf") else x[0]
    for i in range(idx):
        
        if x[i] == float("inf"):
            x[i] = x[i - 1]

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
    
    if domain == "Tireworld":
        
        return domain
    elif domain == "Explodingblocks":
        
        return "ExplodingBlocks"
    elif domain == "Probabilistic_elevators":
        
        return "Elevators"
    elif domain == "First_responders":
        
        return "FirstResponders"
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
    
    fig = plt.figure(constrained_layout=False, figsize=(20, 10))
    gs = gridspec.GridSpec(ncols=4, nrows=5, figure=fig,
                           hspace=0.1,wspace=0.06)
    
    metrics =  ["Variational Difference (Ground Truth)",
                "Elapsed Time",
                "LAO (Aggregate Success Rate)",
                "LAO (Aggregate Costs)",
                "Success Rate"]
    
    PLOTTERS = [
            ("IPML-R", "IPML-R (Ours)", "blue", "-"),
            ("IPML-S", "IPML-S (Ours)", "green", "-."),
            ("GLIB_G1", "GLIB (G1)", "black", "--"),
            ("GLIB_L2", "GLIB (L2)", "red", ":"),
        ]
    
    for column, domain in enumerate(domains):
        
        title = get_title_for_domain(domain)
        r1 = fig.add_subplot(gs[0, column])
        r2 = fig.add_subplot(gs[1, column])
        r3 = fig.add_subplot(gs[2, column])
        r4 = fig.add_subplot(gs[3, column])
        r5 = fig.add_subplot(gs[4, column])
        
        axes = [r1, r2, r3, r4, r5]
        
        r1.set_title(title, fontsize=15)
        
        for axis in [r1, r2, r3, r4]:
            
            axis.set_xticklabels([])
        
        if column == 0:
            
            r1.set_ylabel("Variational\nDistance")
            r2.set_ylabel("Learning\nTime (s)")
            r3.set_ylabel("LAO*\nSuccess Rate")
            r4.set_ylabel("LAO* Costs")
            r5.set_ylabel("FFReplan\nSuccess Rate")
        else:
            for ax in axes:
                ax.set_yticklabels([])
            
        r1.set_ylim([-0.05, 1.1])
        r2.set_ylim([0, 4000])
        r3.set_ylim([-0.05, 1.1])
        r4.set_ylim([0, 52])
        r5.set_ylim([-0.05, 1.1])
            
        bin_data = domain_data[domain]
        cost, sr = get_ground_truth_cost_sr_for_domain(domain)
        
        for ax, metric in zip(axes, metrics):
            for alg, label, color, linestyle in PLOTTERS:
                
                x = np.asarray(bin_data[alg][X_METRIC])
                y = np.asarray(bin_data[alg][metric]["data"])
                yerr = np.asarray(bin_data[alg][metric]["std"])
                
                # if ax == r4:
                #
                #     ax.scatter([-50], [cost], marker="*", color="black")
                               
                
                ax.plot(x, y, label=label, color=color, linestyle=linestyle)
                ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.09)
                

    fig.legend(
        labels=["IPML-R (Ours)", "_b", 
                "IPML-S (Ours)", "_b",
                "GLIB-G", "_b",
                "GLIB-L", "_b"], 
               loc="center", ncols=4,
                bbox_to_anchor=(0.5, 0.93), frameon=False, fontsize=13)

    fig.text(0.4, 0.063, "# of Samples (s, a, s') Used for Learning", fontsize=13.0)
    
    fig.savefig("%s/supplemental.pdf" % (results_dir), bbox_inches="tight")
    fig.savefig("%s/supplemental.png" % (results_dir), bbox_inches="tight")
            

if __name__ == "__main__":
    
    BASE_DIR = "/home/rushang/work/git/stochastic-AIA/results/"
    RESULTS_DIR = "/home/rushang/work/git/stochastic-AIA/results"
    START_RUN = 0
    END_RUN = 9
    
    algs = ["IPML-R", "IPML-S", "GLIB_G1", "GLIB_L2"]
    
    SECONDARY_Y_METRIC = "Interaction"
    X_METRIC = "Elapsed Time"
    X_AXIS_LABEL ="Time (s)"
    SECONDARY_Y_AXIS_LABEL = "# of Samples"
    SECONDARY_Y_AXIS_LEGEND_LABEL = "Sample Count"
    
    SECONDARY_Y_METRIC = "Elapsed Time"
    X_METRIC = "Interaction"
    X_AXIS_LABEL = "# of Samples (s, a, s') Used for Learning"
    SECONDARY_Y_AXIS_LABEL = "Learning Time (s)"
    SECONDARY_Y_AXIS_LEGEND_LABEL = "Learning Time"
    
    metrics =  ["Variational Difference (Ground Truth)",
                "Elapsed Time",
                "LAO (Aggregate Success Rate)",
                "LAO (Aggregate Costs)",
                "Success Rate"]
    metrics_default_values = [1.0, 0, 0]
    
    
    fig = plt.figure(constrained_layout=False, figsize=(20, 5))
    gs = gridspec.GridSpec(ncols=4, nrows=2, figure=fig,
                           hspace=0.1,wspace=0.06)
    
    PLOTTERS = [
            ("IPML-R", "IPML-R (Ours)", "blue", "-", "--"),
            # ("IPML-S", "IPML-S (Ours)", "red", "-", "--"),
            ("GLIB_G1", "GLIB (G1)", "grey", "-", "--"),
            # ("GLIB_L2", "GLIB (L2)", "green", "-", "--"),
        ]
    
    RUNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    DOMAINS = ["Tireworld", "Explodingblocks", "Probabilistic_elevators",
                    "First_responders"]
    domain_data = {}
    
    for column, DOMAIN  in enumerate(DOMAINS):
        data = {}
        for alg in algs:
            data[alg] = {}
            for metric in metrics:
                
                data[alg][metric] = {}
                data[alg][X_METRIC] = {}
        
        for run_no in RUNS:
            
            ipml_r_file = "%s/run%u/%s/ipml_randomized.csv" % (BASE_DIR, run_no, 
                                                                 DOMAIN)
            
            ipml_s_file = "%s/run%u/%s/ipml_sequential.csv" % (BASE_DIR, run_no, 
                                                                 DOMAIN)
            
            g1_file = "%s/run%u/%s/glib_g1_lndr.csv" % (BASE_DIR, run_no, 
                                                                 DOMAIN)
            g2_file = "%s/run%u/%s/glib_l2_lndr.csv" % (BASE_DIR, run_no, 
                                                                 DOMAIN)
            
            for metric in metrics:
                
                data["IPML-R"][X_METRIC][run_no], data["IPML-R"][metric][run_no] = get_data(ipml_r_file, 
                                                          X_METRIC, metric, "IPML-R", metric)
                
                data["IPML-S"][X_METRIC][run_no], data["IPML-S"][metric][run_no] = get_data(ipml_s_file, 
                                                          X_METRIC, metric, "IPML-S", metric)
                
                data["GLIB_G1"][X_METRIC][run_no], data["GLIB_G1"][metric][run_no] = get_data(g1_file, 
                                                          X_METRIC, metric, alg, metric)
                
                data["GLIB_L2"][X_METRIC][run_no], data["GLIB_L2"][metric][run_no] = get_data(g2_file, 
                                                          X_METRIC, metric, alg, metric)
            
            
        max_x = compute_max_for_matric(data, algs, X_METRIC, RUNS)
        print(max_x)
        NUM_BINS = 30
        
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
    
    #     title = get_title_for_domain(DOMAIN)
    #
    #     r1 = fig.add_subplot(gs[0, column])
    #     r2 = fig.add_subplot(gs[1, column])
    #
    #     r1.set_title(title, fontsize=15)
    #
    #     if column == 0:
    #         r1.set_ylabel("Variational Distance", fontsize=13.0)
    #         r2.set_ylabel("Success Rate", fontsize=13.0)
    #     else:
    #
    #         r1.set_yticklabels([])
    #         r2.set_yticklabels([])
    #
    #     r1.set_xticklabels([])
    #
    #
    #
    #     r1_time = r1.twinx()
    #     r2_time = r2.twinx()
    #
    #
    #     if column != 3:
    #
    #         r1_time.set_yticklabels([])
    #         r2_time.set_yticklabels([])
    #
    #     vd_lines = []
    #
    #     for alg, label, color, l1style, l2style in PLOTTERS:
    #         line = r1.plot(bin_data[alg][X_METRIC], 
    #                 bin_data[alg]["Variational Difference (Ground Truth)"]["data"],
    #                 label=label,
    #                 color=color,
    #                 linestyle=l1style)
    #
    #         x = np.asarray(bin_data[alg][X_METRIC])
    #         y = np.asarray(bin_data[alg]["Variational Difference (Ground Truth)"]["data"])
    #         yerr = np.asarray(bin_data[alg]["Variational Difference (Ground Truth)"]["std"])
    #
    #
    #
    #         line = r1_time.plot(bin_data[alg][X_METRIC], 
    #                 bin_data[alg][SECONDARY_Y_METRIC]["data"],
    #                 label=label,
    #                 color=color,
    #                 linestyle=l2style)
    #
    #         vd_lines.append(line)
    #
    #         r1.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.09)
    #
    #
    #         r2.plot(bin_data[alg][X_METRIC], 
    #                 bin_data[alg]["LAO (Aggregate Success Rate)"]["data"],
    #                 label=label,
    #                 color=color,
    #                 linestyle=l1style)
    #
    #         x = np.asarray(bin_data[alg][X_METRIC])
    #         y = np.asarray(bin_data[alg]["LAO (Aggregate Success Rate)"]["data"])
    #         yerr = np.asarray(bin_data[alg]["LAO (Aggregate Success Rate)"]["std"])
    #
    #         r2.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.09)
    #
    #         r2_time.plot(bin_data[alg][X_METRIC], 
    #                 bin_data[alg][SECONDARY_Y_METRIC]["data"],
    #                 label=label,
    #                 color=color,
    #                 linestyle=l2style)
    #
    #
    #         r1.set_ylim([-0.05, 1.1])
    #         r2.set_ylim([-0.05, 1.1])
    #         #
    #         r1_time.set_ylim([0, 4000])
    #         r2_time.set_ylim([0, 4000])
    #
    # fig.text(0.4, 0.02, X_AXIS_LABEL, fontsize=13.0)
    #
    #
    # fig.legend(labels=["IPML-R (Ours)", "_b", "GLIB-G"], loc="center", ncols=2,
    #            bbox_to_anchor=(0.42, 1.00), frameon=False, fontsize=13)
    #
    # fig.legend(labels=["_", "_b", "_GLIB-G", "_ASF", "_ASF", "_ASF", "_ASF", "_ASF", "IPML-R (Ours)", "GLIB-G"], loc="center", ncols=2,
    #            bbox_to_anchor=(0.72, 1.00), frameon=False, fontsize=13)
    #
    # fig.text(0.18, 0.99, "Variational Distance/Success Rate", fontsize=13)
    # fig.text(0.57, 0.99, SECONDARY_Y_AXIS_LEGEND_LABEL, fontsize=13)
    # fig.text(0.93, 0.35, SECONDARY_Y_AXIS_LABEL, rotation=90, fontsize=13)
    #
    # # plt.show()
    #
    # fig.savefig("%s/ipml_results.pdf" % (RESULTS_DIR), bbox_inches="tight")
    # fig.savefig("%s/ipml_results.png" % (RESULTS_DIR), bbox_inches="tight")
    # ipml = "%s/run0/%s/ipml_randomized.csv" % (BASE_DIR, DOMAIN)
    # glib = "%s/run0/%s/glib_g1_lndr.csv" % (BASE_DIR, DOMAIN)
    #
    #
    # fig, ax2 = plt.subplots()
    #
    # x, y = get_data(ipml, "Elapsed Time", 
    #                 "Variational Difference (Ground Truth)")
    #
    # glib_x, glib_y = get_data(glib, "Elapsed Time", 
    #                 "Variational Difference (Ground Truth)")
    #
    # BIN_SIZE = 30
    # max_size = max(x + glib_x)
    # num_bins = int(max_size // BIN_SIZE) + 1
    #
    # x, y = convert_to_bins(num_bins, BIN_SIZE, x, y)
    # y = average_out(y)
    # y = no_change(y, 1.0)
    #
    # glib_x, glib_y = convert_to_bins(num_bins, BIN_SIZE, glib_x, glib_y)
    # glib_y = average_out(glib_y)
    # glib_y = no_change(glib_y, 1.0)
    #
    #
    # ax2.plot(x, y, label="IPML-R", color="blue")
    # ax2.plot(glib_x, glib_y, label="GLIB (G1)", color="grey")
    # ax2.set_title("Tireworld")
    # ax2.set_xlabel("Elapsed learning time (units of 20 seconds)")
    # ax2.set_ylabel("Variational Distance")
    #
    #
    # ax = ax2.twinx() 
    # x, y = get_data(ipml, "Elapsed Time", 
    #                 "Interaction")
    #
    # glib_x, glib_y = get_data(glib, "Elapsed Time", 
    #                 "Interaction")
    #
    #
    # x, y = convert_to_bins(num_bins, BIN_SIZE, x, y)
    # y = sum_out(y)
    # y = no_change(y, 0.0)
    #
    # glib_x, glib_y = convert_to_bins(num_bins, BIN_SIZE, glib_x, glib_y)
    # glib_y = max_out(glib_y)
    # glib_y = no_change(glib_y, 0)
    # print(glib_y)
    #
    # ax.plot(x, y, label="IPML-R", linestyle="--", color="blue")
    # ax.plot(glib_x, glib_y, label="GLIB (G1)", linestyle="--", color="grey")
    # ax.set_xlabel("Elapsed learning time (units of 20 seconds)")
    # ax.set_ylabel("# Samples")


        
    # for i in range(1, len(new_y)):
    #
    #     new_y[i] += new_y[i - 1]
    
    # ax = ax2.twinx() 
    #
    # fh = open("%s/run0/%s/ipml_randomized.csv" % (BASE_DIR, DOMAIN), "r")
    # ipml_r_fh = csv.DictReader(fh, delimiter=";")
    #
    # #     new_y[i] += new_y[i - 1]
    #
    #
    #
    #
    # # ax.scatter(new_x, new_y, label="IPML-R (samples)")
    # ax.set_ylabel("Total samples generated")
    
    ####
    
    # fig.legend()
    #
    # plt.show()
        
    