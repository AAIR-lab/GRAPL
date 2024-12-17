import wandb
import sys
import os
import pandas as pd


FILENAME = "~/results/"
# loaded_experiment_df = pd.read_csv(FILENAME)

PROJECT_NAME = "ConvertedExperiments"


algos = ["drift", "oracle", "qace", "qlearning"]

domains = ["first_responders", "tireworld"]

TAGS_COL = "Task Number"    
METRIC_COLS =["Task Name", "Task Number", "Total Steps", "Total Goals Reached", "Avg. Success Rate", "Avg. Reward"]


if __name__ == "__main__":

    if len(sys.argv) != 2 and  len(sys.argv) != 3:
        print("Usage: python run_status.py <directory> <domain_name>")
        sys.exit(1)

    directory = sys.argv[1]
    if len(sys.argv) == 3:
        domains = [sys.argv[2]]

    assert(os.path.isdir(directory))
    failed_runs = []
    still_running = []
    successful_runs = []
    success_count = 0
    failure_count = 0
    incomplete_count = 0

    for domain in domains:
        domain_dir = os.path.abspath(directory) + "/" + domain
        for dir in sorted(os.listdir(domain_dir)):
            if dir[:-1] != "run":
                continue
            for algo in algos:
                tt = False
                all_success = False
                failed_algo_run = []
                file_name = domain_dir + "/" + dir + "/" + algo + "-run_info" + ".txt"
                file_handle = open(file_name, "r")
                lines = []
                line_dict = {}
                
                for line in file_handle:
                    line = line.strip()
                    if ": False, " in line:
                        task = line.split(":")[0]
                        failing_run_string = "Domain:" + domain + " | Run:" + dir + " | Algorithm:" + algo + " | Task:" + task
                        failed_algo_run.append(failing_run_string)
                    elif "Time Taken:" in line:
                        time_taken = line.split(":")[1].strip()
                        tt = True
                    elif "all_succeeded: True" in line:
                        all_success = True
                
                if tt and all_success:
                    successful_runs.append("Domain:" + domain + " | Run:" + dir + " | Algorithm:" + algo)
                    success_count += 1
                    loaded_experiment_df = pd.read_csv(domain_dir + "/" + dir + "/" + algo + "_steps_vs_tasks.csv", sep = ';')
                    run_name =  algo
                    run = wandb.init(
                            project=PROJECT_NAME, name=domain+"_"+run_name+"_"+dir[-1], config={"domain": domain, "run": dir[-1], "algorithm": algo}, 
                            )
                    wandb.log({"domain": domain, "run": dir[-1], "algorithm": algo})
                    for i, row in loaded_experiment_df.iterrows():
                        
                        # tags = row[TAGS_COL]

                        metrics = {}
                        for metric_col in METRIC_COLS:
                            metrics[metric_col] = row[metric_col]

                        # for key, val in metrics.items():
                        #     if isinstance(val, list):
                        #         for _val in val:
                        #             run.log({key: _val})
                        #     else:
                        wandb.log(metrics)
                            

                        # run.summary.update(summaries)
                    wandb.finish()
                        
                    
                    # print("CSV: ", domain_dir + "/" + dir + "/" + algo + "_steps_vs_tasks.csv")


                elif tt and not all_success:
                    failure_count += 1
                    failed_runs.extend(failed_algo_run)
                elif not tt:
                    still_running.append("Domain:" + domain + " | Run:" + dir + " | Algorithm:" + algo)
                    incomplete_count += 1

                        
                file_handle.close()

    print("Status Report:\n--------------------")
    print(incomplete_count, "Incomplete Runs")
    print(failure_count, "Failed Runs")
    print(success_count, "Successful Runs")
    print("--------------------\n")

    print("\nIncomplete Runs:\n----------------")
    print("\n".join(still_running))

    print("\nFailed Runs:\n------------")
    print("\n".join(failed_runs))

    print("\nSuccessful Runs:\n----------------")
    print("\n".join(successful_runs))


# EXPERIMENT_NAME_COL = "Experiment"
# NOTES_COL = "Notes"
# TAGS_COL = "Tags"
# CONFIG_COLS = ["Num Layers"]
# SUMMARY_COLS = ["Final Train Acc", "Final Val Acc"]
# METRIC_COLS = ["Training Losses"]

# for i, row in loaded_experiment_df.iterrows():
#     run_name = row[EXPERIMENT_NAME_COL]
#     notes = row[NOTES_COL]
#     tags = row[TAGS_COL]

#     config = {}
#     for config_col in CONFIG_COLS:
#         config[config_col] = row[config_col]

#     metrics = {}
#     for metric_col in METRIC_COLS:
#         metrics[metric_col] = row[metric_col]

#     summaries = {}
#     for summary_col in SUMMARY_COLS:
#         summaries[summary_col] = row[summary_col]

#     run = wandb.init(
#         project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
#     )

#     for key, val in metrics.items():
#         if isinstance(val, list):
#             for _val in val:
#                 run.log({key: _val})
#         else:
#             run.log({key: val})

#     run.summary.update(summaries)
#     run.finish()