import sys
import os

"""
Example use: 
1. python3 run_status.py ~/Code/differential-learning-private/ elevators
2. python3 run_status.py ~/Code/differential-learning-private/ 
"""

algos = ["drift", "oracle", "qace", "qlearning", "qace-stateless"]

domains = ["elevators"]

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
    