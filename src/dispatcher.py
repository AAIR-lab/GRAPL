import concurrent.futures
import os
import subprocess
import time
from utils import time_thread
import config

import argparse
from utils.file_utils import FileUtils

def submit_task(cmd,  env=None, log_file=None, timeout=None):

    fh = open(log_file, "w")
    subprocess.call(cmd, stdout=fh, stderr=fh,
                    env=env,
                    timeout=timeout)
    fh.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Job dispatcher")

    parser.add_argument("--base-dir", required=True, type=str,
                   help="The base directory for the experiments")

    parser.add_argument("--task-dir", required=True, type=str,
                   help="The path to the task files")

    parser.add_argument("--total-runs", required=True, type=int,
                        help="The total number of runs")

    parser.add_argument("--max-workers", type=int, default=None,
                        help="The maximum number of cores to use.")

    parser.add_argument("--max-steps", default=10000, type=int,
                        help="The simulator step budget")

    parser.add_argument("--explore-mode", type=str,
                        default="random_walk",
                        choices=["random_walk"],
                        help="The exploration method.")

    parser.add_argument("--sampling-count", type=int,
                        default=100,
                        help="The sampling count")

    parser.add_argument("--num-simulations", type=int,
                        default=50,
                        help="The number of simulations to perform")

    args = parser.parse_args()

    FileUtils.initialize_directory(args.base_dir, clean=True)

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers)

    start_time = time.time()
    futures = []

    time_thread = time_thread.TimeThread(time_limit_in_sec=float("inf"),
                                         leave=True)
    time_thread.start()

    for run_no in range(args.total_runs):

        base_dir = "%s/run%u" % (args.base_dir, run_no)
        FileUtils.initialize_directory(base_dir, clean=True)

        for alg in ["oracle", "qace", "drift", "qlearning"]:

            cmd = []
            cmd.append("python3")
            cmd.append("src/main.py")
            cmd.append("--debug")
            cmd.append("--base-dir")
            cmd.append(base_dir)
            cmd.append("--task-dir")
            cmd.append(args.task_dir)
            cmd.append("--algorithm")
            cmd.append(alg)
            cmd.append("--max-steps")
            cmd.append("%s" % (args.max_steps))
            cmd.append("--fail-fast")

            cmd.append("--explore-mode")
            cmd.append(args.explore_mode)

            cmd.append("--num-simulations")
            cmd.append(str(args.num_simulations))

            cmd.append("--sampling-count")
            cmd.append(str(args.sampling_count))

            log_file = "%s/%s_logs.txt" % (base_dir, alg)

            future = executor.submit(submit_task,
                                     cmd,
                                     env=os.environ.copy(),
                                     log_file=log_file,
                                     timeout=None)

            futures.append(future)
    executor.shutdown()

    for future in concurrent.futures.as_completed(futures):

        pass

    time_thread.stop()
    time_thread.join()
    print("Took %.2f seconds" % (time.time() - start_time))


