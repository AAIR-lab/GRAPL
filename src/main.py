import config
import argparse
import os
import sys
import glob

from utils.file_utils import FileUtils
from evaluators.q_evaluator import QLearningEvaluator
from evaluators.gt_evaluator import GTEvaluator
from evaluators.qace_evaluator import QaceEvaluator
from evaluators.drift_evaluator import  DriftEvaluator
from evaluators.qace_stateless import QaceStatelessEvaluator
from evaluators.random_evaluator import RandomEvaluator
from multiprocessing import Process

import random
import time
import traceback

class Task:

    def __init__(self, domain_file, problem_file):
        self.domain_file = domain_file
        self.problem_file = problem_file


def setup_base_dir(base_dir, clean):
    assert not os.path.exists(base_dir) or os.path.isdir(base_dir)
    FileUtils.initialize_directory(base_dir, clean=clean)


def get_default_tasks(domain_name="tireworld"):
    tasks = []
    tasks_dir = "%s/%s" % (config.BENCHMARKS_DIR, domain_name)

    domain_files = glob.glob("%s/*domain*.pddl" % (tasks_dir))
    assert len(domain_files) == 1
    domain_file = domain_files[0]

    problem_files = glob.glob("%s/*.pddl" % (tasks_dir))
    for problem_file in problem_files:

        if problem_file != domain_file:
            tasks.append(Task(domain_file, problem_file))

    return tasks

def get_taskable_tasks(task_dir):

    task_dir = os.path.abspath(task_dir)
    tasks = []
    domain_files = sorted(glob.glob("%s/*domain*pddl" % (task_dir)))
    for domain_file in domain_files:


        problem_file = domain_file.replace("domain-", "task-")
        tasks.append(Task(domain_file, problem_file))
        # problem_file = problem_file.replace(".pddl", "-t*.pddl")
        # for problem_file in sorted(glob.glob("%s" % (problem_file))):
        #
        #     tasks.append(Task(domain_file, problem_file))

    return tasks


# from multiprocessing.managers import BaseManager
#
#
# class MyManager(BaseManager):
#     pass
#
#
# MyManager.register("QLearningEvaluator", QLearningEvaluator)
# MyManager.register("GTEvaluator", GTEvaluator)

def run_algorithm(fh, algorithm, tasks, fail_fast=False, **kwargs):

    all_succeeded = True
    for task_no, task in enumerate(tasks):

        task_name = "task-t%u" % (task_no)
        algorithm.switch_to_last_good_model()
        success = True

        start_time = time.time()
        try:
            algorithm.switch_to_last_good_model()
            algorithm.solve_task(task_no, task.domain_file, task.problem_file,
                                 **kwargs)
        except Exception as e:

            if fail_fast:

                raise e

            traceback.print_exception(e)
            success = False

        all_succeeded &= success
        fh.write("%s: %s, %.2f\n" % (task_name, success,
                                     time.time() - start_time))

    algorithm.shutdown()
    fh.write("all_succeeded: %s\n" % (all_succeeded))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Differential Learning")

    parser.add_argument("--base-dir", required=True, type=str,
                        help="The base directory")

    parser.add_argument("--clean", default=False, action="store_true",
                        help="Delete all files in the base-dir")

    parser.add_argument("--algorithm", required=True,
                        choices=["qlearning", "oracle", "qace", "drift",
                                 "qace-stateless", "random"],
                        help="The algorithm to run")

    parser.add_argument("--debug", default=False, action="store_true",
                        help="Enable debug")

    parser.add_argument("--max-steps", default=5000, type=int,
                        help="The simulator step budget")

    parser.add_argument("--domain-file", type=str,
                        default=None, help="Path to the domain file")

    parser.add_argument("--problem-file", type=str,
                        default=None, help="Path to the domain file")

    parser.add_argument("--fail-fast", default=False,
                        action="store_true",
                        help="Fail fast mode.")

    parser.add_argument("--task-dir", type=str,
                        help="The path to the tasks dir.")

    parser.add_argument("--explore-mode", type=str,
                        default="random_walk",
                        choices=["random_walk"],
                        help="The exploration method.")

    parser.add_argument("--sampling-count", type=int,
                        default=5,
                        help="The sampling count")

    parser.add_argument("--num-simulations", type=int,
                        default=50,
                        help="The number of simulations to perform")

    parser.add_argument("--num-rw-tries", default=100,
                        type=int,
                        help="The number of random walk tries per iteration.")

    args = parser.parse_args()

    args.base_dir = os.path.abspath(args.base_dir)
    setup_base_dir(args.base_dir, args.clean)

    if args.domain_file is not None:

        assert args.problem_file is not None

        args.domain_file = os.path.abspath(args.domain_file)
        args.problem_file = os.path.abspath(args.problem_file)
        tasks = [Task(args.domain_file, args.problem_file)]
    elif args.task_dir is not None:

        tasks = get_taskable_tasks(args.task_dir)
    else:

        assert False

    # manager = MyManager()
    # manager.start()

    seed = int(time.time())
    random.seed(seed)
    print("Using seed", seed)

    run_info_file = "%s/%s-run_info.txt" % (args.base_dir, args.algorithm)
    fh = open(run_info_file, mode="w", buffering=1)
    fh.write("Seed: %s\n" % (seed))

    start_time = time.time()
    if args.algorithm == "qlearning":

        algorithm = QLearningEvaluator(
            args.base_dir,
            num_simulations=args.num_simulations,
            debug=args.debug)
    elif args.algorithm == "oracle":

        algorithm = GTEvaluator(
            args.base_dir,
            num_simulations=args.num_simulations,
            debug=args.debug)
    elif args.algorithm == "qace":

        algorithm = QaceEvaluator(
            args.base_dir,
            num_simulations=args.num_simulations,
            sampling_count=args.sampling_count,
            explore_mode=args.explore_mode,
            num_rw_tries=args.num_rw_tries,
            debug=args.debug)
    elif args.algorithm == "qace-stateless":

        algorithm = QaceStatelessEvaluator(
            args.base_dir,
            num_simulations=args.num_simulations,
            sampling_count=args.sampling_count,
            explore_mode=args.explore_mode,
            num_rw_tries=args.num_rw_tries,
            debug=args.debug)
    elif args.algorithm == "drift":

        algorithm = DriftEvaluator(
            args.base_dir,
            num_simulations=args.num_simulations,
            sampling_count=args.sampling_count,
            explore_mode=args.explore_mode,
            num_rw_tries=args.num_rw_tries,
            debug=args.debug)
        
    elif args.algorithm == "random":

        algorithm = RandomEvaluator(
            args.base_dir,
            num_simulations=args.num_simulations,
            debug=args.debug)
    else:

        # Should never reach here.
        assert False

    run_algorithm(fh, algorithm, tasks,
                  fail_fast=args.fail_fast,
                  simulator_budget=args.max_steps)

    fh.write("Time Taken: %.2f\n" % (time.time() - start_time))
    fh.close()

    # Tireworld, 1-drift, seeds with success
    # 1701296892
    # 1701305081

    # Drift: 1701311940

    # Error with intelligent bfs
    # seed = 1701324116
