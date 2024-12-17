'''
Created on Jan 17, 2023

@author: rkaria
'''
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import config
import argparse
import os
import sys

from utils.file_utils import FileUtils
from vd import VariationalDistance
from evaluators.result_logger import ResultLogger
from evaluators.glib_evaluator import GLIBEvaluator
from evaluators.saia_evaluator import SAIAEvaluator

import random
import time

def check_args_for_vd(args):
    pass


def check_args_for_glib(args):
    if args.glib:
        assert args.curiosity_name is not None


def check_args_for_ipml(args):
    pass


def check_and_get_actual_base_dir(args):

    base_dir = os.path.abspath(args.base_dir)
    assert os.path.isdir(base_dir) or not os.path.exists(base_dir)

    vd_filepath = "%s/vd_transitions.pkl" % (base_dir)

    if args.vd:

        return base_dir, None, vd_filepath
    elif args.experiment_name is not None:

        experiment_name = args.experiment_name
    elif args.glib:

        experiment_name = "%s_%s" % (args.curiosity_name.lower(),
                                          args.learning_name.lower())
    elif args.ipml:

        experiment_name = "ipml"
        if args.randomize_pal:
            experiment_name += "_randomized"
        else:
            experiment_name += "_sequential"

        if args.count_sdm_samples:
            experiment_name += "_sdm_counted"
    else:

        assert False

    data_dir = "%s/%s" % (base_dir, experiment_name)
    results_filename = "%s.csv" % (experiment_name)

    result_logger = ResultLogger(base_dir, filename=results_filename,
                                 clean_file=False)

    return data_dir, result_logger, vd_filepath


def setup(args):
    assert os.path.isdir(args.base_dir) or not os.path.exists(args.base_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="IPML",
                                     description="IPML code (IJCAI-23)")

    parser.add_argument("--clean", default=False, action="store_true",
                        help="Clean the experiment dir before running")

    parser.add_argument("--base-dir", required=True, type=str,
                        help="The base directory to store the results in")

    parser.add_argument("--domain-file", required=True, type=str,
                        help="The gym domain name to use")

    parser.add_argument("--seed", default=None, type=int,
                        help="The seed to use")

    parser.add_argument("--glib", default=False, action="store_true",
                        help="Run GLIB")

    parser.add_argument("--ipml", default=False, action="store_true",
                        help="Run IPML")

    parser.add_argument("--problem-file", required=True, type=str,
                        help="The problem file (or directory) to use")

    parser.add_argument("--sampling-count", default=5, type=int,
                        help="The sampling count to use for IPML.")

    parser.add_argument("--vd", default=False, action="store_true",
                        help="Run Variational Distance")

    parser.add_argument("--max-samples", default=3500, type=int,
                        help="The max samples to generate")

    parser.add_argument("--randomize-pal", default=False,
                        action="store_true",
                        help="Randomize the PAL tuples")

    parser.add_argument("--disable-evaluator", default=False,
                        action="store_true",
                        help="Disable the evaluators from running")

    parser.add_argument("--curiosity-name", type=str, default=None,
                        choices=["GLIB_G1", "GLIB_L2"],
                        help="The name of the curiosity routine for glib")
    parser.add_argument("--learning-name", default="LNDR", type=str,
                        choices=["LNDR"],
                        help="The name of the learning routine for glib")

    parser.add_argument("--experiment-name", default=None, type=str,
                        help="The experiment name")

    parser.add_argument("--count-sdm-samples", default=False,
                        action="store_true",
                        help="Set this flag to count SDM samples in IPML")

    parser.add_argument("--dry-run", default=False,
                        action="store_true",
                        help="Only dry run and initialize directories")

    parser.add_argument("--explore-mode", default="random_walk",
                        choices=["all_bfs", "random_walk"],
                        help="The explore mode to use.")

    args = parser.parse_args()

    # if args.seed is None:
    #     args.seed = time.time()

    # Seed the randomizer
    # random.seed(args.seed)

    if "PYTHONHASHSEED" not in os.environ:
        print("PYTHONHASHSEED=0 must be set while running IPML.")
        parser.print_usage()
        sys.exit(1)

    # print("Using seed:", args.seed)
    print("Using PYTHONHASHSEED:", os.environ["PYTHONHASHSEED"])

    if sum([args.vd, args.ipml, args.glib]) != 1:
        print("Exactly  one of --vd, --ipml, --glib must be passed.")
        parser.print_usage()
        sys.exit(1)

    check_args_for_vd(args)
    check_args_for_glib(args)
    check_args_for_ipml(args)

    data_dir, result_logger, vd_filepath = check_and_get_actual_base_dir(args)
    FileUtils.initialize_directory(data_dir, clean=args.clean)

    if args.vd and not args.dry_run:

        VariationalDistance.generate_bfs_samples(args.domain_file,
                                                 args.problem_file,
                                                 output_filepath=vd_filepath,
                                                 max_samples=args.max_samples)
    elif args.glib:

        vd_transitions = VariationalDistance.load_transitions(vd_filepath)
        glib_runner = GLIBEvaluator(args.gym_domain_name, args.seed,
                                    args.curiosity_name, args.learning_name,
                                    result_logger, data_dir, vd_transitions,
                                    args.disable_evaluator)
        if not args.dry_run:
            glib_runner.run(args)
    elif args.ipml:

        vd_transitions = VariationalDistance.load_transitions(vd_filepath)
        ipml_runner = SAIAEvaluator(args.domain_file,
                                    args.problem_file,
                                    result_logger,
                                    vd_transitions, args.disable_evaluator)

        if not args.dry_run:
            ipml_runner.run(args, data_dir=data_dir)