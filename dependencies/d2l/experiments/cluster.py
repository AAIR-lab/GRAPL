#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time

import yaml

from run import do

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str,  required=True, help="Experiment configuration")
    parser.add_argument('--task', type=int, default=None, help="Task ID, if in run mode")
    return parser


def main(parser, args):
    args = parser.parse_args(args)
    with open("{}.yml".format(args.exp), 'r') as f:
        data = yaml.load(f, Loader=yaml.BaseLoader)
        run = data["run"]
        experiments = data["experiments"]

    if not experiments:
        raise RuntimeError("No experiments found in the configuration file")

    if args.task is None:
        generate_script(timeout=run["time"], mem=run["mem"], num_tasks=len(experiments), experiment_set=args.exp)
    else:
        # Simply run the whole thing!
        if args.task - 1 > len(experiments):
            raise RuntimeError("Task ID #{} not defined on experiment set {}.".format(args.task, args.exp))

        expid = experiments[args.task-1]  # because slurm task IDs range from 1 to n, not from 0.
        do(expid)


def generate_script(num_tasks, timeout, mem, experiment_set):
    tpl = """#!/usr/bin/env bash

### Set name.
#SBATCH --job-name=sltp
### Redirect stdout and stderr.
#SBATCH --output=slurm.log
#SBATCH --error=slurm.err
### Let later steps append their logs to the output and error files.
#SBATCH --open-mode=append
### Set partition.
#SBATCH --partition=infai_2
### Set quality-of-service group.
#SBATCH --qos=normal
### Set time limit (in min).
#SBATCH --time={time}
### Set memory limit.
#SBATCH --mem-per-cpu={mem}
### Number of tasks.
#SBATCH --array=1-{num_tasks}
### Adjustment to priority ([-2147483645, 2147483645]).
#SBATCH --nice=2000
### Send mail? Mail type can be e.g. NONE, END, FAIL, ARRAY_TASKS.
#SBATCH --mail-type=END,FAIL,REQUEUE,STAGE_OUT
#SBATCH --mail-user=guillem.frances@unibas.ch


module purge
LMOD_DISABLE_SAME_NAME_AUTOSWAP=no module load Python/3.5.2-goolf-1.7.20
LMOD_DISABLE_SAME_NAME_AUTOSWAP=no module load GCC/5.4.0-2.26
source ${{HOME}}/lib/virtualenvs/concepts/bin/activate

export LIBRARY_PATH=$LIBRARY_PATH:{libpath}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{libpath}
export FS_PATH={fspath}

EXPDIR={exp_dir}
mkdir -p ${{EXPDIR}}
./cluster.py --exp {experiment_set} --task ${{SLURM_ARRAY_TASK_ID}}  > {output}.log 2>{output}.err
"""

    filename = "{}.sh".format(experiment_set)
    libpath = os.path.expanduser("~/local/lib")
    exp_dir = os.path.join(CURRENT_DIR, "{}_{}".format(experiment_set, time.strftime("%y%m%d")))
    with open(filename, "w") as f:
        output = "${EXPDIR}/out_${SLURM_ARRAY_TASK_ID}"
        f.write(tpl.format(time=timeout, mem=mem, num_tasks=num_tasks, experiment_set=experiment_set, libpath=libpath,
                           exp_dir=exp_dir, output=output, fspath=os.environ.get("FS_PATH", "")))
    print("Written cluster script {}".format(filename))


if __name__ == "__main__":
    main(create_parser(), sys.argv[1:])
