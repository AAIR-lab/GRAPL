#!/bin/bash

#SBATCH -c 1                       # number of cores (8 GiB memory)
#SBATCH -t 0-4:00                  # wall time (D-HH:MM)
#SBATCH -o slurm_logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm_logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
USER=`whoami`

7z a /scratch/$USER/results.zip -r  \
    /scratch/$USER/results/*.csv \
    /scratch/$USER/results/*logs.txt \
    /scratch/$USER/results/*run_info.txt

