#!/bin/bash

#SBATCH -c 1                        # number of cores
#SBATCH -t 0-4:00                  # wall time (D-HH:MM)
#SBATCH -A rkaria             # Account hours will be pulled from (commented out with double # in front)
##SBATCH -o slurm_logs/slurm.%j.out             # STDOUT (%j = JobId)
##SBATCH -e slurm_logs/slurm.%j.err             # STDERR (%j = JobId)
##SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
##SBATCH --mail-user=Rushang.Karia@asu.edu # send-to address


7z a -aoa /scratch/rkaria/results.zip -ir\!/scratch/rkaria/*.csv 

