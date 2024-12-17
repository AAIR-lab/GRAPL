#!/bin/bash

#SBATCH -c 1                       # number of cores (16 GiB memory)
#SBATCH -t 0-4:00                  # wall time (D-HH:MM)
##SBATCH -A rkaria             # Account hours will be pulled from (commented out with double # in front)
##SBATCH -o ../slurm_logs/slurm.%j.out             # STDOUT (%j = JobId)
##SBATCH -e ../slurm_logs/slurm.%j.err             # STDERR (%j = JobId)
##SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
##SBATCH --mail-user=Rushang.Karia@asu.edu # send-to address


# SOL use instructions
# module purge
# module load mamba/latest
# source activate ipml_conda
# source ipml_env/bin/activate

# SOL setup instructions
# module purge
# module load mamba/latest
# mamba create -n ipml_conda -c conda-forge
# source activate ipml_conda

# pip3 install pip --upgrade
# pip3 install --upgrade pip
# pip3 install networkx
# pip3 install pydot
# pip3 install gym
# pip3 install pddlgym
# pip3 install termcolor
# pip3 install seaborn
# pip3 install argparse
# pip3 install tqdm
# pip3 install graphviz
# pip3 uninstall pddlgym
# mamba install pygraphviz
# pip3 install termcolor
# pip3 install seaborn
# pip3 install docker
# pip3 install urllib3==1.26.0


rm -fr /scratch/rkaria/*
