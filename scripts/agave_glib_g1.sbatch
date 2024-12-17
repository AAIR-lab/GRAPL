#!/bin/bash

#SBATCH -c 2                       # number of cores (16 GiB memory)
#SBATCH -t 0-4:00                  # wall time (D-HH:MM)



PROJECT_DIR="/home/rkaria/stochastic-AIA"

BASE_DIR=$1
DOMAIN=$2
LOG_FILE=$3

install_dependencies() {

    # Upgrade pip first.
    pip3 install --upgrade pip

    pip3 install --upgrade pip
    pip3 install networkx
    pip3 install pydot
    pip3 install gym
    pip3 install pddlgym
    pip3 install termcolor
    pip3 install seaborn
    pip3 install argparse
    pip3 install tqdm
    pip3 install graphviz
}

# module purge

# Load the anaconda python3.
# This py3.7 package provdies python-dev needed to compile tensorflow.
# module load anaconda/py3
# source activate ipml_conda
# source activate python-3.7.2

# Create a virtual env for the generalized-heuristics project if it has not
# been already created.
#
# Allow errors since we don't care about this error.
# mkdir venv_ipml

# Create the virtual environment and activate it.
# python3 -m venv venv_ipml
# source venv_ipml/bin/activate

# python3 --version

# Install the dependencies
# install_dependencies

# Navigate to the project directory.
pushd /home/rkaria/stochastic-AIA


echo "Running" PYTHONHASHSEED=0 python3 src/main.py --base-dir $BASE_DIR --gym-domain-name $DOMAIN --curiosity GLIB_G1
export PYTHONHASHSEED=0 
mkdir -p $BASE_DIR
python3 src/main.py --base-dir $BASE_DIR --gym-domain-name $DOMAIN --glib --curiosity GLIB_G1 &> $LOG_FILE;

# Pop back to where we were.
popd
