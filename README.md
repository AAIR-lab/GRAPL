# GHN
Python implementation of Generalized Heuristic Networks

This repo is being updated with new changes (You can refer to the aaai-21 branch for the exact code used for the AAAI-21
submission). 

# Installation

Use the following commands to install the GHN system on an Ubuntu 18.04 machine.

    sudo apt install graphviz graphviz-dev python3-pip cmake

    pip3 install --upgrade pip
    pip3 install networkx
    pip3 install tensorflow==2.3.1
    pip3 install pygraphviz
    pip3 install dulwich
    pip3 install tqdm
    pip3 install matplotlib
    pip3 install pydot
    pip3 install pyyaml==5.3.1
    pip3 install tinydb
    pip3 install pandas
    pip3 install bokeh
    pip3 install psutil

Next from the project root directory,

    cd bin/fast-downward
    ./build.py release

# Reproducing the AAAI-21 experiments

The GHN system uses YAML config files for supported domains to run the experiments.
The AAAI-21 YAML files can be found in the experiments/ directory.

Each YAML file consists of several phases which

 1. Generates problem files for training and testing
 2. Runs baseline solvers FF and FD (and Pyperplan) to solve the test problems.
 3. Runs the leapfrogging steps for 3 iterations to train GHN leapfrog networks.
 4. Runs all GHN trained networks on the test data.

These experiments take an average of 12 hours to complete for all phases in a multicore setting.

Use the following command line to run the experiments from the root directory.

python3 generalized_learning.py --base-dir `<directory_where_to_store_results>` --config-file `<path_to_yaml_file>`

#### Example:

To run blocksworld experiments, run the following

`python3 generalized_learning.py --base-dir ./results --config-file experiments/aaai21/leapfrogs/blocksworld.yaml`

# Fast training

If you do not wish to run the complete AAAI-21 suite, then the bare minimum required is the `oracle` phases that are present in the YAML files.

The `oracle` solver uses an external solver (FF) to solve the training problems, the `oracle` model then trains a GHN using the solved problems and finally, the `pyperplan oracle solver` uses pyperplan and the learned GHN to solve the test problems. This pipeline takes under 20 mins to complete training in most cases.

The setup involves a training_directory where the training problem files are kept and a test_directory to store the test files. Note that the domain file and problem file must end with extensions `*.domain.pddl` and `*.problem.pddl`.

You can use the `oracle` model and solver from the yaml config files to solve the training problems using FF (without leapfrogging) and train the GHN and then use it to solve the problems in the test directory.

For examples of such configuration files, please take a look at the `example.yaml` files found in `generalized_learning/benchmarks/<domain>`.
The only change needed in those files is the solver for the test problems with pyperplan instead of our own internal A* implementation. The pyperplan config can be found in the YAML files for the AAAI-21 experiments.

# Using your own data/new domains
Follow the setup in the Fast training section above to train and use GHNs with your own data.

As an additional note, once you have trained a model, you can run pyperplan individually using the sourced version of pyperplan provided in the dependencies/ directory.

The command line arguments to enable the GHN heuristic are `python3 dependencies/pyperplan/src/pyperplan.py -s <search_algorithm> -H nnplact --model-dir <path_to_model_dir> --model-name <model_name_as_in_yaml> <domain_file> <problem_file>`

