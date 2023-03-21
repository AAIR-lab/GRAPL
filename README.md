System Requirements
====================
```
Ubuntu 18.04.5
Python 3.6.9
```

Installation
=============

Use the following commands to install our software on your box.
Its best to create a virtual environment to not pollute your environemnt.

```
sudo apt install cmake git python3-venv python3-pip openjdk-11-jdk

python3 -m venv grl_env
source grl_env/bin/activate

# Upgrade pip first.
pip3 install pip --upgrade

# Install all the packages.
pip3 install tensorflow==2.5.0
pip3 install networkx
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
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cpu.html
pip3 install bitarray
pip3 install natsort==7.1.1
pip3 install antlr4-python3-runtime==4.7.2
pip3 install multipledispatch==0.6.0
pip3 install JPype1
```

Running our experiments
========================

All the problem files and configuration are included in the experiments/ directory.

Run this command first to setup the dataset.
```
cp -fr experiments/ /tmp/results
```
The directory structure is as follows:
```
domain
    - l<run_no>
        - l<leapfrog_iteration_no>
        - t<competition_instance_no>
```
Most experiments should take less than 1 hour to complete.

Running IJCAI-22 GRL training experiments
------------------------------------------
```
PYTHONHASHSEED=0 python3 generalized_learning.py --base-dir /tmp/results \
    --config-file experiments/<domain>/<domain>_l3_run<run_no>_td_dl_full.yaml
```
where
```
domain = ["academic_advising", "game_of_life_2", "sysadmin", "wildfire_2"]
run_no = [0..9]
```
For example, to run sysadmin training for run 0, run
```
PYTHONHASHSEED=0 python3 generalized_learning.py --base-dir /tmp/results \
    --config-file experiments/sysadmin/sysadmin_l3_run0_td_dl_full.yaml
```

Running IJCAI-22 GRL zero-shot transfer experiments
----------------------------------------------------
```
PYTHONHASHSEED=0 python3 generalized_learning.py --base-dir /tmp/results \
    --config-file experiments/<domain>/<domain>_t<instance_no>_l3_run<run_no>_td_dl_full.yaml
```
where
```
domain = ["academic_advising", "game_of_life_2", "sysadmin", "wildfire_2"]
instance_no = [5..10]
run_no = [0..9]
```
See training section for an example command.

Generate Description Logic Features
------------------------------------
To generate Description Logic features for a given RDDL/PPDDL/PDDL problem follow the followin steps
* Create a directory somewhere (`mkdir /tmp/dl_features/`)
* Copy the domain file and a single problem file to this directory.

> **Note** 
For RDDL, the domain name must match the rddl file name. i.e. if the domain name is `abc_mdp`, the rddl file should be `abc_mdp.rddl`). PDDL files have no such restriction.
* Run the following command `python3 scripts/d2l_features.py --domain-file <path_to_domain_file> --problem-file <path_to_problem_file>`
* The features should be generated in `<directory>/features.io`

You can also run `python3 scripts/d2l_features.py -h` for some more options such as changing the feature complexity etc.

# Contributors
[Rushang Karia](https://rushangkaria.github.io) <br>
[Siddharth Srivastava](https://siddharthsrivastava.net)
