# Differential Assessment

# Basic installation

## Requirements
* Ubuntu 18.04 or greater
* Python3 (>= 3.2)

> **Note**
> For ubuntu 20.04 and greater users, python is not mapped to python3.
> Run `sudo apt install python-is-python3` to link python to python3.

## Installation


Run the followin command in a terminal
```
sudo apt install make g++ python3-venv graphviz gcc-multilib g++-multilib graphviz-dev python3-dev
```



Setup a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

Use the following commands to install the required python libraries.

```
pip3 install --upgrade pip
pip3 install networkx
pip3 install pydot
pip3 install gym
pip3 install pddlgym
pip3 install pygraphviz
pip3 install tqdm
pip3 install graphviz
pip3 install scipy
```
## PRP
```
pushd dependencies/prp/src
./build_all
popd
```


## GLib baseline

```
pip3 install termcolor
pip3 install pygraphviz
pip3 install seaborn
```

## Cafeworld TMP sim
```
sudo apt install docker.io
pip3 install docker
pip3 install urllib3==1.26.0
```

## Remove pddlgym. We use our own internal one.
```
pip3 uninstall pddlgym
```

## Plotting

> For LaTeX plotting in matplotlib
sudo apt install cm-super 
sudo apt install dvipng

# Running the Code

`src/main.py` contains the main() routine to execute the code.

### Example
All tasks run in the ICAPS-24 submission are present in the `benchmarks` directory. To run the experiments, you simply
have to specify the `--task-dir` parameter along with the `--algorithm` and results directory `--base-dir`.

```
PYTHONHASHSEED=0 python3 src/main.py --base-dir /tmp/results --algorithm drift --task-dir benchmarks/tireworld/1-action-drift
```
The above command will run the tireworld results from the paper using our method (CLaP) and store the results in /tmp/results.
This is the fastest experiment to run and evaluate and depending on your configuration this should not take more than 1 hour
in total.

`qace` represents the U+C Learner while `qace-stateless` represents the A+C Learner from the paper. These methods can be found
in the `src/evaluators/` directory with their corresponding name.

## Task format
To automatically run your own pddl files in this setting, you need to create a directory and place all domain files and task files
for each task in it. The file format is `domain-t<no>.pddl` and `problem-t<no>.pddl` where `<no>` represents the task number.

For example, to create 5 blocksworld tasks (lets assume the domain does not change), create a directory called `blocksworld`
and place all problem files in this directory. You also need 5 copies of the domain file (named using the convention described above).
