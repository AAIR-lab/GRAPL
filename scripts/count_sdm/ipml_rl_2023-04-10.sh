#!/bin/bash

DOMAIN=$1


START_RUN=0
END_RUN=10

START_PROBLEM=0
END_PROBLEM=4

source env/bin/activate

export PYTHONHASHSEED=0
export PYTHONPATH=.

for (( p=START_PROBLEM;p<END_PROBLEM;p++ ))
do
    for (( run=START_RUN;run<END_RUN;run++ ))
    do
    python3 src/ipml_rl.py --base-dir ./results/run"$run"/p"$p" --gym-domain-name $DOMAIN --rl --experiment-name ipml_rl --problem-idx $p --ipml
    done
done
