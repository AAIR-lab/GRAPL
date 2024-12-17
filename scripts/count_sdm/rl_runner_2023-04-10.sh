#!/bin/bash

# Script to run experiments on the server machines *EXACT*

RESULTS_DIR=./results

rm -fr $RESULTS_DIR

source env/bin/activate

export PYTHONHASHSEED=0
export PYTHONPATH=.


for domain in "Tireworld" "Explodingblocks" "First_responders" "Probabilistic_elevators"
do
    ./scripts/count_sdm/rl_2023-04-10.sh $domain &
    ./scripts/count_sdm/ipml_rl_2023-04-10.sh $domain &
done
