#!/bin/bash

# Script to run experiments on the server machines *EXACT*

RESULTS_DIR=./results

rm -fr $RESULTS_DIR

source env/bin/activate

export PYTHONHASHSEED=0
export PYTHONPATH=.

for domain in "Tireworld" "Explodingblocks" "First_responders" "Probabilistic_elevators"
do
    python3 src/main.py --base-dir $RESULTS_DIR --gym-domain-name $domain --vd
done

for domain in "Tireworld" "Explodingblocks" "First_responders" "Probabilistic_elevators"
do
    python3 src/main.py --base-dir $RESULTS_DIR --gym-domain-name $domain --ipml &
    python3 src/main.py --base-dir $RESULTS_DIR --gym-domain-name $domain --ipml --count-sdm-samples &
    python3 src/main.py --base-dir $RESULTS_DIR --gym-domain-name $domain --randomize-pal --ipml &
    python3 src/main.py --base-dir $RESULTS_DIR --gym-domain-name $domain --randomize-pal --ipml --count-sdm-samples &
done
