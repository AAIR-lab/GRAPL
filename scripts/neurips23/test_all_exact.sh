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
    python3 src/main.py --base-dir $RESULTS_DIR --gym-domain-name $domain --ipml --count-sdm-samples &> "$RESULTS_DIR"/"$domain"_ipml_s_c.log &
    python3 src/main.py --base-dir $RESULTS_DIR --gym-domain-name $domain --randomize-pal --ipml --count-sdm-samples &> "$RESULTS_DIR"/"$domain"_ipml_r_c.log &
    python3 src/main.py --base-dir $RESULTS_DIR --gym-domain-name $domain --glib --curiosity GLIB_L2 &> "$RESULTS_DIR"/"$domain"_glib_l2.log & 
    python3 src/main.py --base-dir $RESULTS_DIR --gym-domain-name $domain --glib --curiosity GLIB_G1 &> "$RESULTS_DIR"/"$domain"_glib_g1.log &
done
