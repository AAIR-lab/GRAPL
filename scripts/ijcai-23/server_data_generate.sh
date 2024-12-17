#!/usr/bin/bash

BASE_DIR=$1
START_RUN=$2
END_RUN=$3

for (( run_no=$START_RUN; run_no<=$END_RUN; run_no++ ))
do
    for domain in "Tireworld" "Tireworld_og" "Explodingblocks" "Explodingblocks_og" "Probabilistic_elevators" "First_responders"
    do
    PYTHONHASHSEED=0 python3 src/ipml.py --base-dir $BASE_DIR/run"$run_no" --gym-domain-name $domain --vd
    done
done
