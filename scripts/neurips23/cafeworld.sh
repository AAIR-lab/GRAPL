#!/bin/bash


START_RUN=0
END_RUN=9

export PYTHONHASHSEED=0

BASE_DIR=./results

rm -fr $BASE_DIR

# for domain in "Tireworld" "Explodingblocks" "Probabilistic_elevators" "First_responders"
for domain in "Cafeworld"
do
for (( run_no=$START_RUN; run_no<=$END_RUN; run_no++ ))
do

    mkdir -p $BASE_DIR/"run"$run_no
    python3 src/main.py --base-dir $BASE_DIR/"run"$run_no --gym-domain-name $domain --vd &> $BASE_DIR/"run"$run_no/vd.log
    python3 src/main.py --base-dir $BASE_DIR/"run"$run_no --gym-domain-name $domain --ipml --count-sdm-samples --randomize-pal &> $BASE_DIR/"run"$run_no/ipml.log
done
done
