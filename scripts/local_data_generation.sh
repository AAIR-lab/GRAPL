#!/usr/bin/bash

BASE_DIR=$1

for domain in "Tireworld" "Tireworld_og" "Explodingblocks" "Explodingblocks_og" "Probabilistic_elevators" "First_responders"
do
PYTHONHASHSEED=0 python3 src/ipml.py --base-dir $BASE_DIR --gym-domain-name $domain --vd
done
