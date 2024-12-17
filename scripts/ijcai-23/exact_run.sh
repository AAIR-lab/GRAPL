#!/bin/bash


START_RUN=0
END_RUN=9

export PYTHONHASHSEED=0

# for domain in "Tireworld" "Explodingblocks" "Probabilistic_elevators" "First_responders"
for domain in "Probabilistic_elevators"
do
for (( run_no=$START_RUN; run_no<=$END_RUN; run_no++ ))
do
    ./scripts/ijcai-23/server.sh results $domain $run_no $run_no 8h &
done
done
