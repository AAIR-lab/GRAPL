#!/bin/bash

function run_sbatch() {

    cmd=$1
    until $cmd
    do
        echo "Command $cmd failed, ...retrying"
        sleep 2
    done

}

SCRATCH_DIR=/scratch/rkaria/results
START_RUN=0
END_RUN=9

export PYTHONHASHSEED=0

for DOMAIN in "Tireworld" "Explodingblocks" "Probabilistic_elevators" "First_responders"
do
for (( run_no=$START_RUN; run_no<=$END_RUN; run_no++ ))
do    
    BASE_DIR="$SCRATCH_DIR"/run"$run_no"/
    LOG_DIR="$SCRATCH_DIR"/run"$run_no"/"$DOMAIN"
    mkdir -p $BASE_DIR
    mkdir -p $LOG_DIR

    run_sbatch "sbatch scripts/agave_vd.sbatch $BASE_DIR $DOMAIN $LOG_DIR/vd.log"
done
done
