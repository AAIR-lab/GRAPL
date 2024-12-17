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
    
    run_sbatch "sbatch scripts/agave_glib_g1.sbatch $BASE_DIR $DOMAIN $LOG_DIR/glib_g1_lndr.log"
    run_sbatch "sbatch scripts/agave_glib_l2.sbatch $BASE_DIR $DOMAIN $LOG_DIR/glib_l2_lndr.log"
    run_sbatch "sbatch scripts/agave_ipml_r.sbatch $BASE_DIR $DOMAIN $LOG_DIR/ipml_randomized.log"
done
done
