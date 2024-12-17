#!/bin/bash

function run_sbatch() {

    cmd=$1
    until $cmd
    do
        echo "Command $cmd failed, ...retrying"
        sleep 2
    done

}

USER=`whoami`
SCRATCH_DIR=/scratch/$USER/results
START_RUN=0
END_RUN=9

MAX_STEPS=100000
SAMPLING_COUNT=100
NUM_SIMULATIONS=10

export PYTHONHASHSEED=0

for DOMAIN in "blocks" "first_responders" "elevators" "tireworld"
do
for (( run_no=$START_RUN; run_no<=$END_RUN; run_no++ ))
do
    BASE_DIR="$SCRATCH_DIR"/"$DOMAIN"/run"$run_no"/

    mkdir -p $BASE_DIR
    TASK_DIR="benchmarks"/"$DOMAIN"/"3-action-drift/"
    
    run_sbatch "sbatch scripts/icaps-24/diff-learn.sbatch $BASE_DIR $TASK_DIR qlearning $MAX_STEPS $SAMPLING_COUNT $NUM_SIMULATIONS $BASE_DIR/qlearning_logs.txt"
    run_sbatch "sbatch scripts/icaps-24/diff-learn.sbatch $BASE_DIR $TASK_DIR random $MAX_STEPS $SAMPLING_COUNT $NUM_SIMULATIONS $BASE_DIR/random_logs.txt"
    run_sbatch "sbatch scripts/icaps-24/diff-learn.sbatch $BASE_DIR $TASK_DIR oracle $MAX_STEPS $SAMPLING_COUNT $NUM_SIMULATIONS $BASE_DIR/oracle_logs.txt"
    run_sbatch "sbatch scripts/icaps-24/diff-learn.sbatch $BASE_DIR $TASK_DIR qace $MAX_STEPS $SAMPLING_COUNT $NUM_SIMULATIONS $BASE_DIR/qace_logs.txt"
    run_sbatch "sbatch scripts/icaps-24/diff-learn.sbatch $BASE_DIR $TASK_DIR qace-stateless $MAX_STEPS $SAMPLING_COUNT $NUM_SIMULATIONS $BASE_DIR/qace-stateless_logs.txt"
    run_sbatch "sbatch scripts/icaps-24/diff-learn.sbatch $BASE_DIR $TASK_DIR drift $MAX_STEPS $SAMPLING_COUNT $NUM_SIMULATIONS $BASE_DIR/drift_logs.txt"
done
done
