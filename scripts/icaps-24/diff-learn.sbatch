#!/bin/bash

#SBATCH -c 1                       # number of cores (8 GiB memory)
#SBATCH -t 0-8:00                  # wall time (D-HH:MM)
#SBATCH -o slurm_logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm_logs/slurm.%j.err # file to save job's STDERR (%j = JobId)

BASE_DIR=$1
TASK_DIR=$2
ALGORITHM=$3
MAX_STEPS=$4
SAMPLING_COUNT=$5
NUM_SIMULATIONS=$6
LOG_FILE=$7

echo "Running" PYTHONHASHSEED=0 python3 src/main.py --debug --base-dir $BASE_DIR --task-dir $TASK_DIR --algorithm $ALGORITHM --max-steps $MAX_STEPS --sampling-count $SAMPLING_COUNT
export PYTHONHASHSEED=0 
mkdir -p $BASE_DIR
python3 src/main.py \
    --debug \
    --base-dir $BASE_DIR \
    --task-dir $TASK_DIR \
    --algorithm $ALGORITHM \
    --max-steps $MAX_STEPS \
    --sampling-count $SAMPLING_COUNT \
    --num-simulations $NUM_SIMULATIONS > $LOG_FILE 2>&1 ;

