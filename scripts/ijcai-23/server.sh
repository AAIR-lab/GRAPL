#!/bin/bash

BASE_DIR=$1
DOMAIN=$2
START_RUN=$3
END_RUN=$4
TIMEOUT=$5

export PYTHONHASHSEED=0

for (( run_no=$START_RUN; run_no<=$END_RUN; run_no++ ))
do
#    for domain in "Tireworld" "Tireworld_og" "Explodingblocks" "Explodingblocks_og" "Probabilistic_elevators" "First_responders"

LOG_DIR="$BASE_DIR"/run"$run_no"/"$DOMAIN"
mkdir -p $LOG_DIR

LOG_FILE="$LOG_DIR"/vd.log
timeout $TIMEOUT python3 src/ipml.py --base-dir $BASE_DIR/run"$run_no" --gym-domain-name $DOMAIN --vd &> $LOG_FILE

LOG_FILE="$LOG_DIR"/glib_glib_g1.log
timeout $TIMEOUT python3 src/ipml.py --base-dir $BASE_DIR/run"$run_no" --gym-domain-name $DOMAIN --glib --curiosity GLIB_G1 &> $LOG_FILE &

LOG_FILE="$LOG_DIR"/glib_glib_l2.log
timeout $TIMEOUT python3 src/ipml.py --base-dir $BASE_DIR/run"$run_no" --gym-domain-name $DOMAIN --glib --curiosity GLIB_L2 &> $LOG_FILE &

LOG_FILE="$LOG_DIR"/ipml_randomized.log
timeout $TIMEOUT python3 src/ipml.py --base-dir $BASE_DIR/run"$run_no" --gym-domain-name $DOMAIN --ipml --randomize-pal &> $LOG_FILE &

LOG_FILE="$LOG_DIR"/ipml_sequential.log
timeout $TIMEOUT python3 src/ipml.py --base-dir $BASE_DIR/run"$run_no" --gym-domain-name $DOMAIN --ipml  &> $LOG_FILE &

done
