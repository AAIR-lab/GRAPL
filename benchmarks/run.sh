#!/bin/bash

export PYTHONHASHSEED=0

BASE_DIR=/tmp/qace
DOMAIN=$1

python3 src/ipml.py --base-dir $BASE_DIR/$DOMAIN --domain-file benchmarks/$DOMAIN/domain.pddl --problem-file benchmarks/$DOMAIN --vd
python3 src/ipml.py --base-dir $BASE_DIR/$DOMAIN --domain-file benchmarks/$DOMAIN/domain.pddl --problem-file benchmarks/$DOMAIN/training_problem.pddl --randomize-pal --count-sdm-samples --ipml
