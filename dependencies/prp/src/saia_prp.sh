#! /bin/bash

# A version of validate-prp for use from Stochastic-AIA.
# This script will always be called from the Stochastic-AIA source code.

PRP_ROOT_DIR=$1
DOMAIN_FILE=$2
PROBLEM_FILE=$3

"$PRP_ROOT_DIR"/src/prp $DOMAIN_FILE $PROBLEM_FILE --dump-policy 2 --optimize-final-policy 1

python3 "$PRP_ROOT_DIR"/prp-scripts/translate_policy.py > human_policy.out

python3 "$PRP_ROOT_DIR"/prp-scripts/validator.py $DOMAIN_FILE $PROBLEM_FILE human_policy.out prp

# dot -Tpng graph.dot > graph.png
# dot -Tpng graph_state_based.dot > graph_state_based.png

