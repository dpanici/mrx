#!/bin/bash
# MRX fixed-boundary relaxation of one equilibrium file, then the Poincare sections of its initial field (the file's
# own nested surfaces, B = dA') and of the relaxed field.
#   bash 08_mrx_run.sh FILE TAG "NR NT NZ" STEPS
FILE=$1; TAG=$2; RES=${3:-"12 16 16"}; STEPS=${4:-20}
OUT=mrx_$TAG
python -u /home/user/mrx/scripts/relax.py --geometry.path $FILE --geometry.resolution $RES \
    --geometry.max-batch ${MAXBATCH:-262144} --budget.steps $STEPS --budget.chunk 5 --budget.floor-tol 0 --output.out $OUT > $OUT.log 2>&1 || exit 1
python -u /home/user/mrx/scripts/poincare_trace.py --geometry $FILE $OUT/checkpoints/state_000000.h5 \
    $OUT/checkpoints/best.h5 --planes 0 0.25 0.5 --lines 60 --periods 300 >> $OUT.log 2>&1
