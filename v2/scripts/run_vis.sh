#!/usr/bin/env bash
set -e

PY=${PY:-python}
MODEL=${MODEL:-seq2seq}
LOGDIR="v2/logs"
mkdir -p "$LOGDIR"
NOW=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOGDIR/${MODEL}_vis_$NOW.log"

nohup $PY -m v2.src.visualize \
  --y-true "v2/results/${MODEL}_y_true.npy" \
  --y-pred-mean "v2/results/${MODEL}_y_pred_mean.npy" \
  --y-samples-first "v2/results/${MODEL}_y_samples_first.npy" \
  --out "v2/results/plots/${MODEL}_scenarios.png" \
  --index 0 \
  > "$LOGFILE" 2>&1 &
echo "Visualization started. Log: $LOGFILE"