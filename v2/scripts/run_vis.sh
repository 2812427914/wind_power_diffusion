#!/usr/bin/env bash
set -e

PY=${PY:-python}
MODEL=${MODEL:-seq2seq}
TURB_ID=${TURB_ID:-90}
LOGDIR="v2/logs"
mkdir -p "$LOGDIR"
NOW=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOGDIR/${MODEL}_vis_$NOW.log"

nohup $PY -m v2.src.visualize \
  --y-true "v2/results/${MODEL}_y_true.npy" \
  --y-pred-mean "v2/results/${MODEL}_y_pred_mean.npy" \
  --y-samples-all "v2/results/${MODEL}_y_samples_all.npy" \
  --turb-ids "v2/results/${MODEL}_turb_ids.npy" \
  --turb-id "${TURB_ID}" \
  --out "v2/results/plots/${MODEL}_${TURB_ID}_scenarios.png" \
  --index -1 \
  > "$LOGFILE" 2>&1 &
echo "Visualization started. Log: $LOGFILE"