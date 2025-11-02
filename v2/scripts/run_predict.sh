#!/usr/bin/env bash
set -e

PY=${PY:-python}
MODEL=${MODEL:-seq2seq}
SAMPLE_INDEX=${SAMPLE_INDEX:-0}
LOGDIR="v2/logs"
mkdir -p "$LOGDIR"
NOW=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOGDIR/${MODEL}_predict_$NOW.log"

nohup $PY -m v2.src.predict \
  --model "$MODEL" \
  --data v2/results/cleaned.csv \
  --hist-len 24 \
  --pred-len 24 \
  --checkpoint "v2/results/checkpoints/${MODEL}/best.pth" \
  --samples 100 \
  --out-prefix "v2/results/${MODEL}" \
  --sample-index "$SAMPLE_INDEX" \
  > "$LOGFILE" 2>&1 &
echo "Prediction started. Log: $LOGFILE (sample_index=$SAMPLE_INDEX)"