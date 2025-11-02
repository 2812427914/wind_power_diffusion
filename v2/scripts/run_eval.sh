#!/usr/bin/env bash
set -e

PY=${PY:-python}
MODEL=${MODEL:-seq2seq}
LOGDIR="v2/logs"
mkdir -p "$LOGDIR"
NOW=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOGDIR/${MODEL}_eval_$NOW.log"

nohup $PY -m v2.src.evaluate \
  --model "$MODEL" \
  --data v2/results/cleaned.csv \
  --hist-len 24 \
  --pred-len 24 \
  --stride 1 \
  --checkpoint "v2/results/checkpoints/${MODEL}/best.pth" \
  --splits-path "v2/results/splits_${MODEL}.json" \
  --out "v2/results/metrics_${MODEL}.json" \
  > "$LOGFILE" 2>&1 &
echo "Evaluation started. Log: $LOGFILE"