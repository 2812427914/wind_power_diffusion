#!/usr/bin/env bash
set -e

PY=${PY:-python}
MODEL=${MODEL:-seq2seq}
LOGDIR="v2/logs"
mkdir -p "$LOGDIR"
NOW=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOGDIR/${MODEL}_train_$NOW.log"

nohup $PY -m v2.src.train \
  --model "$MODEL" \
  --data data/wtbdata_hourly.csv \
  --hist-len 24 \
  --pred-len 24 \
  --stride 1 \
  --batch-size 256 \
  --epochs 2000 \
  --lr 1e-3 \
  --hidden-size 32 \
  --emb-dim 5 \
  --num-layers 2 \
  --dropout 0.1 \
  --teacher-start 0 \
  --teacher-end 0 \
  --patience 100 \
  --seed 42 \
  --num-workers 20 \
  --time-encode sin-cos \
  --shuffle-split \
  > "$LOGFILE" 2>&1 &
echo "Training started. Log: $LOGFILE"