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
  --batch-size 64 \
  --epochs 2000 \
  --lr 5e-4 \
  --hidden-size 128 \
  --emb-dim 8 \
  --num-layers 2 \
  --dropout 0.1 \
  --teacher-start 0 \
  --teacher-end 0 \
  --patience 2000 \
  --seed 42 \
  --num-workers 10 \
  --time-encode sin-cos \
  --shuffle-split \
  --diffusion-timesteps 30 \
  --diffusion-t-embed-dim 16 \
  --diffusion-k-steps 2 \
  > "$LOGFILE" 2>&1 &
echo "Training started. Log: $LOGFILE"