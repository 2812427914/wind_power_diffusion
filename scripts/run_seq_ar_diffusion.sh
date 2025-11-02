#!/bin/bash
set -e
cd "$(dirname "$0")/../src"
mkdir -p ../logs ../results ../checkpoints
LOGTIME=$(date +"%Y%m%d_%H%M%S")

echo "Stage 1: Training"
python train.py --model seq_ar_diffusion \
       --use_lazy_dataset \
       --epochs 50 \
       --batch_size 1024 \
       --device cuda 2>&1 | tee ../logs/train_seq_ar_diffusion_${LOGTIME}.log

echo "Stage 2: Predicting"
python predict.py --model seq_ar_diffusion \
       --use_lazy_dataset \
       --n_samples 200 \
       --batch_size 256 \
       --device cuda 2>&1 | tee ../logs/predict_seq_ar_diffusion_${LOGTIME}.log

echo "Stage 3: Evaluating"
python evaluate.py --model seq_ar_diffusion 2>&1 | tee ../logs/eval_seq_ar_diffusion_${LOGTIME}.log