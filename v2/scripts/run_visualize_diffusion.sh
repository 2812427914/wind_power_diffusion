#!/bin/bash
set -e

MODEL="seq2seq_ar_diffusion"
CHECKPOINT="v2/results/checkpoints/${MODEL}/best.pth"
DATA="v2/results/cleaned.csv"
OUT="v2/results/plots/${MODEL}_diffusion_process.png"

echo "Visualizing diffusion process for ${MODEL}..."
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT}"

python -m v2.src.visualize_diffusion_process \
  --checkpoint "${CHECKPOINT}" \
  --data "${DATA}" \
  --out "${OUT}" \
  --timesteps 10 \
  --turb-id 90 \
  --seed 42

echo "Done!"