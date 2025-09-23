#!/bin/bash
set -e
cd "$(dirname "$0")/../src"
mkdir -p ../logs
LOGTIME=$(date +"%Y%m%d_%H%M%S")
nohup python evaluate.py --model diffusion      >> ../logs/eval_diffusion_$LOGTIME.log 2>&1 &
nohup python evaluate.py --model gan           >> ../logs/eval_gan_$LOGTIME.log 2>&1 &
nohup python evaluate.py --model vae           >> ../logs/eval_vae_$LOGTIME.log 2>&1 &
nohup python evaluate.py --model seq_vae       >> ../logs/eval_seq_vae_$LOGTIME.log 2>&1 &
nohup python evaluate.py --model seq_diffusion >> ../logs/eval_seq_diffusion_$LOGTIME.log 2>&1 &
nohup python evaluate.py --model seq_ar_diffusion >> ../logs/eval_seq_ar_diffusion_$LOGTIME.log 2>&1 &