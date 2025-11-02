#!/bin/bash
set -e
cd "$(dirname "$0")/../src"
mkdir -p ../logs
LOGTIME=$(date +"%Y%m%d_%H%M%S")
nohup python train.py --model diffusion      >> ../logs/train_diffusion_$LOGTIME.log 2>&1 &
nohup python train.py --model gan           >> ../logs/train_gan_$LOGTIME.log 2>&1 &
nohup python train.py --model vae           >> ../logs/train_vae_$LOGTIME.log 2>&1 &
nohup python train.py --model seq_vae       >> ../logs/train_seq_vae_$LOGTIME.log 2>&1 &
nohup python train.py --model seq_diffusion >> ../logs/train_seq_diffusion_$LOGTIME.log 2>&1 &
nohup python train.py --model seq_ar_diffusion >> ../logs/train_seq_ar_diffusion_$LOGTIME.log 2>&1 &