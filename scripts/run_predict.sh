#!/bin/bash
set -e
cd "$(dirname "$0")/../src"
mkdir -p ../logs
LOGTIME=$(date +"%Y%m%d_%H%M%S")
nohup python predict.py --model diffusion --batch_size 512 --n_samples 10      >> ../logs/predict_diffusion_$LOGTIME.log 2>&1 &
nohup python predict.py --model gan                                            >> ../logs/predict_gan_$LOGTIME.log 2>&1 &
nohup python predict.py --model vae                                            >> ../logs/predict_vae_$LOGTIME.log 2>&1 &
nohup python predict.py --model seq_vae --n_samples 100                        >> ../logs/predict_seq_vae_$LOGTIME.log 2>&1 &
nohup python predict.py --model seq_diffusion --n_samples 100                  >> ../logs/predict_seq_diffusion_$LOGTIME.log 2>&1 &
nohup python predict.py --model seq_ar_diffusion --n_samples 100               >> ../logs/predict_seq_ar_diffusion_$LOGTIME.log 2>&1 &