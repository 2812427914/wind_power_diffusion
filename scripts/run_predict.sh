#!/bin/bash
set -e
cd "$(dirname "$0")/../src"
python predict.py --model diffusion --batch_size 512 --n_samples 10
python predict.py --model gan
python predict.py --model vae
python predict.py --model seq_vae --n_samples 100
python predict.py --model seq_diffusion --n_samples 100
python predict.py --model seq_ar_diffusion --n_samples 100