#!/bin/bash
set -e
cd "$(dirname "$0")/../src"
python train.py --model diffusion
python train.py --model gan
python train.py --model vae
python train.py --model seq_vae
python train.py --model seq_diffusion
python train.py --model seq_ar_diffusion