#!/bin/bash
set -e
cd "$(dirname "$0")/../src"
python predict.py --model diffusion --batch_size 512 --n_samples 10
python predict.py --model gan
python predict.py --model vae