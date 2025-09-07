#!/bin/bash
set -e
cd "$(dirname "$0")/../src"
python evaluate.py --model diffusion
python evaluate.py --model gan
python evaluate.py --model vae
python evaluate.py --model seq_vae
python evaluate.py --model seq_diffusion
python evaluate.py --model seq_ar_diffusion