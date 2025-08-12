#!/bin/bash
set -e
cd "$(dirname "$0")/../src"
python evaluate.py --model diffusion
python evaluate.py --model gan
python evaluate.py --model vae