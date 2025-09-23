#!/bin/bash
set -e
cd "$(dirname "$0")/../src"
mkdir -p ../logs
LOGTIME=$(date +"%Y%m%d_%H%M%S")
nohup python visualize.py >> ../logs/vis_$LOGTIME.log 2>&1 &