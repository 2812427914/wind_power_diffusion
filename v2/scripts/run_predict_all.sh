#!/usr/bin/env bash
set -e

# 定义要运行的模型列表
MODELS=("seq2seq" "gan" "vae" "seq2seq_ar_diffusion")
SAMPLE_INDEX=${SAMPLE_INDEX:-0}

echo "Starting predictions for models: ${MODELS[*]}"
echo "Sample index: $SAMPLE_INDEX"

for MODEL in "${MODELS[@]}"; do
    echo "Launching prediction for $MODEL..."
    # 传递参数给子脚本
    MODEL=$MODEL SAMPLE_INDEX=$SAMPLE_INDEX bash v2/scripts/run_predict.sh
done

echo "All prediction jobs launched. Monitor logs in v2/logs/"
echo "To check running processes: ps aux | grep 'predict'"