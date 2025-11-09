#!/usr/bin/env bash
set -e

PY=${PY:-python}
MODELS=("seq2seq" "gan" "vae" "seq2seq_ar_diffusion")
LOGDIR="v2/logs"
mkdir -p "$LOGDIR"

echo "Starting visualizations for all models and turb_ids..."
echo "Models: ${MODELS[*]}"

for MODEL in "${MODELS[@]}"; do
    echo "Processing model: $MODEL"
    
    TURB_IDS_FILE="v2/results/${MODEL}_turb_ids.npy"
    if [ ! -f "$TURB_IDS_FILE" ]; then
        echo "  Warning: $TURB_IDS_FILE not found. Make sure prediction is completed for $MODEL."
        echo "  Skipping $MODEL..."
        continue
    fi
    
    # 直接生成 0~133 的风机ID列表
    TURB_ID_LIST=$(seq 0 133)
    TURB_COUNT=134
    echo "  Using hardcoded turbine IDs: 0~133"

    # 为每个 turb_id 启动可视化
    for TURB_ID in $TURB_ID_LIST; do
        echo "    Launching visualization for turb_id $TURB_ID"
        NOW=$(date +"%Y%m%d_%H%M%S")
        LOGFILE="$LOGDIR/${MODEL}_vis_${TURB_ID}_$NOW.log"

        nohup $PY -m v2.src.visualize \
          --y-true "v2/results/${MODEL}_y_true.npy" \
          --y-pred-mean "v2/results/${MODEL}_y_pred_mean.npy" \
          --y-samples-all "v2/results/${MODEL}_y_samples_all.npy" \
          --turb-ids "v2/results/${MODEL}_turb_ids.npy" \
          --turb-id "$TURB_ID" \
          --out "v2/results/plots/${MODEL}_${TURB_ID}_scenarios.png" \
          --index -1 \
          > "$LOGFILE" 2>&1 &
    done
done

echo "All visualization jobs launched. Check logs in $LOGDIR/"
echo "To check running processes: ps aux | grep 'visualize'"