#!/usr/bin/env bash
set -e

PY=${PY:-python}
MODELS=${MODELS:-seq2seq,seq2seq_diffusion,seq2seq_ar_diffusion,gan,vae}

# 优先使用 TURB_IDS（逗号分隔），否则回退到单个 TURB_ID
TURB_IDS=${TURB_IDS:-}
TURB_ID=${TURB_ID:-90}
TARGET_IDS="${TURB_IDS:-$TURB_ID}"

LOGDIR="v2/logs"
mkdir -p "$LOGDIR"
NOW=$(date +"%Y%m%d_%H%M%S")

# 生成安全的文件名（将逗号替换为短横线）
IDS_SAFE=$(echo "$TARGET_IDS" | tr ',' '-')
OUT=${OUT:-"v2/results/turbine_forecasts_${IDS_SAFE}.csv"}
LOGFILE="$LOGDIR/export_turbine_${IDS_SAFE}_$NOW.log"

nohup $PY -m v2.src.export_turbine_forecasts \
  --models "$MODELS" \
  --turb-ids "$TARGET_IDS" \
  --out "$OUT" \
  --index 0 \
  > "$LOGFILE" 2>&1 &

echo "导出开始。风机: $TARGET_IDS, 模型: $MODELS"
echo "CSV 输出: $OUT"
echo "日志: $LOGFILE"