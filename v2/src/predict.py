import argparse
import os

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from .preprocess import load_and_clean
from .dataset import WindSeqIndexer, WindSeqDataset
from .model_seq2seq import Seq2Seq
from .model_seq2seq_diffusion import Seq2SeqDiffusion
from .model_seq2seq_ar_diffusion import Seq2SeqARDiffusion
from .model_gan import Seq2SeqGAN
from .model_vae import Seq2SeqVAE
from .utils import encode_time_features, set_seed, load_splits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["seq2seq", "seq2seq_diffusion", "seq2seq_ar_diffusion", "gan", "vae"], default="seq2seq")
    p.add_argument("--data", default="v2/results/cleaned.csv")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--id-col", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--wind-col", default=None)
    p.add_argument("--power-col", default=None)

    p.add_argument("--hist-len", type=int, default=24)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--time-encode", choices=["raw", "sin-cos"], default=None)
    p.add_argument("--emb-dim", type=int, default=16)

    p.add_argument("--checkpoint", default="v2/results/checkpoints/seq2seq/best.pth")
    p.add_argument("--samples", type=int, default=100, help="MC dropout sample count for scenarios.")
    p.add_argument("--seed", type=int, default=42, help="MC dropout sample count for scenarios.")
    p.add_argument("--out-prefix", default="v2/results/seq2seq")
    p.add_argument("--shuffle-split", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of test sample to save scenarios for (0-based).",
    )

    p.add_argument("--diffusion-timesteps", type=int, default=50, help="Number of diffusion timesteps")
    p.add_argument("--batch-size", type=int, default=1024, help="Batch size for DataLoader during prediction")

    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    print("Run args:", vars(args))
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs("v2/results", exist_ok=True)

    if args.raw:
        mode = args.time_encode or "raw"
        df, id_col, time_col, wind_col, power_col = load_and_clean(
            csv_path=args.data, out_csv_path="v2/results/cleaned.csv",
            id_col=args.id_col, time_col=args.time_col, wind_col=args.wind_col, power_col=args.power_col,
            time_encode=mode
        )
    else:
        import pandas as pd
        df = pd.read_csv(args.data, parse_dates=True)
        print(f"[DEBUG] Loaded {len(df)} rows from {args.data}")
        print(f"[DEBUG] hist_len={args.hist_len}, pred_len={args.pred_len}, stride={args.stride}")
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower():
                try:
                    df[c] = pd.to_datetime(df[c])
                except Exception:
                    pass
        from .utils import find_column, detect_time_column
        # 写死列名，不猜
        id_col   = "TurbID"
        time_col = "Datetime"
        wind_col = "Wspd"
        power_col = "Patv"
        print(f"[DEBUG] id_col={id_col}, time_col={time_col}, wind_col={wind_col}, power_col={power_col}")

        # Try to infer time encoding mode from pre-fit scalers if available.
        feature_scaler_path = "v2/results/artifacts/feature_scaler.joblib"
        exo_scaler_path = "v2/results/artifacts/exo_scaler.joblib"
        feature_scaler = None
        exo_scaler = None
        inferred_mode = None
        try:
            if os.path.exists(feature_scaler_path):
                feature_scaler = joblib.load(feature_scaler_path)
                feat_n = getattr(feature_scaler, "n_features_in_", None)
                if feat_n is None:
                    # fallback to length of mean_ if present
                    mean_attr = getattr(feature_scaler, "mean_", None)
                    if mean_attr is not None:
                        feat_n = len(mean_attr)
                # infer: raw -> 4 features (wind, power, hour, dayofweek), sin-cos -> 6 features (wind, power, hour_sin, hour_cos, dow_sin, dow_cos)
                if feat_n == 6:
                    inferred_mode = "sin-cos"
                elif feat_n == 4:
                    inferred_mode = "raw"
        except Exception:
            inferred_mode = None

        # Decide final mode: explicit CLI arg takes precedence, else inferred, else default to "raw"
        mode = args.time_encode if args.time_encode is not None else (inferred_mode or "raw")
        print(f"[DEBUG] Using time encoding mode: {mode} (inferred={inferred_mode})")

        # add time features using decided mode
        df = encode_time_features(df, time_col, mode=mode)
        print(f"[DEBUG] Data head:\n{df.head()}")

    # load checkpoint and config (source of truth) early to enforce strict consistency
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Enforce train-time hyperparams for strict consistency
    hist_len = int(config.get("hist_len", args.hist_len))
    pred_len = int(config.get("pred_len", args.pred_len))
    stride = int(config.get("stride", args.stride))
    mode_ckpt = config.get("time_encode", None)
    if mode_ckpt is not None and mode_ckpt != mode:
        print(f"[WARN] Overriding time encoding mode to checkpoint value: {mode_ckpt}")
        df = encode_time_features(df, time_col, mode=mode_ckpt)
        mode = mode_ckpt

    if mode == "sin-cos":
        feature_cols = [wind_col, power_col, "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        exo_cols = [wind_col, "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    else:
        feature_cols = [wind_col, power_col, "hour", "dayofweek"]
        exo_cols = [wind_col, "hour", "dayofweek"]
    target_col = power_col

    # Decide model name from checkpoint if present
    model_name = str(config.get("model", args.model or "seq2seq")).lower()

    indexer = WindSeqIndexer(df, id_col, time_col, feature_cols, target_col, hist_len, pred_len, stride, freq="1h")
    # Prefer saved splits; fall back to recompute if missing
    splits_path = f"v2/results/splits_{model_name}.json"
    try:
        splits = load_splits(splits_path, indexer.groups)
        print(f"[INFO] Loaded saved splits from: {splits_path} (train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)})")
    except Exception as e:
        print(f"[WARN] Failed to load saved splits ({e}). Recomputing splits.")
        splits = indexer.split_811(shuffle=args.shuffle_split, seed=args.shuffle_seed)
        print(f"[INFO] Recomputed splits: train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}")

    # Checkpoint already loaded and hyperparams enforced above.

    # Resolve scalers: prefer checkpoint-embedded; fallback to artifacts
    feature_scaler = ckpt.get("feature_scaler", None)
    target_scaler = ckpt.get("target_scaler", None)
    exo_scaler = ckpt.get("exo_scaler", None)
    if feature_scaler is None:
        feature_scaler = joblib.load("v2/results/artifacts/feature_scaler.joblib")
    if target_scaler is None:
        target_scaler = joblib.load("v2/results/artifacts/target_scaler.joblib")
    if exo_scaler is None:
        exo_scaler = joblib.load("v2/results/artifacts/exo_scaler.joblib")

    # Align turbine id->idx mapping with training
    id2idx_ckpt = ckpt.get("id2idx", None)
    if id2idx_ckpt is not None:
        id2idx_ckpt = {str(k): int(v) for k, v in id2idx_ckpt.items()}
        for g in indexer.groups:
            tid = str(g["id"])
            if tid in id2idx_ckpt:
                g["id_idx"] = id2idx_ckpt[tid]

        def _filter_indices(indices):
            out = []
            for g_idx, s, w in indices:
                tid = str(indexer.groups[g_idx]["id"])
                if tid in id2idx_ckpt:
                    out.append((g_idx, s, w))
            return out

        test_indices = _filter_indices(splits.test)
        n_turbines_ckpt = max(id2idx_ckpt.values()) + 1 if len(id2idx_ckpt) > 0 else indexer.n_turbines
    else:
        test_indices = splits.test
        n_turbines_ckpt = indexer.n_turbines

    test_ds = WindSeqDataset(indexer.groups, test_indices, feature_cols, target_col, hist_len, pred_len, exo_cols,
                             feature_scaler=feature_scaler, target_scaler=target_scaler, exo_scaler=exo_scaler)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[INFO] Test split samples: {len(test_indices)}")
    print(f"[INFO] Test loader batches: {len(test_loader)}")
    if len(test_indices) == 0:
        print("Error: No test samples constructed after windowing and split.")
        print("请检查 hist_len, pred_len, stride 参数，或数据清洗后是否还有足够的连续数据段。")
        print("建议：")
        print("1. 检查数据清洗逻辑是否过于严格导致数据量太少。")
        print("2. 调整 hist_len, pred_len, stride 参数，保证每个连续段长度大于 hist_len+pred_len。")
        print("3. 检查原始数据是否有大量缺失或中断。")
        return

    # device, checkpoint, and config already loaded above
    config = config
    hidden_size = config.get("hidden_size", 128)
    num_layers = config.get("num_layers", 2)
    dropout = config.get("dropout", 0.2)
    emb_dim = config.get("emb_dim", args.emb_dim)
    # Build model by name
    if model_name == "seq2seq":
        model_cls = Seq2Seq
    elif model_name in ("seq2seq_diffusion", "seq2seq+diffusion", "seq2seq_diff"):
        model_cls = Seq2SeqDiffusion
    elif model_name == "seq2seq_ar_diffusion":
        model_cls = Seq2SeqARDiffusion
    elif model_name == "gan":
        model_cls = Seq2SeqGAN
    elif model_name == "vae":
        model_cls = Seq2SeqVAE
    else:
        print(f"[WARN] Unknown model '{model_name}', defaulting to Seq2Seq")
        model_cls = Seq2Seq

    # For diffusion models, use checkpoint config to ensure architecture matches
    if model_name == "seq2seq_ar_diffusion":
        model = model_cls(
            input_dim=len(feature_cols),
            exo_dim=len(exo_cols),
            n_turbines=n_turbines_ckpt,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            timesteps=config.get("diffusion_timesteps", 100),
            beta_start=config.get("diffusion_beta_start", 1e-4),
            beta_end=config.get("diffusion_beta_end", 0.02),
            schedule=config.get("diffusion_schedule", "linear"),
            t_embed_dim=config.get("diffusion_t_embed_dim", 32),
            k_steps=config.get("diffusion_k_steps", 4),
        )
    else:
        model = model_cls(
            input_dim=len(feature_cols),
            exo_dim=len(exo_cols),
            n_turbines=n_turbines_ckpt,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Collect N examples for visualization (only those starting at 00:00)
    y_trues = []
    y_preds_mean = []
    y_preds_samples = []
    turb_ids = []  # 保存风机ID

    max_collect = np.inf  # 可扩展到指定样本数
    empty_loader = True
    offset = 0           # 全局样本偏移量（跨 batch 计数）
    total_kept = 0

    for batch in test_loader:
        empty_loader = False
        X = batch["X"].to(device)               # [B, hist, F]
        X_future = batch["X_future"].to(device) # [B, pred, exo_dim]
        y_true_batch = batch["y"].cpu().numpy() # [B, pred, 1]
        y0 = batch["y0_scaled"].to(device)      # [B, 1]
        turb_idx_batch = batch["turb_idx"].to(device)

        B = X.shape[0]
        keep_local_idx = []
        keep_turb_ids = []

        # 找出本批次中满足“预测起始时间为 0 点”的样本
        for i in range(B):
            global_idx = offset + i
            g_idx, seg_start, win_start = test_ds.indices[global_idx]
            g = test_ds.groups[g_idx]
            gdf = g["df"]
            s = seg_start + win_start
            he = s + hist_len
            pred_start_time = gdf.iloc[he]["Datetime"]
            hour_val = int(getattr(pred_start_time, "hour", -1))
            if hour_val == 0:
                keep_local_idx.append(i)
                keep_turb_ids.append(g["id"])

        # 若本批没有符合条件的样本，跳过
        if len(keep_local_idx) == 0:
            offset += B
            continue

        # 取子集进行推理
        X_sel = X[keep_local_idx]
        Xf_sel = X_future[keep_local_idx]
        y0_sel = y0[keep_local_idx]
        turb_idx_sel = turb_idx_batch[keep_local_idx]
        y_true_sel = y_true_batch[keep_local_idx]  # [b_sel, pred, 1]

        # MC Dropout scenarios: 训练模式开启 dropout，但无梯度
        model.train()
        preds_samples = []
        for _ in range(args.samples):
            yhat_scaled = model(X_sel, Xf_sel, y0_sel, turb_idx_sel, pred_steps=y_true_sel.shape[1], teacher_forcing=0.0)
            yhat = target_scaler.inverse_transform(
                yhat_scaled.detach().cpu().numpy().reshape(-1, 1)
            ).reshape(y_true_sel.shape)  # [b_sel, pred, 1]
            preds_samples.append(yhat[:, :, 0])  # [b_sel, pred]
        model.eval()

        preds_samples_np = np.stack(preds_samples, axis=0)  # [S, b_sel, pred]
        mean_np = preds_samples_np.mean(axis=0)             # [b_sel, pred]

        # 逐样本落盘收集
        b_sel = len(keep_local_idx)
        for j in range(b_sel):
            y_trues.append(y_true_sel[j, :, 0])
            y_preds_mean.append(mean_np[j])
            y_preds_samples.append(preds_samples_np[:, j, :])
            turb_ids.append(keep_turb_ids[j])
            total_kept += 1
            if total_kept >= max_collect:
                break

        if total_kept >= max_collect:
            break

        offset += B

    if empty_loader or len(y_trues) == 0 or len(y_preds_mean) == 0:
        print("Error: No test samples found. Please check your data and split settings.")
        return

    y_trues = np.stack(y_trues, axis=0)                  # [N, pred]
    y_preds_mean = np.stack(y_preds_mean, axis=0)        # [N, pred]
    turb_ids = np.array(turb_ids)                        # [N]
    
    # 保存所有样本的场景序列，形状为 [N, S, pred]
    if len(y_preds_samples) > 0:
        y_samples_all = np.stack(y_preds_samples, axis=0)  # [N, S, pred]
    else:
        y_samples_all = np.zeros((len(y_trues), args.samples, pred_len))

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    np.save(f"{args.out_prefix}_y_true.npy", y_trues)
    np.save(f"{args.out_prefix}_y_pred_mean.npy", y_preds_mean)
    np.save(f"{args.out_prefix}_turb_ids.npy", turb_ids)  # 新增：保存风机ID
    # 保存所有样本的场景序列
    np.save(f"{args.out_prefix}_y_samples_all.npy", y_samples_all)
    # 兼容旧脚本，继续保存第一条
    if y_samples_all.shape[0] > 0:
        np.save(f"{args.out_prefix}_y_samples_first.npy", y_samples_all[0])
    else:
        np.save(f"{args.out_prefix}_y_samples_first.npy", np.zeros((args.samples, pred_len)))

    print("Saved predictions to prefix:", args.out_prefix)
    print(f"Filtered {len(y_trues)} samples that start at 0:00 hour")


if __name__ == "__main__":
    main()