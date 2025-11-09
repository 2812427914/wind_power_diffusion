import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import WindSeqIndexer, WindSeqDataset
from .model_seq2seq import Seq2Seq
from .model_seq2seq_diffusion import Seq2SeqDiffusion
from .model_gan import Seq2SeqGAN
from .model_vae import Seq2SeqVAE
from .utils import (
    encode_time_features,
    find_column,
    detect_time_column,
    set_seed,
    save_json,
    load_json,
)
from .train import eval_epoch, eval_epoch_ar_diff


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["seq2seq", "seq2seq_diffusion", "gan", "vae"], default="seq2seq", help="Model name.")
    p.add_argument("--data", default="v2/results/cleaned.csv", help="Path to cleaned CSV or raw CSV.")
    p.add_argument("--id-col", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--wind-col", default=None)
    p.add_argument("--power-col", default=None)

    # 这些 CLI 参数仅用于校验；实际评估强制使用 checkpoint 的配置，确保与训练完全一致
    p.add_argument("--hist-len", type=int, default=None)
    p.add_argument("--pred-len", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)

    p.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (*.pth)")
    p.add_argument("--splits-path", default="v2/results/splits_seq2seq.json", help="Path to saved splits json from training.")
    p.add_argument("--batch-size", type=int, default=None, help="Eval batch size; default to train batch-size if present, else 256.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="v2/results/metrics_seq2seq.json")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    print("Run args:", vars(args))
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # load checkpoint and config (source of truth)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    if not isinstance(config, dict):
        config = {}
        print(f"no cnofig found")
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    # enforce train-time hyperparams to guarantee consistency
    hist_len = int(config.get("hist_len", 24))
    pred_len = int(config.get("pred_len", 24))
    stride = int(config.get("stride", 1))
    time_encode_mode = config.get("time_encode", "raw")
    emb_dim = int(config.get("emb_dim", 16))
    hidden_size = int(config.get("hidden_size", 128))
    num_layers = int(config.get("num_layers", 2))
    dropout = float(config.get("dropout", 0.2))
    eval_bs = int(args.batch_size or config.get("batch_size", 256))

    # optional sanity check
    mismatches = []
    if args.hist_len is not None and args.hist_len != hist_len:
        mismatches.append(("hist_len", args.hist_len, hist_len))
    if args.pred_len is not None and args.pred_len != pred_len:
        mismatches.append(("pred_len", args.pred_len, pred_len))
    if args.stride is not None and args.stride != stride:
        mismatches.append(("stride", args.stride, stride))
    if mismatches:
        print("[WARN] CLI params differ from checkpoint config; will use checkpoint values for strict consistency:")
        for k, cli_v, ckpt_v in mismatches:
            print(f"  - {k}: cli={cli_v} -> ckpt={ckpt_v}")

    # load data
    df = pd.read_csv(args.data)
    id_col   = "TurbID"
    time_col = "Datetime"
    wind_col = "Wspd"
    power_col = "Patv"
    # encode time features according to training mode
    df = encode_time_features(df, time_col, mode=time_encode_mode)

    # feature/exo columns (must mirror training)
    if time_encode_mode == "sin-cos":
        feature_cols = [wind_col, power_col, "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        exo_cols = [wind_col, "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    else:
        feature_cols = [wind_col, power_col, "hour", "dayofweek"]
        exo_cols = [wind_col, "hour", "dayofweek"]
    target_col = power_col

    # indexer
    indexer = WindSeqIndexer(
        df=df,
        id_col=id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        target_col=target_col,
        hist_len=hist_len,
        pred_len=pred_len,
        stride=stride,
        freq="1h",
    )

    # load splits saved during training; do NOT recompute to ensure strict consistency
    if not os.path.exists(args.splits_path):
        print(f"[ERROR] Saved splits not found at {args.splits_path}. Please run training first to save splits.")
        sys.exit(1)
    from .utils import load_splits
    splits = load_splits(args.splits_path, indexer.groups)
    print(f"[INFO] Loaded splits: train={len(splits.train)}, val={len(splits.val)}, test={len(splits.test)}")

    # 只从 checkpoint 拿 scaler，绝不 fallback
    feature_scaler = ckpt["feature_scaler"]
    target_scaler = ckpt["target_scaler"]
    exo_scaler    = ckpt["exo_scaler"]
    print("[DEBUG] target_scaler.scale_ :", getattr(target_scaler, "scale_", None))

    # # load scalers from checkpoint; fallback to artifacts if missing
    # feature_scaler = ckpt.get("feature_scaler", None)
    # target_scaler = ckpt.get("target_scaler", None)
    # exo_scaler = ckpt.get("exo_scaler", None)
    # if feature_scaler is None or target_scaler is None or exo_scaler is None:
    #     print("[WARN] Scalers not embedded in checkpoint; falling back to artifacts in v2/results/artifacts")
    #     feature_scaler = feature_scaler or joblib.load("v2/results/artifacts/feature_scaler.joblib")
    #     target_scaler = target_scaler or joblib.load("v2/results/artifacts/target_scaler.joblib")
    #     exo_scaler = exo_scaler or joblib.load("v2/results/artifacts/exo_scaler.joblib")
    # debug: 打印 scaler 的 mean_ 和 var_，确保和训练时一致
    print("[DEBUG] feature_scaler.mean_:", getattr(feature_scaler, "mean_", None))
    print("[DEBUG] target_scaler.mean_:", getattr(target_scaler, "mean_", None))
    print("[DEBUG] exo_scaler.mean_:", getattr(exo_scaler, "mean_", None))

    # align turbine id mapping with training
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

        val_indices = _filter_indices(splits.val)
        test_indices = _filter_indices(splits.test)
        n_turbines_ckpt = max(id2idx_ckpt.values()) + 1 if len(id2idx_ckpt) > 0 else len(id2idx_ckpt)
    else:
        print("[WARN] id2idx not found in checkpoint; using current indexer mapping. This may reduce strictness if groups changed.")
        val_indices = splits.val
        test_indices = splits.test
        n_turbines_ckpt = indexer.n_turbines

    # 打印验证集样本信息（show 5）
    def print_split_samples(split, name, groups, num=5):
        print(f"\n{name} samples (show {num}):")
        for i in range(min(num, len(split))):
            g_idx, seg_start, win_start = split[i]
            g = groups[g_idx]
            df_g = g["df"]
            s = seg_start + win_start
            he = s + hist_len
            pe = he + pred_len
            turb_id = g["id"]
            start_time = df_g.iloc[s][time_col]
            end_time = df_g.iloc[pe - 1][time_col]
            print(f"  [{i}] TurbID={turb_id}, start_idx={s}, hist=({start_time}~{df_g.iloc[he-1][time_col]}), pred=({df_g.iloc[he][time_col]}~{end_time})")
    print_split_samples(val_indices, "Val", indexer.groups)

    # datasets and loaders (no fitting scalers here)
    val_ds = WindSeqDataset(
        indexer.groups, val_indices,
        feature_cols, target_col, hist_len, pred_len, exo_cols,
        feature_scaler=feature_scaler, target_scaler=target_scaler, exo_scaler=exo_scaler
    )
    test_ds = WindSeqDataset(
        indexer.groups, test_indices,
        feature_cols, target_col, hist_len, pred_len, exo_cols,
        feature_scaler=feature_scaler, target_scaler=target_scaler, exo_scaler=exo_scaler
    )

    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[INFO] Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

    # model selection
    model_name = str(config.get("model", args.model or "seq2seq")).lower()
    if model_name == "seq2seq":
        model_cls = Seq2Seq
    elif model_name in ("seq2seq_diffusion", "seq2seq+diffusion", "seq2seq_diff"):
        model_cls = Seq2SeqDiffusion
    elif model_name == "gan":
        model_cls = Seq2SeqGAN
    elif model_name == "vae":
        model_cls = Seq2SeqVAE
    else:
        print(f"[WARN] Unknown model '{model_name}', defaulting to Seq2Seq")
        model_cls = Seq2Seq

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

    # criterion and evaluation (reuse train.eval_epoch for strict consistency)
    criterion = nn.MSELoss()
    if model_name == "seq2seq_ar_diffusion":
        val_metrics = eval_epoch_ar_diff(model, val_loader, device, target_scaler)
        test_metrics = eval_epoch_ar_diff(model, test_loader, device, target_scaler)
    else:
        val_metrics = eval_epoch(model, val_loader, device, criterion, target_scaler)
        test_metrics = eval_epoch(model, test_loader, device, criterion, target_scaler)

    # save metrics
    metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "config_used": {
            "hist_len": hist_len,
            "pred_len": pred_len,
            "stride": stride,
            "time_encode": time_encode_mode,
            "batch_size_eval": eval_bs,
        },
        "checkpoint": os.path.abspath(args.checkpoint),
        "splits_path": os.path.abspath(args.splits_path),
        "data": os.path.abspath(args.data),
    }
    save_json(metrics, args.out)
    print(f"[INFO] Saved metrics to: {args.out}")
    print("[RESULT] val:", val_metrics)
    print("[RESULT] test:", test_metrics)

    # debug: 直接对比训练时保存的 val 预测和当前评估指标
    try:
        val_true_path = f"v2/results/{model_name}_val_y_true.npy"
        val_pred_path = f"v2/results/{model_name}_val_y_pred_mean.npy"
        if os.path.exists(val_true_path) and os.path.exists(val_pred_path):
            val_true = np.load(val_true_path)
            val_pred = np.load(val_pred_path)

            from .utils import mae, rmse
            val_mae_saved = mae(val_true, val_pred)
            val_rmse_saved = rmse(val_true, val_pred)
            print(f"[DEBUG] Saved val MAE: {val_mae_saved:.3f}, RMSE: {val_rmse_saved:.3f} (from npy)")
        else:
            print("[DEBUG] Saved val prediction npy not found for direct comparison.")
    except Exception as e:
        print("[DEBUG] Error comparing saved val npy:", e)


    # ------------------------------------------------------------------
    # 严格一致性核查：重新跑验证集 vs 训练时保存的 val_pred
    # ------------------------------------------------------------------
    from .utils import mae, rmse
    import os.path as osp

    val_true_path = f"v2/results/{model_name}_val_y_true.npy"
    val_pred_path = f"v2/results/{model_name}_val_y_pred_mean.npy"

    if osp.exists(val_true_path) and osp.exists(val_pred_path):
        y_true_saved = np.load(val_true_path)          # [N, T]
        y_pred_saved = np.load(val_pred_path)          # [N, T]

        # 重新跑一次验证集，收集 y_true / y_pred
        model.eval()
        y_true_recompute, y_pred_recompute = [], []
        for batch in val_loader:
            X = batch["X"].to(device, non_blocking=True)
            X_future = batch["X_future"].to(device, non_blocking=True)
            y0  = batch["y0_scaled"].to(device, non_blocking=True)
            turb_idx = batch["turb_idx"].to(device, non_blocking=True)

            # 前向
            yhat_scaled = model(X, X_future, y0, turb_idx,
                                pred_steps=pred_len, teacher_forcing=0.0)
            yhat = target_scaler.inverse_transform(
                yhat_scaled.cpu().numpy().reshape(-1, 1)
            ).reshape(yhat_scaled.shape)          # [B, T, 1]
            y_true_recompute.append(batch["y"].numpy())   # [B, T, 1]
            y_pred_recompute.append(yhat)

        y_true_recompute = np.concatenate(y_true_recompute, axis=0).squeeze(-1)  # [N, T]
        y_pred_recompute = np.concatenate(y_pred_recompute, axis=0).squeeze(-1)  # [N, T]

        # 指标
        mae_re  = mae(y_true_recompute, y_pred_recompute)
        rmse_re = rmse(y_true_recompute, y_pred_recompute)
        mae_sv  = mae(y_true_saved, y_pred_saved)
        rmse_sv = rmse(y_true_saved, y_pred_saved)

        print(f"\n[一致性核查]")
        print(f"  保存值  — MAE: {mae_sv:7.3f}   RMSE: {rmse_sv:7.3f}")
        print(f"  重跑值  — MAE: {mae_re:7.3f}   RMSE: {rmse_re:7.3f}")
        print(f"  差异    — MAE: {abs(mae_sv - mae_re):7.3f}   RMSE: {abs(rmse_sv - rmse_re):7.3f}")

        # 若差异大，打印前 5 条样本
        if abs(mae_sv - mae_re) > 1e-3:
            print("\n  前 5 条样本对比（true | saved_pred | recompute_pred | diff）：")
            for i in range(5):
                t, ps, pr = y_true_saved[i, :], y_pred_saved[i, :], y_pred_recompute[i, :]
                print(f"    sample-{i}: true={t[0]:7.2f}  saved={ps[0]:7.2f}  recompute={pr[0]:7.2f}  diff={abs(ps[0]-pr[0]):7.2f}")
    else:
        print("\n[一致性核查] 保存的 val_pred .npy 不存在，跳过。")


if __name__ == "__main__":
    main()