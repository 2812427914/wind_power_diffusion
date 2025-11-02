import argparse
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from .preprocess import load_and_clean
from .dataset import WindSeqIndexer, WindSeqDataset
from .model_seq2seq import Seq2Seq
from .model_seq2seq_diffusion import Seq2SeqDiffusion
from .model_gan import Seq2SeqGAN
from .model_vae import Seq2SeqVAE
from .utils import set_seed, mae, rmse, save_splits, load_splits


def train_epoch(model, loader, device, criterion, teacher_forcing):
    model.train()
    total_loss = 0.0
    for batch in loader:
        X = batch["X"].to(device)  # [B, hist, F]
        X_future = batch["X_future"].to(device)  # [B, pred, exo_dim]
        y_scaled = batch["y_scaled"].to(device)  # [B, pred, 1]
        y0 = batch["y0_scaled"].to(device)  # [B, 1]
        turb_idx = batch["turb_idx"].to(device)  # [B]

        optimizer.zero_grad()
        yhat_scaled = model(X, X_future, y0, turb_idx, pred_steps=y_scaled.size(1), teacher_forcing=teacher_forcing, y_truth=y_scaled)
        loss = criterion(yhat_scaled, y_scaled)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device, criterion, target_scaler):
    model.eval()
    total_loss = 0.0
    ys = []
    yhs = []
    for batch in loader:
        X = batch["X"].to(device)
        X_future = batch["X_future"].to(device)
        y_scaled = batch["y_scaled"].to(device)
        y_true = batch["y"].cpu().numpy()  # original units

        y0 = batch["y0_scaled"].to(device)
        turb_idx = batch["turb_idx"].to(device)
        yhat_scaled = model(X, X_future, y0, turb_idx, pred_steps=y_scaled.size(1), teacher_forcing=0.0, y_truth=None)
        loss = criterion(yhat_scaled, y_scaled)
        total_loss += loss.item() * X.size(0)

        yhat = target_scaler.inverse_transform(yhat_scaled.cpu().numpy().reshape(-1, 1)).reshape(y_true.shape)
        ys.append(y_true)
        yhs.append(yhat)
    y_true_all = np.concatenate(ys, axis=0)
    y_pred_all = np.concatenate(yhs, axis=0)

    metrics = {
        "mae": mae(y_true_all, y_pred_all),
        "rmse": rmse(y_true_all, y_pred_all),
        "mse_scaled": total_loss / len(loader.dataset),
    }
    return metrics

@torch.no_grad()
def predict_loader(model, loader, device, target_scaler):
    """
    Run model on a loader and return (y_true_all, y_pred_all) in original units.
    """
    model.eval()
    ys = []
    yhs = []
    for batch in loader:
        X = batch["X"].to(device)
        X_future = batch["X_future"].to(device)
        y_true = batch["y"].cpu().numpy()  # [B, T, 1]

        y0 = batch["y0_scaled"].to(device)
        turb_idx = batch["turb_idx"].to(device)
        yhat_scaled = model(X, X_future, y0, turb_idx, pred_steps=y_true.shape[1], teacher_forcing=0.0)
        yhat = target_scaler.inverse_transform(yhat_scaled.cpu().numpy().reshape(-1, 1)).reshape(y_true.shape)

        ys.append(y_true)
        yhs.append(yhat)

    if len(ys) == 0:
        return np.zeros((0, 0, 0)), np.zeros((0, 0, 0))
    y_true_all = np.concatenate(ys, axis=0)  # [N, T, 1]
    y_pred_all = np.concatenate(yhs, axis=0)  # [N, T, 1]
    return y_true_all, y_pred_all


def linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    ratio = min(1.0, step / (total_steps - 1))
    return start + (end - start) * ratio


def ensure_dirs(model_name: str):
    os.makedirs(f"v2/results/checkpoints/{model_name}", exist_ok=True)
    os.makedirs("v2/results/artifacts", exist_ok=True)
    os.makedirs("v2/results/plots", exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["seq2seq", "seq2seq_diffusion", "gan", "vae"], default="seq2seq")
    p.add_argument("--data", default="data/wtbdata_hourly.csv")
    p.add_argument("--id-col", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--wind-col", default=None)
    p.add_argument("--power-col", default=None)

    p.add_argument("--hist-len", type=int, default=24)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--time-encode", choices=["raw", "sin-cos"], default="raw")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--emb-dim", type=int, default=16)

    p.add_argument("--teacher-start", type=float, default=1.0)
    p.add_argument("--teacher-end", type=float, default=0.2)

    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shuffle-split", action="store_true")
    p.add_argument("--shuffle-seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Run args:", vars(args))
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    ensure_dirs(args.model)

    # 1) Preprocess
    cleaned_csv = "v2/results/cleaned.csv"
    df, id_col, time_col, wind_col, power_col = load_and_clean(
        csv_path=args.data,
        id_col=args.id_col,
        time_col=args.time_col,
        wind_col=args.wind_col,
        power_col=args.power_col,
        out_csv_path=cleaned_csv,
        time_encode=args.time_encode,
    )

    if args.time_encode == "sin-cos":
        feature_cols = [wind_col, power_col, "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        exo_cols = [wind_col, "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    else:
        feature_cols = [wind_col, power_col, "hour", "dayofweek"]
        exo_cols = [wind_col, "hour", "dayofweek"]
    target_col = power_col

    # 2) Build indexer and splits
    indexer = WindSeqIndexer(
        df=df,
        id_col=id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        target_col=target_col,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        stride=args.stride,
        freq="1h",  # 用小写 h，避免 FutureWarning
    )
    # Load existing splits if present; otherwise compute and save once
    splits_path = f"v2/results/splits_{args.model}.json"
    try:
        if os.path.exists(splits_path):
            splits = load_splits(splits_path, indexer.groups)
            print(f"[INFO] Loaded saved splits from: {splits_path}")
        else:
            splits = indexer.split_811(shuffle=args.shuffle_split, seed=args.shuffle_seed)
            save_splits(splits_path, splits, indexer.groups)
            print(f"[INFO] Saved new splits to: {splits_path}")
    except Exception as e:
        print(f"[WARN] Failed to load/save splits ({e}). Falling back to newly computed splits.")
        splits = indexer.split_811(shuffle=args.shuffle_split, seed=args.shuffle_seed)

    print(f"Samples -> train: {len(splits.train)}, val: {len(splits.val)}, test: {len(splits.test)}")
    # 打印部分样本详细信息
    def print_split_samples(split, name, groups, num=5):
        print(f"\n{name} samples (show {num}):")
        for i in range(min(num, len(split))):
            g_idx, seg_start, win_start = split[i]
            g = groups[g_idx]
            df = g["df"]
            s = seg_start + win_start
            he = s + args.hist_len
            pe = he + args.pred_len
            turb_id = g["id"]
            start_time = df.iloc[s]["Datetime"]
            end_time = df.iloc[pe-1]["Datetime"]
            print(f"  [{i}] TurbID={turb_id}, start_idx={s}, hist=({start_time}~{df.iloc[he-1]['Datetime']}), pred=({df.iloc[he]['Datetime']}~{end_time})")
    print_split_samples(splits.train, "Train", indexer.groups)
    print_split_samples(splits.val, "Val", indexer.groups)
    print_split_samples(splits.test, "Test", indexer.groups)

    if len(splits.train) == 0:
        raise ValueError(
            f"No training samples constructed after windowing and 8:1:1 split.\n"
            f"请检查 hist_len={args.hist_len}, pred_len={args.pred_len}, stride={args.stride} 参数，"
            f"以及数据清洗后是否还有足够的连续数据段。"
            f"\n建议：\n"
            f"1. 检查数据清洗逻辑是否过于严格导致数据量太少。\n"
            f"2. 调整 hist_len, pred_len, stride 参数，保证每个连续段长度大于 hist_len+pred_len。\n"
            f"3. 检查原始数据是否有大量缺失或中断。"
        )

    # 3) Datasets and scalers
    train_ds = WindSeqDataset(indexer.groups, splits.train, feature_cols, target_col, args.hist_len, args.pred_len, exo_cols, fit_scalers=True)
    val_ds = WindSeqDataset(indexer.groups, splits.val, feature_cols, target_col, args.hist_len, args.pred_len, exo_cols,
                            feature_scaler=train_ds.feature_scaler, target_scaler=train_ds.target_scaler, exo_scaler=train_ds.exo_scaler)
    test_ds = WindSeqDataset(indexer.groups, splits.test, feature_cols, target_col, args.hist_len, args.pred_len, exo_cols,
                             feature_scaler=train_ds.feature_scaler, target_scaler=train_ds.target_scaler, exo_scaler=train_ds.exo_scaler)

    joblib.dump(train_ds.feature_scaler, "v2/results/artifacts/feature_scaler.joblib")
    joblib.dump(train_ds.target_scaler, "v2/results/artifacts/target_scaler.joblib")
    joblib.dump(train_ds.exo_scaler, "v2/results/artifacts/exo_scaler.joblib")

        # 打印 scale_ 留证
    print("[DEBUG] target_scaler.scale_ :", getattr(train_ds.target_scaler, "scale_", None))
    
    # debug: 打印 scaler 的 mean_ 和 var_，确保和评估时一致
    print("[DEBUG] train_ds.feature_scaler.mean_:", getattr(train_ds.feature_scaler, "mean_", None))
    print("[DEBUG] train_ds.target_scaler.mean_:", getattr(train_ds.target_scaler, "mean_", None))
    print("[DEBUG] train_ds.exo_scaler.mean_:", getattr(train_ds.exo_scaler, "mean_", None))

    # 4) Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build model by name
    if args.model == "seq2seq":
        model_cls = Seq2Seq
    elif args.model in ("seq2seq_diffusion", "seq2seq+diffusion", "seq2seq_diff"):
        model_cls = Seq2SeqDiffusion
    elif args.model == "gan":
        model_cls = Seq2SeqGAN
    elif args.model == "vae":
        model_cls = Seq2SeqVAE
    else:
        print(f"[WARN] Unknown model '{args.model}', defaulting to Seq2Seq")
        model_cls = Seq2Seq

    model = model_cls(
        input_dim=len(feature_cols),
        exo_dim=len(exo_cols),
        n_turbines=indexer.n_turbines,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # 5) Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_rmse = float("inf")
    best_path = f"v2/results/checkpoints/{args.model}/best.pth"
    last_path = f"v2/results/checkpoints/{args.model}/last.pth"
    patience = args.patience
    wait = 0

    total_steps = max(1, args.epochs - 1)
    for epoch in range(1, args.epochs + 1):
        teacher_forcing = linear_schedule(args.teacher_start, args.teacher_end, epoch - 1, total_steps)
        train_loss = train_epoch(model, train_loader, device, criterion, teacher_forcing)
        val_metrics = eval_epoch(model, val_loader, device, criterion, train_ds.target_scaler)
        # test_metrics = eval_epoch(model, test_loader, device, criterion, train_ds.target_scaler)

        # 仅打印训练/验证信息，验证集的预测保存将在发现更好模型时进行
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} "
              f"train_mse_scaled={train_loss:.6f} "
              f"val_mae={val_metrics['mae']:.3f} val_rmse={val_metrics['rmse']:.3f} "
            #   f"test_mae={test_metrics['mae']:.3f} test_rmse={test_metrics['rmse']:.3f} "
              f"tf={teacher_forcing:.2f}"
              f"lr={current_lr:.6g}")

        # scheduler.step(val_metrics["rmse"])

        # checkpointing
        torch.save({
            "model_state": model.state_dict(),
            "config": vars(args),
            "feature_scaler": train_ds.feature_scaler,
            "target_scaler": train_ds.target_scaler,
            "exo_scaler": train_ds.exo_scaler,
            "id2idx": {str(k): int(v) for k, v in indexer.id2idx.items()},
        }, last_path)
        improved = val_metrics["rmse"] < best_rmse - 1e-6
        if improved:
            best_rmse = val_metrics["rmse"]
            torch.save({
                "model_state": model.state_dict(),
                "config": vars(args),
                "feature_scaler": train_ds.feature_scaler,
                "target_scaler": train_ds.target_scaler,
                "exo_scaler": train_ds.exo_scaler,
                "id2idx": {str(k): int(v) for k, v in indexer.id2idx.items()},
            }, best_path)
            # 仅在发现更好模型时，保存验证集的真实值与预测值，便于后续可视化使用（避免每个 epoch 重复写入）
            try:
                y_val_true, y_val_pred = predict_loader(model, val_loader, device, train_ds.target_scaler)
                if isinstance(y_val_true, np.ndarray) and y_val_true.size > 0:
                    np.save(f"v2/results/{args.model}_val_y_true.npy", y_val_true.squeeze())   # 保存为 [N, T]
                    np.save(f"v2/results/{args.model}_val_y_pred_mean.npy", y_val_pred.squeeze())  # 保存为 [N, T]
                    print(f"Saved validation predictions (best) to: v2/results/{args.model}_val_y_true.npy and v2/results/{args.model}_val_y_pred_mean.npy")
            except Exception as e:
                print("Warning: failed to save validation predictions on best:", e)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best val RMSE: {best_rmse:.3f}")
                break

    print("Training complete. Best checkpoint at:", best_path)