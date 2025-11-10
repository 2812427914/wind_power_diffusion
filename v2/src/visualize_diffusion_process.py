import argparse
import os
import json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .dataset import WindSeqIndexer, WindSeqDataset
from .model_seq2seq_ar_diffusion import Seq2SeqARDiffusion
from .utils import encode_time_features, load_splits, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Visualize forward (noising) and reverse (denoising) processes for 24h sequences")
    p.add_argument("--checkpoint", required=True, help="Path to trained diffusion model checkpoint (*.pth)")
    p.add_argument("--data", default="v2/results/cleaned.csv", help="Path to cleaned data CSV")
    p.add_argument("--out", default="v2/results/plots/seq2seq_ar_diffusion_diffusion_process.png", help="Output image path")
    p.add_argument("--timesteps", type=int, default=10, help="Number of diffusion steps to display (columns)")
    p.add_argument("--turb-id", type=int, default=90, help="Turbine ID to visualize; if not found, fallback to first available")
    p.add_argument("--sample-index", type=int, default=0, help="Fallback index within test split if turb-id not present")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load checkpoint and config
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {}) or {}
    model_name = str(config.get("model", "seq2seq_ar_diffusion")).lower()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint} (model={model_name})")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Enforce train-time hyperparams
    hist_len = int(config.get("hist_len", 24))
    pred_len = int(config.get("pred_len", 24))
    stride = int(config.get("stride", 1))
    time_encode_mode = config.get("time_encode", "raw")
    emb_dim = int(config.get("emb_dim", 16))
    hidden_size = int(config.get("hidden_size", 128))
    num_layers = int(config.get("num_layers", 2))
    dropout = float(config.get("dropout", 0.2))

    # Diffusion params (strictly use checkpoint)
    timesteps_total = int(config.get("diffusion_timesteps", 100))
    beta_start = float(config.get("diffusion_beta_start", 1e-4))
    beta_end = float(config.get("diffusion_beta_end", 0.02))
    schedule = str(config.get("diffusion_schedule", "linear"))
    t_embed_dim = int(config.get("diffusion_t_embed_dim", 32))
    k_steps = int(config.get("diffusion_k_steps", 4))

    # Load data and encode time features using train-time mode
    df = pd.read_csv(args.data)
    id_col, time_col, wind_col, power_col = "TurbID", "Datetime", "Wspd", "Patv"
    df = encode_time_features(df, time_col, mode=time_encode_mode)

    if time_encode_mode == "sin-cos":
        feature_cols = [wind_col, power_col, "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        exo_cols = [wind_col, "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    else:
        feature_cols = [wind_col, power_col, "hour", "dayofweek"]
        exo_cols = [wind_col, "hour", "dayofweek"]
    target_col = power_col

    # Build indexer (freq lowercase '1h' for consistency)
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

    # Load saved splits for this model; fallback to recompute if missing
    splits_path = f"v2/results/splits_{model_name}.json"
    try:
        splits = load_splits(splits_path, indexer.groups)
        print(f"[INFO] Loaded splits from {splits_path} (val={len(splits.val)}, test={len(splits.test)})")
    except Exception as e:
        print(f"[WARN] Failed to load splits: {e}; recomputing 8:1:1 splits")
        splits = indexer.split_811(shuffle=False)

    # Align turbine id mapping with training
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
        n_turbines_ckpt = (max(id2idx_ckpt.values()) + 1) if len(id2idx_ckpt) > 0 else 0
    else:
        print("[WARN] id2idx not in checkpoint; using current indexer mapping")
        test_indices = splits.test
        n_turbines_ckpt = indexer.n_turbines

    # Load scalers strictly from checkpoint
    feature_scaler = ckpt["feature_scaler"]
    target_scaler = ckpt["target_scaler"]
    exo_scaler = ckpt["exo_scaler"]

    # Dataset only for test split; no fitting
    test_ds = WindSeqDataset(
        indexer.groups, test_indices,
        feature_cols, target_col, hist_len, pred_len, exo_cols,
        feature_scaler=feature_scaler, target_scaler=target_scaler, exo_scaler=exo_scaler
    )

    # Select one sample:
    # Priority:
    # 1) target turb-id AND pred-start at 00:00
    # 2) any turb-id with pred-start at 00:00
    # 3) target turb-id (any start hour)
    # 4) fallback to --sample-index
    candidates_tid_midnight = []
    candidates_midnight = []
    candidates_tid_any = []
    for i, (g_idx, seg_start, win_start) in enumerate(test_ds.indices):
        g = indexer.groups[g_idx]
        df_g = g["df"]
        s = seg_start + win_start
        he = s + hist_len
        try:
            pred_start_time = df_g.iloc[he][time_col]
        except Exception:
            pred_start_time = df_g.iloc[he][indexer.time_col]
        try:
            hour_val = int(pd.to_datetime(pred_start_time).hour)
        except Exception:
            hour_val = int(getattr(pred_start_time, "hour", -1))

        if g["id"] == args.turb_id and hour_val == 0:
            candidates_tid_midnight.append(i)
        if hour_val == 0:
            candidates_midnight.append(i)
        if g["id"] == args.turb_id:
            candidates_tid_any.append(i)

    chosen_idx = None
    if len(candidates_tid_midnight) > 0:
        chosen_idx = candidates_tid_midnight[0]
        print(f"[INFO] Chosen sample: turb_id={args.turb_id} starting at 00:00 (index {chosen_idx})")
    elif len(candidates_midnight) > 0:
        chosen_idx = candidates_midnight[0]
        print(f"[INFO] Chosen sample: first midnight-start sample (index {chosen_idx}), turb_id={indexer.groups[test_ds.indices[chosen_idx][0]]['id']}")
    elif len(candidates_tid_any) > 0:
        chosen_idx = candidates_tid_any[0]
        print(f"[INFO] Chosen sample: turb_id={args.turb_id} (any start hour) index {chosen_idx}")
    else:
        chosen_idx = min(max(0, args.sample_index), len(test_ds) - 1)
        print(f"[WARN] No midnight-start sample found; fallback to sample-index={chosen_idx}")

    sample = test_ds[chosen_idx]
    turb_id = sample["turb_id"]
    print(f"[INFO] Visualizing sample {chosen_idx} for Turbine={turb_id}")

    # Prepare tensors
    X_hist = sample["X"].unsqueeze(0).to(device)           # [1, hist, F]
    X_future = sample["X_future"].unsqueeze(0).to(device)  # [1, pred, exo_dim]
    y_true = sample["y"].numpy().squeeze(-1)               # [pred]
    y_scaled = sample["y_scaled"].unsqueeze(0).to(device)  # [1, pred, 1]
    y0_scaled = sample["y0_scaled"].unsqueeze(0).to(device)  # [1, 1]
    turb_idx = sample["turb_idx"].unsqueeze(0).to(device)  # [1]

    # Determine clamp range from data distribution to avoid Inf/NaN after inverse transform
    vals = df[target_col].values
    vals = vals[np.isfinite(vals)]
    if len(vals) > 0:
        pmin = float(np.nanmin(vals))
        pmax = float(np.nanmax(vals))
    else:
        pmin, pmax = float(np.nanmin(df[target_col])), float(np.nanmax(df[target_col]))
    clamp_pad = max(1e-3, 0.05 * (pmax - pmin))
    clamp_min, clamp_max = pmin - clamp_pad, pmax + clamp_pad

    # sanitize y_true for plotting stability
    y_true = np.clip(np.nan_to_num(y_true, nan=0.0, posinf=clamp_max, neginf=clamp_min), clamp_min, clamp_max)

    # Build model strictly as training
    model = Seq2SeqARDiffusion(
        input_dim=len(feature_cols),
        exo_dim=len(exo_cols),
        n_turbines=n_turbines_ckpt if n_turbines_ckpt > 0 else len(indexer.groups),
        emb_dim=emb_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        timesteps=timesteps_total,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule=schedule,
        t_embed_dim=t_embed_dim,
        k_steps=k_steps,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Precompute encoder condition parts once
    enc_out, _, x_future_aug, _ = model._build_conditions(X_hist, X_future, turb_idx)

    # Select timesteps to show (evenly spaced, clamp to at least 2 and at most total)
    n_cols = max(2, min(args.timesteps, timesteps_total))
    fwd_show = np.linspace(0, timesteps_total - 1, n_cols, dtype=int).tolist()
    rev_show = np.linspace(timesteps_total - 1, 0, n_cols, dtype=int).tolist()

    # 1) Forward (noising) snapshots: apply q_sample per-hour (shape [B,1]) then stitch into full 24h
    forward_series = []  # list of [T]
    for t_idx in fwd_show:
        t_tensor = torch.full((1,), int(t_idx), dtype=torch.long, device=device)
        per_hours = []
        for h in range(pred_len):
            y_h = y_scaled[:, h:h+1, :].squeeze(-1)             # [1, 1]
            noise_h = torch.randn_like(y_h)                     # [1, 1]
            y_noisy_h = model.q_sample(y_h, t_tensor, noise_h)  # [1, 1]
            per_hours.append(y_noisy_h.unsqueeze(-1))           # [1, 1, 1]
        y_noisy = torch.cat(per_hours, dim=1)                   # [1, T, 1]
        y_noisy_np = y_noisy.cpu().numpy().reshape(-1, 1)
        try:
            y_noisy_orig = target_scaler.inverse_transform(y_noisy_np.astype(np.float64)).reshape(1, pred_len, 1).squeeze()
        except Exception:
            y_noisy_orig = y_noisy_np.reshape(1, pred_len, 1).squeeze()
        # sanitize: replace NaN/Inf and clamp to plausible power range
        y_noisy_orig = np.nan_to_num(y_noisy_orig, nan=0.0, posinf=clamp_max, neginf=clamp_min)
        y_noisy_orig = np.clip(y_noisy_orig, clamp_min, clamp_max)
        forward_series.append(y_noisy_orig)

    # 2) Reverse (denoising) snapshots: autoregressive update across all 24 hours per diffusion step
    # Start from Gaussian noise
    y_seq = torch.randn((1, pred_len, 1), device=device)
    reverse_series = []  # list of [T]
    for t_idx in range(timesteps_total - 1, -1, -1):
        t_tensor = torch.full((1,), int(t_idx), dtype=torch.long, device=device)
        # Teacher forcing prev_y: use clean previous hour (training-consistent)
        prev_clean = y0_scaled  # [1,1]
        # Update each forecast hour autoregressively at current diffusion step
        for h in range(pred_len):
            if h > 0:
                prev_clean = y_scaled[:, h-1:h, :].squeeze(1)  # [1,1] previous clean y
            exo_t = x_future_aug[:, h:h+1, :]          # [1, 1, exo_dim+emb]
            exo_t_flat = exo_t.squeeze(1)              # [1, exo_dim+emb]
            cond = torch.cat([enc_out, exo_t_flat, prev_clean], dim=1)  # [1, cond_dim]
            h_idx = torch.full((1,), h, dtype=torch.long, device=device)
            x_t = y_seq[:, h:h+1, :].squeeze(1)        # [1, 1]
            # Deterministic denoising snapshot for visualization
            y_h = model.p_sample(cond, x_t, t_tensor, add_noise=False, h_idx=h_idx)  # [1,1]
            y_seq[:, h:h+1, :] = y_h.unsqueeze(1)      # back to [1,1,1]

        if t_idx in rev_show:
            y_seq_np = y_seq.detach().cpu().numpy().reshape(-1, 1)
            try:
                y_seq_orig = target_scaler.inverse_transform(y_seq_np.astype(np.float64)).reshape(1, pred_len, 1).squeeze()  # [T]
            except Exception:
                y_seq_orig = y_seq_np.reshape(1, pred_len, 1).squeeze()
            # sanitize: replace NaN/Inf and clamp to plausible power range
            y_seq_orig = np.nan_to_num(y_seq_orig, nan=0.0, posinf=clamp_max, neginf=clamp_min)
            y_seq_orig = np.clip(y_seq_orig, clamp_min, clamp_max)
            reverse_series.append((t_idx, y_seq_orig))

    # Ensure reverse_series corresponds to columns left->right in rev_show order
    reverse_series_sorted = []
    for t in rev_show:
        for (tt, series) in reverse_series:
            if tt == t:
                reverse_series_sorted.append(series)
                break

    # Prepare plotting + save series for later use
    # Stack lists into arrays
    forward_arr = np.stack(forward_series, axis=0) if len(forward_series) > 0 else np.zeros((0, pred_len))
    reverse_arr = np.stack(reverse_series_sorted, axis=0) if len(reverse_series_sorted) > 0 else np.zeros((0, pred_len))

    # Save arrays and metadata next to the image
    base = os.path.splitext(os.path.basename(args.out))[0]
    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    fwd_path = os.path.join(out_dir, f"{base}_forward.npy")
    rev_path = os.path.join(out_dir, f"{base}_reverse.npy")
    true_path = os.path.join(out_dir, f"{base}_y_true.npy")
    meta_path = os.path.join(out_dir, f"{base}_meta.json")

    np.save(fwd_path, forward_arr)
    np.save(rev_path, reverse_arr)
    np.save(true_path, y_true.astype(np.float64))

    meta = {
        "fwd_timesteps": list(map(int, fwd_show)),
        "rev_timesteps": list(map(int, rev_show)),
        "turb_id": int(turb_id),
        "chosen_index": int(chosen_idx),
        "hist_len": int(hist_len),
        "pred_len": int(pred_len),
        "diffusion_timesteps_total": int(timesteps_total),
        "schedule": str(schedule),
        "data_path": args.data,
        "checkpoint": args.checkpoint,
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved data arrays and meta: {fwd_path}, {rev_path}, {true_path}, {meta_path}")
    except Exception as e:
        print(f"[WARN] Failed to save metadata JSON: {e}")

    # Also save CSV files for easy inspection
    fwd_csv_path = os.path.join(out_dir, f"{base}_forward.csv")
    rev_csv_path = os.path.join(out_dir, f"{base}_reverse.csv")
    true_csv_path = os.path.join(out_dir, f"{base}_y_true.csv")

    try:
        hour_cols = [f"hour_{h}" for h in range(pred_len)]

        # Forward CSV: rows are timesteps in fwd_show, columns are hour_0..hour_{T-1}
        df_fwd = pd.DataFrame(forward_arr, columns=hour_cols)
        df_fwd.insert(0, "timestep", list(map(int, fwd_show)))
        df_fwd.to_csv(fwd_csv_path, index=False)

        # Reverse CSV: rows are timesteps in rev_show (sorted to match plotting), columns are hour_0..hour_{T-1}
        df_rev = pd.DataFrame(reverse_arr, columns=hour_cols)
        df_rev.insert(0, "timestep", list(map(int, rev_show)))
        df_rev.to_csv(rev_csv_path, index=False)

        # y_true CSV: two columns hour, y_true
        df_true = pd.DataFrame({
            "hour": list(range(pred_len)),
            "y_true": y_true.astype(np.float64)
        })
        df_true.to_csv(true_csv_path, index=False)

        print(f"[INFO] Saved CSVs: {fwd_csv_path}, {rev_csv_path}, {true_csv_path}")
    except Exception as e:
        print(f"[WARN] Failed to save CSVs: {e}")

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8), squeeze=False)
    fig.suptitle(f"Diffusion Process (24h sequences) - Turbine {turb_id}", fontsize=16, fontweight="bold")

    # Unify y-limits across all panels
    y_all = [*y_true.tolist()]
    for ser in forward_series:
        y_all.extend(ser.tolist())
    for ser in reverse_series_sorted:
        y_all.extend(ser.tolist())
    y_min = np.nanmin(y_all)
    y_max = np.nanmax(y_all)
    pad = max(1e-3, 0.05 * (y_max - y_min))
    y_min -= pad
    y_max += pad

    hours = np.arange(pred_len)

    # Plot forward row (noising): true (black) + noisy (red)
    for i, t_idx in enumerate(fwd_show):
        ax = axes[0, i]
        ax.plot(hours, y_true, color="black", linewidth=2.0, label="True")
        ax.plot(hours, forward_series[i], color="red", linewidth=1.8, alpha=0.9, label=f"Noisy t={t_idx}")
        ax.set_title(f"Forward (t={t_idx})", fontsize=11)
        ax.set_ylim([y_min, y_max])
        ax.grid(alpha=0.3, linestyle="--")
        if i == 0:
            ax.set_ylabel("Power (kW)")
        ax.set_xlabel("Hour")
        if i == n_cols - 1:
            ax.legend(loc="best", fontsize=9)

    # Plot reverse row (denoising): true (black) + denoised (green)
    for i, t_idx in enumerate(rev_show):
        ax = axes[1, i]
        ax.plot(hours, y_true, color="black", linewidth=2.0, label="True")
        ax.plot(hours, reverse_series_sorted[i], color="green", linewidth=1.8, alpha=0.9, label=f"Denoised t={t_idx}")
        ax.set_title(f"Reverse (t={t_idx})", fontsize=11)
        ax.set_ylim([y_min, y_max])
        ax.grid(alpha=0.3, linestyle="--")
        if i == 0:
            ax.set_ylabel("Power (kW)")
        ax.set_xlabel("Hour")
        if i == 0:
            ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[SUCCESS] Saved diffusion visualization to: {args.out}")


if __name__ == "__main__":
    main()