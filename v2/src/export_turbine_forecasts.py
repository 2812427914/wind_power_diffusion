import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="导出指定风机在不同模型下的 0-23 点预测（均值与真实）为 CSV")
    p.add_argument(
        "--models",
        type=str,
        default="seq2seq,seq2seq_diffusion,seq2seq_ar_diffusion,gan,vae",
        help="模型名称列表，逗号分隔。可选：seq2seq, seq2seq_diffusion, seq2seq_ar_diffusion, gan, vae",
    )
    p.add_argument(
        "--turb-ids",
        type=str,
        default=None,
        help="多个风机ID，逗号分隔，例如 '90,91,105'。若未提供，则使用 --turb-id。",
    )
    p.add_argument(
        "--turb-id",
        type=str,
        default=None,
        help="单个风机ID（与训练数据里的 TurbID 一致），与 --turb-ids 二选一。",
    )
    p.add_argument(
        "--index",
        type=int,
        default=-1,
        help="当同一风机存在多个样本时，指定导出其中一个样本的相对索引（>=0）。默认 -1 导出该风机的所有样本。",
    )
    p.add_argument(
        "--out",
        type=str,
        default="v2/results/turbine_forecasts.csv",
        help="输出 CSV 文件路径",
    )
    return p.parse_args()


def _load_model_outputs(model: str):
    """
    加载某个模型的预测输出文件（由 v2/src/predict.py 生成）。
    返回 (y_true [N,T], y_pred_mean [N,T], turb_ids [N], y_samples_all [N,S,T] or None)
    若文件缺失（不含场景文件）则返回 None。
    """
    prefix = f"v2/results/{model}"
    y_true_path = f"{prefix}_y_true.npy"
    y_pred_mean_path = f"{prefix}_y_pred_mean.npy"
    turb_ids_path = f"{prefix}_turb_ids.npy"
    y_samples_all_path = f"{prefix}_y_samples_all.npy"

    if not (os.path.exists(y_true_path) and os.path.exists(y_pred_mean_path) and os.path.exists(turb_ids_path)):
        print(f"[WARN] 模型 {model} 的输出文件缺失，跳过。需要：{y_true_path}, {y_pred_mean_path}, {turb_ids_path}")
        return None

    try:
        y_true = np.load(y_true_path)            # [N, T]
        y_pred_mean = np.load(y_pred_mean_path)  # [N, T]
        turb_ids = np.load(turb_ids_path)        # [N]
        y_samples_all = None
        if os.path.exists(y_samples_all_path):
            try:
                y_samples_all = np.load(y_samples_all_path)  # [N, S, T]
            except Exception as e:
                print(f"[WARN] 读取场景文件失败 {y_samples_all_path}: {e}. 将不导出场景。")
                y_samples_all = None
        return y_true, y_pred_mean, turb_ids, y_samples_all
    except Exception as e:
        print(f"[WARN] 读取模型 {model} 的输出文件失败：{e}. 跳过该模型。")
        return None


def main():
    args = parse_args()
    models: List[str] = [m.strip() for m in args.models.split(",") if m.strip()]

    # 解析风机ID列表
    turb_ids_list: List[int] = []
    if args.turb_ids:
        try:
            turb_ids_list = [int(t.strip()) for t in args.turb_ids.split(",") if t.strip() != ""]
        except Exception:
            print(f"[ERROR] 无法解析 --turb-ids='{args.turb_ids}'，请使用逗号分隔的整数列表。")
            sys.exit(2)
    elif args.turb_id is not None:
        try:
            turb_ids_list = [int(args.turb_id)]
        except Exception:
            print(f"[ERROR] 无法解析 --turb-id='{args.turb_id}'，请提供整数。")
            sys.exit(2)
    else:
        print("[ERROR] 必须指定 --turb-ids 或 --turb-id 之一。")
        sys.exit(2)

    export_all_samples = args.index < 0

    rows = []
    pairs_tried = 0
    matched_pairs = 0

    for model in models:
        data = _load_model_outputs(model)
        if data is None:
            continue

        y_true, y_pred_mean, turb_ids, y_samples_all = data
        if y_true.ndim != 2 or y_pred_mean.ndim != 2:
            print(f"[WARN] 模型 {model} 的数组维度异常：y_true.shape={y_true.shape}, y_pred_mean.shape={y_pred_mean.shape}。跳过。")
            continue

        # 尝试将风机ID数组转为整数，增强兼容性
        try:
            turb_ids_arr = turb_ids.astype(int)
        except Exception:
            turb_ids_arr = turb_ids

        # 场景数（若存在）
        scenarios = 0
        if y_samples_all is not None and getattr(y_samples_all, "ndim", 0) == 3:
            scenarios = y_samples_all.shape[1]

        for tid in turb_ids_list:
            pairs_tried += 1
            indices = np.where(turb_ids_arr == tid)[0]
            if len(indices) == 0:
                print(f"[INFO] 模型 {model} 中未找到风机 {tid} 的样本。")
                continue

            matched_pairs += 1
            print(f"[INFO] 模型 {model} 找到风机 {tid} 的样本数：{len(indices)}")

            # 只导出指定样本或全部样本；--index 是相对该风机在该模型内的索引
            if export_all_samples:
                target_indices = list(indices)
            else:
                if args.index < len(indices):
                    target_indices = [indices[args.index]]
                else:
                    print(f"[WARN] --index={args.index} 超过风机 {tid} 在模型 {model} 的样本数 {len(indices)}。跳过该风机。")
                    continue

            T = y_true.shape[1]
            horizon = min(24, T)  # 只导出 0-23 小时
            for idx in target_indices:
                for h in range(horizon):
                    row = {
                        "model": model,
                        "turb_id": tid,
                        "sample_index": int(idx),
                        "hour": h,
                        "y_true": float(y_true[idx, h]),
                        "y_pred_mean": float(y_pred_mean[idx, h]),
                    }
                    # 附加场景预测（若存在）
                    if scenarios > 0 and h < y_samples_all.shape[2]:
                        for s in range(scenarios):
                            try:
                                row[f"y_pred_s{s}"] = float(y_samples_all[idx, s, h])
                            except Exception:
                                continue
                    rows.append(row)

    if len(rows) == 0:
        print(f"[ERROR] 未收集到任何数据。请确认已运行预测并生成 v2/results/<model>_y_true.npy 等文件，或检查 --models / --turb-ids / --turb-id 参数。")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(rows)
    try:
        df.sort_values(by=["model", "turb_id", "sample_index", "hour"], inplace=True)
    except Exception:
        pass
    df.to_csv(args.out, index=False)
    print(f"[SUCCESS] 已保存 CSV：{args.out}")
    print(f"统计：尝试(model,id)对数={pairs_tried}, 有匹配对数={matched_pairs}, 导出行数={len(rows)}")


if __name__ == "__main__":
    main()