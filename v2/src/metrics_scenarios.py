#!/usr/bin/env python3
# 1) 新增脚本：读取各模型的 y_true 与 y_samples_all，计算 ES/CRPS/TCE 指标。
# 2) ES（Energy Score）：基于样本集的能量分数，适用于多步多维（向量）预测。
# 3) CRPS：逐时刻的样本近似（E|X-y| - 0.5 E|X-X'|），对所有样本与时刻求均值。
# 4) TCE（Tail Coverage Error，尾部覆盖误差）：比较下/上分位数的覆盖率与名义水平的偏差。
#    - 对每个 alpha，计算：lower: P(y_true <= q_alpha) 与 alpha 的偏差；upper: P(y_true >= q_{1-alpha}) 与 alpha 的偏差。
#    - 汇报 lower/upper 以及二者的平均。
# 5) 支持子采样场景数与限制样本数，避免 O(N*S^2*T) 过大；ES 默认精确计算（S<=200 通常可接受）。
# 6) 结果写入 JSON：v2/results/metrics_scenarios.json（可自定义）。
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="计算多场景预测的 ES/CRPS/TCE 指标")
    p.add_argument(
        "--models",
        type=str,
        default="seq2seq,seq2seq_diffusion,seq2seq_ar_diffusion,gan,vae",
        help="模型名称列表，逗号分隔。需已存在 v2/results/<model>_y_true.npy 和 <model>_y_samples_all.npy",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="v2/results",
        help="预测结果所在目录（包含 <model>_y_true.npy / _y_samples_all.npy）",
    )
    p.add_argument(
        "--alphas",
        type=str,
        default="0.05,0.1",
        help="TCE 名义覆盖率列表，逗号分隔，例如 '0.05,0.1'",
    )
    p.add_argument(
        "--subsample-scenarios",
        type=int,
        default=0,
        help="子采样场景数（>0 时从样本场景中均匀子采样至该数；0 表示不子采样）",
    )
    p.add_argument(
        "--max-n",
        type=int,
        default=0,
        help="最多使用前 N 条样本（0 表示使用全部）",
    )
    p.add_argument(
        "--es-norm",
        type=str,
        choices=["l2", "l1"],
        default="l2",
        help="ES 中使用的向量范数（l2: Euclidean，l1: Manhattan）",
    )
    p.add_argument(
        "--out",
        type=str,
        default="v2/results/metrics_scenarios.json",
        help="输出 JSON 指标文件",
    )
    return p.parse_args()


def _maybe_subsample_scenarios(y_samples: np.ndarray, k: int) -> np.ndarray:
    """
    y_samples: [N, S, T]
    均匀子采样到 k 个场景（若 k>0 且 S>k）。
    """
    if k is None or k <= 0:
        return y_samples
    N, S, T = y_samples.shape
    if S <= k:
        return y_samples
    # 均匀取索引
    idx = np.linspace(0, S - 1, num=k, dtype=int)
    return y_samples[:, idx, :]


def _maybe_limit_samples(y_true: np.ndarray, y_samples: np.ndarray, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_n is None or max_n <= 0:
        return y_true, y_samples
    n = min(max_n, y_true.shape[0])
    return y_true[:n], y_samples[:n]


def _align_horizon(y_true: np.ndarray, y_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对齐时间维 T（取二者的最小值），并 squeeze 可能的单例维度。
    y_true: [N, T] or [N, T, 1]
    y_samples: [N, S, T] or [N, S, T, 1]
    """
    if y_true.ndim == 3 and y_true.shape[-1] == 1:
        y_true = y_true.squeeze(-1)
    if y_samples.ndim == 4 and y_samples.shape[-1] == 1:
        y_samples = y_samples.squeeze(-1)
    T_true = y_true.shape[1]
    T_samp = y_samples.shape[2]
    T = min(T_true, T_samp)
    return y_true[:, :T], y_samples[:, :, :T]


def _sanitize(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def energy_score_single(y_true_vec: np.ndarray, samples_mat: np.ndarray, norm: str = "l2") -> float:
    """
    y_true_vec: [T]
    samples_mat: [S, T]
    ES ≈ 1/S Σ||X_s - y|| - 0.5 * 1/S^2 Σ||X_s - X_s'||
    """
    S, T = samples_mat.shape
    diff_y = samples_mat - y_true_vec[None, :]  # [S, T]
    if norm == "l2":
        term1 = np.sqrt((diff_y ** 2).sum(axis=1)).mean()
        # pairwise distances
        # [S, S, T] -> distances [S, S]
        diffs = samples_mat[:, None, :] - samples_mat[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
    else:
        term1 = np.abs(diff_y).sum(axis=1).mean()
        diffs = samples_mat[:, None, :] - samples_mat[None, :, :]
        dists = np.abs(diffs).sum(axis=2)
    term2 = dists.mean()  # 包含对角线 0 项，等价于 1/S^2 Σ
    return float(term1 - 0.5 * term2)


def energy_score(y_true: np.ndarray, y_samples: np.ndarray, norm: str = "l2") -> float:
    """
    y_true: [N, T]
    y_samples: [N, S, T]
    返回所有样本的 ES 平均值
    """
    y_true = _sanitize(y_true)
    y_samples = _sanitize(y_samples)
    N = y_true.shape[0]
    es_vals = []
    for i in range(N):
        es_vals.append(energy_score_single(y_true[i], y_samples[i], norm=norm))
    return float(np.mean(es_vals)) if len(es_vals) > 0 else 0.0


def crps_from_samples_1d(y: np.ndarray, samples: np.ndarray) -> float:
    """
    对单个时刻的 CRPS（1D）：(1/S) Σ|x_i - y| - (1/(2S^2)) Σ|x_i - x_j|
    y: scalar
    samples: [S]
    """
    S = samples.shape[0]
    term1 = np.mean(np.abs(samples - y))
    z = np.sort(samples)
    # E|X - X'| = (2/S^2) * sum_{i=1..S} (2i - S - 1) z_i
    i = np.arange(1, S + 1)
    e_xx = (2.0 / (S * S)) * np.sum((2 * i - S - 1) * z)
    return float(term1 - 0.5 * e_xx)


def crps_mean(y_true: np.ndarray, y_samples: np.ndarray) -> float:
    """
    y_true: [N, T]
    y_samples: [N, S, T]
    返回所有 (N,T) 的 CRPS 平均
    """
    y_true = _sanitize(y_true)
    y_samples = _sanitize(y_samples)
    N, T = y_true.shape
    S = y_samples.shape[1]
    # 向量化实现：对每个 (n, t) 计算
    # term1: mean(|x - y|) -> [N, T]
    term1 = np.mean(np.abs(y_samples - y_true[:, None, :]), axis=1)  # [N, T]
    # term2: E|X-X'| -> 使用排序加权公式
    samples_sorted = np.sort(y_samples, axis=1)  # [N, S, T]
    i = np.arange(1, S + 1).astype(np.float64)[:, None]  # [S, 1]
    weights = (2 * i - (S + 1))  # [S, 1]
    # broadcasting 到 [N, S, T]：先在 S 维相乘再求和
    weighted_sum = (samples_sorted * weights[None, :, :]).sum(axis=1)  # [N, T]
    e_xx = (2.0 / (S * S)) * weighted_sum  # [N, T]
    crps = term1 - 0.5 * e_xx
    return float(np.mean(crps))


def tce_coverage_errors(y_true: np.ndarray, y_samples: np.ndarray, alphas: List[float]) -> Dict[str, Dict[str, float]]:
    """
    计算尾部覆盖误差：
      lower(alpha): | P(y_true <= q_alpha) - alpha |
      upper(alpha): | P(y_true >= q_{1-alpha}) - alpha |
    返回字典：{ "alpha=0.05": {"lower":..., "upper":..., "avg":..., "cov_lower":..., "cov_upper":...}, ... }
    """
    y_true = _sanitize(y_true)
    y_samples = _sanitize(y_samples)
    res = {}
    # 逐 alpha 计算
    for a in alphas:
        q_lower = np.quantile(y_samples, a, axis=1)        # [N, T]
        q_upper = np.quantile(y_samples, 1.0 - a, axis=1)  # [N, T]
        cov_lower = np.mean(y_true <= q_lower)
        cov_upper = np.mean(y_true >= q_upper)
        err_lower = abs(cov_lower - a)
        err_upper = abs(cov_upper - a)
        res[f"alpha={a}"] = {
            "lower": float(err_lower),
            "upper": float(err_upper),
            "avg": float((err_lower + err_upper) / 2.0),
            "cov_lower": float(cov_lower),
            "cov_upper": float(cov_upper),
        }
    return res


def load_outputs(results_dir: str, model: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    返回 y_true [N,T], y_samples_all [N,S,T]；若缺失则返回 None
    """
    y_true_path = os.path.join(results_dir, f"{model}_y_true.npy")
    y_samples_all_path = os.path.join(results_dir, f"{model}_y_samples_all.npy")
    if not (os.path.exists(y_true_path) and os.path.exists(y_samples_all_path)):
        print(f"[WARN] 缺少文件：{y_true_path} 或 {y_samples_all_path}，跳过 {model}")
        return None
    try:
        y_true = np.load(y_true_path)
        y_samples_all = np.load(y_samples_all_path)
        return y_true, y_samples_all
    except Exception as e:
        print(f"[WARN] 加载 {model} 预测文件失败：{e}，跳过。")
        return None


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    alphas = []
    for a in args.alphas.split(","):
        try:
            val = float(a.strip())
            if 0.0 < val < 0.5:
                alphas.append(val)
        except Exception:
            continue
    if len(alphas) == 0:
        alphas = [0.05]

    results = {}
    for model in models:
        data = load_outputs(args.results_dir, model)
        if data is None:
            continue
        y_true, y_samples_all = data
        # 对齐维度并清洗
        y_true, y_samples_all = _align_horizon(y_true, y_samples_all)
        # 可选限制样本与子采样场景
        y_true, y_samples_all = _maybe_limit_samples(y_true, y_samples_all, args.max_n)
        y_samples_all = _maybe_subsample_scenarios(y_samples_all, args.subsample_scenarios)

        if y_true.size == 0 or y_samples_all.size == 0:
            print(f"[WARN] 模型 {model} 数据为空，跳过。")
            continue

        N, S, T = y_samples_all.shape
        print(f"[INFO] {model}: N={N}, S={S}, T={T}")

        # 计算指标
        es = energy_score(y_true, y_samples_all, norm=args.es_norm)
        crps = crps_mean(y_true, y_samples_all)
        tce = tce_coverage_errors(y_true, y_samples_all, alphas)

        results[model] = {
            "N": int(N),
            "S": int(S),
            "T": int(T),
            "ES_mean": float(es),
            "CRPS_mean": float(crps),
            "TCE": tce,
        }
        print(f"[RESULT] {model}: ES={es:.6f}, CRPS={crps:.6f}, TCE={json.dumps(tce)}")

    if len(results) == 0:
        print("[ERROR] 未成功计算任何模型的指标。请先运行预测脚本生成 y_true 与 y_samples_all。")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] 已保存指标到: {args.out}")


if __name__ == "__main__":
    main()