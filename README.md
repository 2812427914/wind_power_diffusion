# Wind Power Probabilistic Scenario Forecasting (风电功率多场景概率预测)

本项目基于风电场SCADA数据，集成了多种**生成式模型**（Diffusion, GAN, VAE, Weibull Diffusion, 以及序列LSTM-VAE/扩散模型），用于生成未来24小时风电功率的**多场景概率预测**。支持多种主流评估指标（CRPS、ES、QS、MAE、RMSE），并提供可视化与对比分析。

---

## 目录结构

```
wind_power_diffusion/
├── data/                        # 数据集与空间分布信息
│   ├── wtbdata_245days.csv      # 原始SCADA风机数据（需自备）
│   ├── wtbdata_hourly.csv       # 整点数据（由脚本自动生成）
│   └── first_50_samples.csv     # 小样本示例
├── notebooks/                   # 交互式实验与可视化
│   └── wind_power_eda.ipynb
├── src/                        # 核心模型与主程序
│   ├── dataloader.py            # 数据加载与预处理（支持序列输入）
│   ├── diffusion.py             # MLP扩散模型
│   ├── diffusion_weibull.py     # Weibull噪声扩散模型
│   ├── baseline_gan.py          # GAN基线模型
│   ├── baseline_vae.py          # VAE基线模型
│   ├── seq_vae.py               # LSTM-VAE序列模型
│   ├── seq_diffusion.py         # LSTM-Diffusion序列模型
│   ├── train.py                 # 训练主程序
│   ├── predict.py               # 预测主程序
│   ├── evaluate.py              # 评估与指标计算
│   └── visualize.py             # 可视化脚本
├── scripts/                     # 一键运行脚本
│   ├── run_train.sh             # 一键训练
│   ├── run_predict.sh           # 一键预测
│   ├── run_eval.sh              # 一键评估
│   ├── run_vis.sh               # 一键可视化
│   ├── data_overview.py         # 数据总览分析
│   ├── extract_hourly_data.py   # 整点数据提取
│   ├── sample_data.py           # 小样本提取
│   └── export_scenarios_to_csv.py # 多场景导出
├── checkpoints/                 # 保存训练好的模型
├── results/                     # 保存预测、评估、可视化结果
├── requirements.txt             # 依赖包
└── docs/                        # 详细开发与用户文档
    ├── architecture.md
    ├── usage.md
    └── develop.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

- 将原始 `wtbdata_245days.csv` 拷贝到 `data/` 目录下。
- 运行数据预处理脚本，生成整点数据：

```bash
python scripts/extract_hourly_data.py
```

### 3. 训练模型

支持多种模型，常用命令如下：

```bash
# 训练所有基线模型
bash scripts/run_train.sh

# 训练序列VAE
python src/train.py --model seq_vae

# 训练序列Diffusion
python src/train.py --model seq_diffusion
```

### 4. 生成多场景预测

```bash
# 预测所有基线模型
bash scripts/run_predict.sh

# 预测序列VAE
python src/predict.py --model seq_vae --n_samples 100

# 预测序列Diffusion
python src/predict.py --model seq_diffusion --n_samples 100
```

### 5. 评估与可视化

```bash
# 评估所有模型
bash scripts/run_eval.sh

# 可视化所有模型对比
bash scripts/run_vis.sh
```

- 结果文件、指标、可视化图片均保存在 `results/` 目录下。

---

## 支持的模型

| 模型名            | 命令参数         | 说明                         |
|-------------------|------------------|------------------------------|
| diffusion         | --model diffusion | MLP扩散模型                  |
| gan               | --model gan       | MLP-GAN基线                  |
| vae               | --model vae       | MLP-VAE基线                  |
| weibull_diffusion | --model weibull_diffusion | Weibull噪声扩散模型  |
| seq_vae           | --model seq_vae   | LSTM-VAE序列生成模型         |
| seq_diffusion     | --model seq_diffusion | LSTM-Diffusion序列生成模型 |

---

## 主要功能

- **多场景概率预测**：每个输入可采样生成多个未来24小时功率序列，反映不确定性。
- **多模型对比**：支持MLP、GAN、VAE、Diffusion、Weibull Diffusion、LSTM-VAE、LSTM-Diffusion等。
- **多指标评估**：CRPS、ES、QS、MAE、RMSE等。
- **可视化**：多场景曲线、指标对比图自动生成。
- **脚本化流程**：一键训练、预测、评估、可视化。

---

## 评估指标

- **MAE**：平均绝对误差
- **RMSE**：均方根误差
- **CRPS**：连续秩概率评分
- **ES**：能量分数
- **QS**：分位数评分

---

## 结果文件说明

- `results/y_pred_<model>.npy`：各模型的均值预测
- `results/y_samples_<model>.npy`：各模型的多场景采样结果
- `results/metrics.json`：所有模型的评估指标
- `results/<model>_scenarios.png`：多场景可视化
- `results/metrics_comparison.png`：指标对比图

---

## 进阶用法

- **采样数/批量大小可调**：`--n_samples`、`--batch_size` 参数
- **支持MPS/CPU/GPU**：自动检测设备，或用 `--device` 指定
- **自定义特征/序列长度**：可在 `dataloader.py` 和主程序中调整

---

## 参考与致谢

- [扩散模型原理](https://arxiv.org/abs/2006.11239)
- [风电功率预测综述](https://en.wikipedia.org/wiki/Wind_power_forecasting)
- [PyTorch](https://pytorch.org/)

---

## 联系与贡献

- 欢迎PR、Issue、建议与合作！
- 详细开发说明见 `docs/develop.md`，架构说明见 `docs/architecture.md`。