# 风电功率扩散模型项目（Wind Power Diffusion Model）

本项目基于风电场SCADA系统采集的风机运行数据，构建并训练扩散模型（Diffusion Model），以生成多场景、带概率的风电功率输出预测。支持多种主流评估指标（CRPS、ES、QS、MAE、RMSE），并以GAN、VAE为基线模型进行对比。

## 目录结构

```
wind_power_diffusion/
├── data/                        # 数据集与空间分布信息
│   └── wtbdata_245days.csv      # SCADA风机数据（请将原始数据拷贝至此）
├── notebooks/                   # 交互式实验与可视化
│   └── wind_power_eda.ipynb     # 数据探索与可视化notebook
├── src/
│   ├── dataloader.py            # 数据加载与预处理
│   ├── diffusion.py             # 扩散模型实现
│   ├── baseline_gan.py          # GAN基线模型
│   ├── baseline_vae.py          # VAE基线模型
│   ├── train.py                 # 训练主程序
│   ├── predict.py               # 预测主程序
│   ├── evaluate.py              # 评估与指标计算
│   └── visualize.py             # 可视化脚本
├── scripts/                     # 一键运行脚本
│   ├── run_train.sh             # 一键训练脚本
│   ├── run_predict.sh           # 一键预测脚本
│   ├── run_eval.sh              # 一键评估脚本
│   └── run_vis.sh               # 一键可视化脚本
├── checkpoints/                 # 保存训练好的模型
├── results/                     # 保存预测、评估结果
├── requirements.txt             # 依赖包
└── docs/                        # 详细开发与用户文档
    ├── architecture.md          # 系统架构说明
    ├── usage.md                 # 使用说明
    └── develop.md               # 开发与扩展说明
```

## 快速开始

1. 安装依赖
    ```bash
    pip install -r requirements.txt
    ```

2. 数据准备
   将原始 `wtbdata_245days.csv` 拷贝到 `data/` 目录下。

3. 训练模型
    ```bash
    bash scripts/run_train.sh
    ```

4. 生成预测
    ```bash
    bash scripts/run_predict.sh
    ```

5. 评估与可视化
    ```bash
    bash scripts/run_eval.sh
    bash scripts/run_vis.sh
    ```

## 主要功能

- SCADA风机数据加载与预处理
- 扩散模型训练与推理
- GAN/VAE基线模型对比
- 多指标评估与可视化

## 评估指标

- CRPS、ES、QS、MAE、RMSE

## 参考

- [扩散模型原理](https://arxiv.org/abs/2006.11239)
- [风电功率预测相关论文](https://en.wikipedia.org/wiki/Wind_power_forecasting)