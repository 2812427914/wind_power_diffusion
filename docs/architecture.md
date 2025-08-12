# 系统架构说明

## 1. 总体架构

本项目采用模块化设计，核心分为数据处理、模型训练、评估与可视化四大部分。各部分通过统一的数据接口和配置参数解耦，便于扩展和复现。

## 2. 主要模块

- **数据加载与预处理**：`src/dataloader.py`
- **扩散模型实现**：`src/diffusion.py`
- **基线模型（GAN/VAE）**：`src/baseline_gan.py`、`src/baseline_vae.py`
- **训练主程序**：`src/train.py`
- **评估与指标**：`src/evaluate.py`
- **可视化**：`src/visualize.py`

## 3. 数据流与调用关系

```mermaid
flowchart TD
    A[数据加载与预处理] --> B[扩散模型训练]
    A --> C[基线模型训练(GAN/VAE)]
    B --> D[多场景功率生成]
    C --> E[基线场景生成]
    D --> F[模型评估与对比]
    E --> F
    F --> G[结果可视化]
    G --> H[报告与文档输出]
```

## 4. 依赖与环境

- Python 3.8+
- 依赖详见 `requirements.txt`
- 推荐使用虚拟环境或conda