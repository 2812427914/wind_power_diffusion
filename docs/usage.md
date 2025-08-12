# 使用说明

## 1. 环境准备

```bash
pip install -r requirements.txt
```

## 2. 数据准备

将原始 `wtbdata_245days.csv` 拷贝到 `data/` 目录下。

## 3. 训练模型

```bash
bash scripts/run_train.sh
```

## 4. 评估与可视化

```bash
bash scripts/run_eval.sh
bash scripts/run_vis.sh
```

## 5. 交互式实验

可在 `notebooks/wind_power_eda.ipynb` 进行数据探索与可视化。