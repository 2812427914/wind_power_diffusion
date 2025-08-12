# 开发与扩展说明

## 1. 代码结构

- `src/dataloader.py`：数据加载与预处理
- `src/diffusion.py`：扩散模型实现
- `src/baseline_gan.py`：GAN基线模型
- `src/baseline_vae.py`：VAE基线模型
- `src/train.py`：训练主程序
- `src/evaluate.py`：评估与指标计算
- `src/visualize.py`：可视化脚本

## 2. 扩展建议

- 新增模型：在 `src/` 下添加新模型文件，并在 `train.py`、`evaluate.py` 注册
- 新增评估指标：在 `evaluate.py` 中实现并注册
- 数据格式变更：修改 `dataloader.py` 适配新格式

## 3. 贡献与协作

- 遵循PEP8代码规范
- 推荐使用pull request协作开发
- 文档与代码同步维护