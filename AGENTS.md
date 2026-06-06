# 项目环境规范

- **Conda**: 必须在 `3DUS` 环境中运行。未激活时入口脚本自动报错退出。
- **CUDA**: 训练默认使用 `cuda:0`。CUDA 不可用时报错，不 fallback 到 CPU。
- **入口脚本**: `denoising-fluorescence/denoising/train_n2n.py`、`train_dncnn.py`、`benchmark.py`；`denoising-SVD/train_n2n_complex.py`、`smoke_test.py`。
