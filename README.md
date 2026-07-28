# CGS 转台强化学习调参

本项目使用分阶段 SAC，为 CGS 转台三环 PIDF 与 DOBC 控制器调节 11 个参数。当前实现以物理模型为训练环境，奖励同时包含频域和时域指标，并保留独立的训练、验证与封存最终测试数据。

仓库只维护一个当前版本。配置、数据和输出路径不使用 `v1`、`v2` 等版本后缀；历史状态由 Git 管理。

## 性能结构

- 时域物理仿真由 Numba 编译为 `float64` 数值内核。
- 每个随机种子默认使用 4 个独立子进程环境。
- 默认同时启动 3 个随机种子，共使用约 12 个 CPU 核心。
- SAC 的梯度更新数随环境数等比例增加，保持每条 transition 的更新比不变。
- checkpoint 保存模型、Replay Buffer、全局随机状态和每个环境的完整状态，可精确续训。

并行化只改变采样和执行方式，不改变控制器结构、物理方程、奖励定义、训练阶段、训练步数或最终验收标准。

## 环境

服务器环境：

- Python 3.11
- PyTorch 2.5.1 + CUDA 12.1
- Stable-Baselines3 2.9.0
- Gymnasium 1.2.3
- Numba 0.66.0

```bash
unset PYTHONPATH
source /data/l50063953/miniconda3/etc/profile.d/conda.sh
conda env create -f environment-server.yml
conda activate elc-rl-server
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_server_runtime.py --device cuda
```

## 本机验证

```bash
python -m pytest -q
python scripts/check_tuning_env.py --quick
python scripts/benchmark_parallel_env.py --n-envs 4 --steps-per-env 64
python scripts/train_sac.py \
  --seed 20260801 \
  --device cpu \
  --n-envs 4 \
  --engineering-check-steps-per-stage 4 \
  --output-dir outputs/engineering_check
```

工程检查输出不能参与正式候选选择。

同时检查三个种子的并行启动器：

```bash
python -u scripts/train_all_seeds.py \
  --device cpu \
  --n-envs 4 \
  --parallel-seeds 3 \
  --engineering-check-steps-per-stage 4
```

该命令默认写入 `outputs/sac_training_engineering_check/`，不会混入正式训练目录。

## 正式训练

按配置同时启动全部种子：

```bash
python -u scripts/train_all_seeds.py --device cuda
```

默认并行度来自 `config/sac_training.json`：

```text
3 个并行种子 × 每种子 4 个环境 = 12 个环境进程
```

需要降低服务器负载时可以覆盖并行度：

```bash
python -u scripts/train_all_seeds.py \
  --device cuda \
  --parallel-seeds 2 \
  --n-envs 2
```

单独运行一个种子：

```bash
python -u scripts/train_sac.py --seed 20260801 --device cuda --n-envs 4
```

输出位置：

```text
outputs/sac_training/seed_<seed>/
logs/sac_training/seed_<seed>.log
logs/sac_training/launcher_state.json
```

## 中断与续训

正常终止信号会触发可恢复 checkpoint。恢复时必须使用相同源码、配置、训练数据和环境数：

```bash
python -u scripts/train_all_seeds.py --device cuda --resume
```

如果输入指纹或环境数变化，程序会拒绝续训；此时应开启全新的实验目录。

## 候选与最终测试

三个正式种子全部完成后：

```bash
python scripts/select_final_candidate.py
```

训练和候选选择只使用 40 个训练模型与 16 个验证模型。唯一候选锁定后，才允许使用隔离的 24 个封存最终测试模型。详细服务器操作见 `SERVER_TRAINING.md` 和 `SERVER_RELEASE.md`。
