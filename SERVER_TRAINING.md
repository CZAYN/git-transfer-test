# CGS 转台正式 SAC 服务器训练

本训练包只用于基于 `physics_v1` 的正式长程训练。它不包含封存测试数据，也不使用冒烟训练产生的模型或输出。

## 1. 解压和建立环境

以下示例使用 Linux Bash；所有项目路径均为相对路径，核心程序同时兼容 Windows 和 Linux Python。

```bash
unzip elc_rl_server_training_v1.zip -d elc_rl_server_training_v1
cd elc_rl_server_training_v1
unset PYTHONPATH
source /data/l50063953/miniconda3/etc/profile.d/conda.sh
conda env create -f environment-server.yml
conda activate elc-rl-server
```

当前服务器已验证的环境组合为 Python 3.11、PyTorch 2.5.1、CUDA 12.1、
Stable-Baselines3 2.9.0、Gymnasium 1.2.3 和 TensorBoard 2.20.0。
`environment-server.yml` 通过 Conda 通道安装这些依赖，不访问被公司代理拦截的
PyPI 或 `download.pytorch.org`。环境建立后只需安装项目本身：

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

## 2. 服务器预检

```bash
python scripts/check_server_runtime.py --device cuda
```

预检必须确认 CUDA 可用、训练输入哈希有效、磁盘空间充足，并且环境能够完成一次安全交互。

## 3. 启动三个独立随机种子

```bash
python scripts/train_sac.py --seed 20260801 --device cuda
python scripts/train_sac.py --seed 20260802 --device cuda
python scripts/train_sac.py --seed 20260803 --device cuda
```

单 GPU 服务器应顺序运行；多 GPU 或调度器环境可将每个种子作为独立任务，并通过 `--device cuda:0` 等参数指定设备。

每个种子的默认输出位于：

```text
outputs/sac_training_v1/seed_<seed>/
```

## 4. 中断后续训

程序接收到正常的终止信号后，会在当前环境步结束时保存模型、Replay Buffer、训练状态和随机状态。

```bash
python scripts/train_sac.py --seed 20260801 --device cuda --resume
```

续训时配置、源码和训练数据哈希必须与原运行完全一致。

## 5. TensorBoard

```bash
tensorboard --logdir outputs/sac_training_v1
```

## 6. 选择唯一候选

三个种子全部完成后运行：

```bash
python scripts/select_final_candidate.py
```

输出位于：

```text
outputs/sac_training_v1/selection/
├─ candidate_leaderboard.json
├─ final_candidate.npz
└─ final_candidate_audit.json
```

候选选择只使用40个训练电机和16个验证电机。将整个 `selection` 目录以及三个种子的 `seed_summary.json` 下载回本机，再进行候选锁定和一次性封存测试。

## 7. 工程检查模式

工程检查只验证正式训练程序，不产生可参与候选选择的参数：

```bash
python scripts/train_sac.py \
  --seed 20260801 \
  --device cuda \
  --engineering-check-steps-per-stage 2 \
  --output-dir outputs/formal_engineering_check
```

正式训练不得添加这个参数。
