# CGS 转台强化学习调参

该项目使用分阶段 SAC，为 CGS 转台三环 PIDF 与 DOBC 控制器调节 11 个参数。

## 服务器环境

- Python 3.11
- PyTorch 2.5.1
- CUDA 12.1
- Gymnasium 1.2.x
- Stable-Baselines3 2.9

服务器首次安装：

```bash
unset PYTHONPATH
source /data/l50063953/miniconda3/etc/profile.d/conda.sh
conda activate elc-rl-server
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_server_runtime.py --device cuda
```

以后更新源码：

```bash
git pull --ff-only
```

纯 Python 源码更新无需重新安装；修改依赖或 `pyproject.toml` 后需要重新执行 editable 安装。

## 数据隔离

仓库只包含正式训练与验证所需的运行输入。训练输出、检查点、原始测量和封存最终测试数据不进入 Git。最终测试只能在三个正式种子训练完成并冻结唯一候选后执行。
