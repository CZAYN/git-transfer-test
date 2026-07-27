# CGS 转台单次上传服务器流程

本发布包只需要上传一次。它包含两个相互分离的内部 ZIP：

- `training/elc_rl_server_training_v1.zip`：正式训练、验证和候选选择。
- `final_test/elc_rl_server_final_test_v1.zip`：封存最终测试；候选冻结前不得解压。

## 1. 解压发布包和训练包

```bash
unzip elc_rl_server_release_v1.zip -d elc_rl_server_release_v1
cd elc_rl_server_release_v1
mkdir -p runtime/training
unzip training/elc_rl_server_training_v1.zip -d runtime/training
cd runtime/training
```

激活已建立的服务器环境：

```bash
unset PYTHONPATH
source /data/l50063953/miniconda3/etc/profile.d/conda.sh
conda activate elc-rl-server
python -m pip install --no-deps --no-build-isolation -e .
```

## 2. 预检和正式训练

```bash
python scripts/check_server_runtime.py --device cuda
```

先运行工程检查：

```bash
python scripts/train_sac.py \
  --seed 20260801 \
  --device cuda \
  --engineering-check-steps-per-stage 2 \
  --output-dir outputs/formal_engineering_check
```

工程检查通过后，在单 GPU 上顺序运行三个正式种子：

```bash
python scripts/train_sac.py --seed 20260801 --device cuda
python scripts/train_sac.py --seed 20260802 --device cuda
python scripts/train_sac.py --seed 20260803 --device cuda
```

中断后使用相同种子恢复：

```bash
python scripts/train_sac.py --seed 20260801 --device cuda --resume
```

## 3. 选择并冻结唯一候选

三个种子全部完成后运行：

```bash
python scripts/select_final_candidate.py
sha256sum outputs/sac_training_v1/selection/final_candidate.npz \
  > outputs/sac_training_v1/selection/FINAL_CANDIDATE.sha256
chmod 444 outputs/sac_training_v1/selection/final_candidate.npz
```

从此停止训练、验证和候选选择，不得根据最终测试结果更换候选。

## 4. 此时才解压封存最终测试包

回到发布包根目录：

```bash
cd ../..
mkdir -p runtime/final_test
unzip final_test/elc_rl_server_final_test_v1.zip -d runtime/final_test
cp runtime/training/outputs/sac_training_v1/selection/final_candidate.npz \
  runtime/final_test/final_candidate.npz
cd runtime/final_test
sha256sum final_candidate.npz
```

该哈希必须与 `runtime/training/outputs/sac_training_v1/selection/FINAL_CANDIDATE.sha256`
中的值一致。

## 5. 锁定并执行唯一一次最终测试

```bash
python scripts/lock_final_candidate.py \
  final_candidate.npz \
  --output final_candidate_training_lock.json \
  --declare-training-complete

python scripts/run_final_test.py \
  final_candidate.npz \
  --candidate-lock final_candidate_training_lock.json
```

最终输出：

```text
runtime/final_test/outputs/final_test_v1/
├── final_test_report.json
└── FINAL_TEST_CONSUMED.json
```

测试结果是最终结果，不得用于重新训练或重新选择候选。
