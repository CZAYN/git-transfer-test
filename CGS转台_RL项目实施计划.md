# OSN600 9# CGS 转台三环 PID + DOBC 强化学习调参项目

## 0. Codex 执行要求

请严格按本文档分阶段实现，不要跳过对象模型验证直接训练强化学习。

核心原则：

1. 原始 Excel/NPZ 提供的是**被控对象频响、辨识模型和部分基准参数**。
2. Excel 中每个频率点不是独立训练样本。
3. 不得伪造位置环 PID、DOBC 参数、reward 或最优标签。
4. 真正的强化学习 transition：
   \[
   (s_t,a_t,r_t,s_{t+1},done)
   \]
   必须由仿真环境在调参过程中生成。
5. 第一版先针对该台 OSN600 9# CGS 转台完成自动调参，不宣称对任意电机或任意转台泛化。
6. 所有假设、默认值和未从 Excel 得到的参数必须写入配置文件并在日志中明确标注。
7. 先完成确定性仿真、频响复现和传统基准控制器，再接入 SAC/TD3。
8. 不要把 `CGS转台_RL训练种子数据.npz` 当成现成的离线强化学习数据集。

---

# 1. 项目目标

输入该转台的三环开环频响数据：

- 电流环频响；
- 速度环频响；
- 位置环频响。

输出 11 个控制参数，顺序必须固定为：

```text
0  kppos
1  kipos
2  kdpos
3  kpspeed
4  kispeed
5  kdspeed
6  kgspeed
7  tauspeed
8  kpcurr
9  kicurr
10 kdcurr
```

其中：

- `kppos, kipos, kdpos`：位置环 PID；
- `kpspeed, kispeed, kdspeed`：速度环 PID；
- `kgspeed, tauspeed`：速度环 DOBC 参数；
- `kpcurr, kicurr, kdcurr`：电流环 PID。

最终需要实现：

```text
三环频响
    ↓
对象模型/环境上下文
    ↓
强化学习迭代调参
    ↓
输出最终 11 个参数
```

---

# 2. 已提供数据文件

将以下文件放入项目的 `data/` 目录：

```text
data/
├── CGS转台_RL训练种子数据.xlsx
└── CGS转台_RL训练种子数据.npz
```

## 2.1 NPZ 中的主要数组

```python
data["current_frequency_hz"]          # shape: (18,)
data["current_response_db_deg"]       # shape: (18, 2)

data["speed_amplitudes_mA"]           # shape: (6,)
data["speed_frequency_hz"]            # shape: (6, 15)
data["speed_response_db_deg"]         # shape: (6, 15, 2)

data["position_frequency_hz"]         # shape: (15,)
data["position_response_db_deg"]      # shape: (15, 2)

data["input_vector_v1"]               # shape: (96,)
data["input_vector_v1_scaled"]        # shape: (96,)

data["identified_model_ids"]          # shape: (5,)
data["identified_model_matrix"]       # shape: (5, 9)

data["output_names"]                  # shape: (11,)
data["baseline_analog_values"]        # shape: (11,)
data["metadata_json"]
```

## 2.2 频响数组最后一维含义

```text
[..., 0] = magnitude_db
[..., 1] = phase_deg
```

## 2.3 `identified_model_matrix` 的列顺序

```text
0 a1
1 a0
2 b0
3 tau_us
4 dc_gain
5 time_constant_s
6 mse
7 rms_mag_error_db
8 rms_phase_error_deg
```

辨识模型统一采用：

\[
G(s)=\frac{b_0}{a_1s+a_0}e^{-\tau s}
\]

模型 ID 包括：

```text
current_500mV
speed_100mA
speed_250mA
speed_500mA
speed_700mA
```

---

# 3. 已确认的数据事实

## 3.1 电流环

- 有效频响点：18；
- 频率范围约 52.63–3494.74 Hz；
- 使用 Excel 的 `After Fix data`；
- 原始区是同一组数据，但高频点顺序错乱，不能重复计入。

辨识模型参数来自 NPZ 中 `current_500mV`。

Excel 给出的模拟域基准参数：

```text
Kp_current = 11.85013761011272
Ki_current = 13452.616563995738
Kd_current = 0
```

有效参考指标：

```text
Gain margin       ≈ 9.9733 dB
Phase margin      ≈ 70 deg
3-dB bandwidth    ≈ 1525.16 Hz
Settling time     ≈ 0.004 s
```

## 3.2 速度环

共有六种激励：

```text
10, 50, 100, 250, 500, 700 mA
```

每组 15 个频点。

角色划分：

```text
500 mA        V1 标称对象
100/250/700   域随机化对象
10/50         鲁棒性测试，不作标称对象
```

只有 100/250/500/700 mA 有辨识模型。

Excel 给出的模拟域基准参数：

```text
Kp_speed = 5.743161339098969
Ki_speed = 636.9940387304257
Kd_speed = 0.0015641745297378033
```

有效参考指标：

```text
Gain margin       ≈ 15.7332 dB
Phase margin      ≈ 65 deg
3-dB bandwidth    ≈ 102.371 Hz
```

以下值无效，禁止作为标签或 reward：

```text
RiseTime = 99999
SettlingTime = 99999
Overshoot = 999.99
```

这些值由数值溢出产生。

## 3.3 位置环

- 有效频响点：15；
- 频率范围约 10.42–260.42 Hz；
- Excel 未提供位置环 PID 参数；
- Excel 未提供 DOBC 的 `KG` 和 `Tm`。

---

# 4. 重要建模解释

## 4.1 Excel 频响是静态对象上下文

输入的 96 维频响描述这台转台的对象特性。在一次 episode 内，它通常保持不变。

不要错误地认为每次 PID 调整后，原始开环对象频响都会改变。

强化学习的动态状态变化主要来自：

- 当前 11 个控制参数；
- 当前闭环性能指标；
- 当前扰动/工况；
- 是否稳定、是否越限。

因此推荐状态为：

\[
s_t=[
x_{\mathrm{freq}},
z_{\theta,t},
m_t,
c_t
]
\]

其中：

- \(x_{\mathrm{freq}}\)：96 维固定对象频响上下文；
- \(z_{\theta,t}\)：11 维当前控制参数的归一化表示；
- \(m_t\)：当前闭环性能指标；
- \(c_t\)：当前工况参数，可选。

## 4.2 96 维输入的顺序

```text
电流环：18 × [magnitude_db, phase_deg] = 36
速度环：500mA 15 × [magnitude_db, phase_deg] = 30
位置环：15 × [magnitude_db, phase_deg] = 30

总计：96
```

直接优先使用：

```python
data["input_vector_v1_scaled"]
```

原始向量也必须保留用于绘图和解释。

## 4.3 三环连接关系

第一版采用：

```text
位置参考
  ↓
位置 PID
  ↓ 速度参考
速度 PID + DOBC 补偿
  ↓ 电流参考
电流 PID
  ↓ 驱动输入
电流对象
  ↓ 实际电流
机械/速度对象
  ↓ 实际速度
积分
  ↓ 实际位置
```

初步假设：

- 电流辨识模型：驱动输入到电流；
- 速度辨识模型：电流到速度；
- 位置由速度积分得到。

必须通过实测频响验证该假设，防止重复计算电流环动态。

位置开环实测数据优先作为：

```text
speed_model × 1/s
```

的验证数据。

若误差明显，再增加可替换的位置传感器/滤波模型，不能直接硬凑参数。

---

# 5. 项目目录结构

请创建：

```text
cgs_rl_tuner/
├── README.md
├── pyproject.toml
├── requirements.txt
├── data/
│   ├── CGS转台_RL训练种子数据.xlsx
│   └── CGS转台_RL训练种子数据.npz
├── configs/
│   ├── plant.yaml
│   ├── controller.yaml
│   ├── reward.yaml
│   └── training.yaml
├── src/
│   └── cgs_rl/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── transfer_models.py
│       ├── delays.py
│       ├── controllers.py
│       ├── dobc.py
│       ├── simulator.py
│       ├── metrics.py
│       ├── parameterization.py
│       ├── env.py
│       ├── baseline_design.py
│       ├── train.py
│       ├── evaluate.py
│       ├── export_params.py
│       └── plotting.py
├── scripts/
│   ├── inspect_data.py
│   ├── validate_frequency_response.py
│   ├── run_baseline.py
│   ├── train_sac.py
│   └── evaluate_policy.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_transfer_models.py
│   ├── test_controllers.py
│   ├── test_metrics.py
│   ├── test_env.py
│   └── test_reproducibility.py
└── outputs/
    ├── figures/
    ├── checkpoints/
    ├── logs/
    └── exported_parameters/
```

---

# 6. 第一阶段：数据加载与校验

实现 `data_loader.py`。

要求：

1. 加载 NPZ；
2. 校验所有必需 key；
3. 校验 shape；
4. 检查 NaN/Inf；
5. 解码 `metadata_json`；
6. 返回带类型注解的 dataclass；
7. 明确 `baseline_analog_values` 中位置 PID 和 DOBC 是 NaN；
8. 禁止自动用 0 替换缺失的最优参数。

应提供：

```python
@dataclass(frozen=True)
class CGSSeedData:
    current_frequency_hz: np.ndarray
    current_response_db_deg: np.ndarray
    speed_amplitudes_mA: np.ndarray
    speed_frequency_hz: np.ndarray
    speed_response_db_deg: np.ndarray
    position_frequency_hz: np.ndarray
    position_response_db_deg: np.ndarray
    input_vector_v1: np.ndarray
    input_vector_v1_scaled: np.ndarray
    identified_model_ids: list[str]
    identified_model_matrix: np.ndarray
    output_names: list[str]
    baseline_analog_values: np.ndarray
    metadata: dict
```

`scripts/inspect_data.py` 输出：

- shape；
- 频率范围；
- 各模型参数；
- 缺失参数；
- 六组速度曲线角色；
- 图表。

---

# 7. 第二阶段：传递函数与延迟模型

实现：

```python
class FirstOrderDelayPlant
```

表示：

\[
G(s)=\frac{b_0}{a_1s+a_0}e^{-\tau s}
\]

需要支持：

- 连续频率响应；
- 离散化；
- reset；
- 单步更新；
- 纯延迟缓冲；
- 可选分数延迟；
- 批量 Bode 计算。

第一版允许将延迟换算为最近整数采样点，但必须：

- 在日志中输出近似误差；
- 保留后续分数延迟接口；
- 单元测试延迟缓冲正确性。

禁止把 Hz 直接当作 rad/s：

\[
\omega=2\pi f
\]

---

# 8. 第三阶段：频响复现验证

实现：

```text
scripts/validate_frequency_response.py
```

验证顺序：

## 8.1 电流模型

使用 `current_500mV` 参数，计算模型在实测 18 个频率点上的：

- dB；
- phase degree。

与实测比较：

- RMS magnitude error；
- RMS phase error；
- 最大误差。

结果应接近 NPZ 中保存的拟合误差。

## 8.2 速度模型

分别验证：

```text
speed_100mA
speed_250mA
speed_500mA
speed_700mA
```

禁止用一个模型强行拟合六条曲线。

## 8.3 位置关系验证

比较：

\[
G_{\mathrm{position,pred}}(s)=\frac{G_{\mathrm{speed}}(s)}{s}
\]

与实测位置开环频响。

需要生成：

```text
outputs/figures/current_fit.png
outputs/figures/speed_fit_100mA.png
outputs/figures/speed_fit_250mA.png
outputs/figures/speed_fit_500mA.png
outputs/figures/speed_fit_700mA.png
outputs/figures/position_validation.png
```

若位置误差过大：

- 打印警告；
- 不要静默继续；
- 在配置中启用可替换的位置修正模型。

**只有频响验证通过，才进入强化学习阶段。**

---

# 9. 第四阶段：三环控制器

## 9.1 PID

实现离散 PID，至少支持：

- P、PI、PID；
- 输出限幅；
- anti-windup；
- 积分状态 reset；
- 导数低通滤波；
- 多采样率更新；
- 参数运行时修改。

控制器接口：

```python
class PIDController:
    def reset(self) -> None: ...
    def set_gains(self, kp: float, ki: float, kd: float) -> None: ...
    def step(self, error: float, dt: float) -> float: ...
```

## 9.2 多环采样率

Excel 中已知或可推断：

```text
current loop dt ≈ 25 us
speed loop dt   ≈ 200 us
```

位置环采样周期未在 Excel 中明确给出。

因此：

```yaml
position_loop_dt_s: 0.001
```

只能作为 V1 可配置假设，日志必须显示：

```text
ASSUMPTION: position_loop_dt_s is not measured from Excel.
```

第一版基础仿真步长取电流环步长：

```text
base_dt = 25 us
```

更新频率：

```text
current PID  每个 base step 更新
speed PID    每 8 个 base step 更新
position PID 按 position_loop_dt_s 更新
```

必须检查各周期是否为 `base_dt` 的整数倍。

---

# 10. 第五阶段：DOBC

第一版实现可替换的速度环扰动观测器。

名义速度对象：

\[
G_n(s)=\frac{b_0}{a_1s+a_0}
\]

滤波器：

\[
Q(s)=\frac{K_G}{T_ms+1}
\]

建议采用标准逆模型残差形式：

\[
\hat d(s)=Q(s)\left[G_n^{-1}(s)\omega(s)-i(s)\right]
\]

速度环输出：

\[
i_{\mathrm{ref}}=i_{\mathrm{PID}}-i_{\mathrm{DOB}}
\]

符号必须通过负载阶跃测试验证。

要求：

1. `dobc.py` 独立于 PID；
2. `KG=0` 时 DOBC 完全关闭；
3. `Tm>0`；
4. 对速度噪声使用滤波导数；
5. 避免不适当的理想微分；
6. 提供正负扰动测试；
7. 提供无扰动时不恶化稳态性能的测试。

由于原文件未给出完整 DOBC 结构，该实现必须标记为：

```text
V1 assumption: first-order Q-filter DOBC.
```

不得将它表述为原设备唯一正确结构。

---

# 11. 第六阶段：确定性三环仿真

实现 `simulator.py`。

每个仿真 episode 至少包括：

1. 位置阶跃指令；
2. 速度或位置跟踪阶段；
3. 指定时刻施加负载扰动；
4. 记录：
   - position；
   - speed；
   - current；
   - three references；
   - controller outputs；
   - disturbance estimate；
   - saturation flags。

仿真必须支持：

- 固定随机种子；
- 选择速度对象模型；
- 参数扰动；
- 编码器噪声；
- 负载扰动；
- 电流、速度、位置限幅；
- 不稳定提前终止。

---

# 12. 第七阶段：闭环指标

实现 `metrics.py`。

至少计算：

```text
position_rise_time
position_settling_time
position_overshoot
position_steady_state_error
position_IAE

speed_peak_deviation_after_disturbance
speed_recovery_time
speed_IAE

current_peak
current_RMS
control_effort
control_rate_penalty

current_bandwidth
speed_bandwidth
position_bandwidth

current_phase_margin
speed_phase_margin
position_phase_margin

unstable
saturated
```

注意：

- 带宽和稳定裕度来自闭环/环路传递函数；
- 上升时间等来自时域响应；
- 速度 Excel 中的 99999 指标禁止使用；
- 指标计算失败必须返回明确 failure 状态，不能填 99999。

带宽层级约束：

```text
current bandwidth > speed bandwidth > position bandwidth
```

违反时加惩罚。

---

# 13. 第八阶段：参数化

11 个物理参数尺度差异很大，禁止直接让 Actor 输出原始绝对值。

维护归一化参数：

\[
z\in[-1,1]^{11}
\]

映射原则：

- 严格为正且跨数量级的参数用对数映射；
- 可为 0 的微分参数使用线性映射；
- `Tm` 使用对数映射；
- 所有物理参数有明确上下界；
- 上下界写入 `controller.yaml`。

推荐动作：

\[
a_t=\Delta z_t,\quad a_t\in[-1,1]^{11}
\]

更新：

\[
z_{t+1}=\operatorname{clip}(z_t+\alpha\odot a_t,-1,1)
\]

其中 `alpha` 为各参数最大单步调整比例。

基准值：

```text
speed/current PID 使用 Excel 模拟域参数
```

位置 PID 与 DOBC 初始值：

- 不从 Excel 伪造；
- 由 `baseline_design.py` 基于稳定约束搜索；
- 初始可令 `KG=0` 表示关闭 DOBC；
- `Tm` 使用配置的保守正值；
- 所有来源写入导出结果。

---

# 14. 第九阶段：强化学习环境

实现 Gymnasium 风格环境：

```python
class CGSTuningEnv(gym.Env):
    ...
```

## 14.1 Observation

推荐 V1：

```text
96 维 频响上下文
11 维 当前参数 z
10~20 维 当前性能指标
可选 5~10 维 工况信息
```

频响上下文在同一 episode 内可以保持不变。

不要把 `next_state_frequency_response` 强制理解为每步都变化；若对象不变，它可以复制当前频响上下文。

## 14.2 Action

```text
11 维连续参数增量
```

## 14.3 Reward

配置化加权代价：

\[
r_t=-J_t
\]

至少包含：

```text
position tracking error
rise time
settling time
overshoot
steady-state error
disturbance peak
disturbance recovery time
current peak
control effort
parameter step size
bandwidth hierarchy penalty
low phase-margin penalty
instability penalty
saturation penalty
```

所有指标先归一化，再加权。

禁止直接将不同量纲原始数值相加。

## 14.4 Done

以下情况终止：

```text
达到性能目标
系统不稳定
电流/控制量严重越限
连续多步无改进
达到最大调参步数
出现 NaN/Inf
```

## 14.5 Reset

每个 episode：

1. 选择 plant variant；
2. 选择或扰动辨识模型；
3. 在安全基准附近随机初始化控制参数；
4. 生成参考信号和扰动；
5. 计算初始指标；
6. 返回 observation。

---

# 15. 域随机化

训练集使用：

```text
speed_100mA
speed_250mA
speed_500mA
```

建议将 `speed_700mA` 留作主要测试对象。

可在辨识模型范围内随机化：

```text
a0
b0
tau
R
L
load disturbance
friction
encoder noise
reference amplitude
```

不得随意设置超出数据支持范围的大扰动。

第一版范围应参考四个辨识模型的最小值与最大值。

数据划分必须按对象/episode，不得把同一个 episode 的 step 随机拆到训练和测试两边。

---

# 16. 算法

第一版使用 SAC。

原因：

- 11 维连续动作；
- 参数搜索需要探索；
- 可直接接 Gymnasium 环境。

同时保留 TD3 接口作为对比。

训练代码要求：

- 固定随机种子；
- 保存配置副本；
- 保存最佳 checkpoint；
- TensorBoard 或 CSV 日志；
- 定期在固定验证环境评估；
- 不以训练 reward 单独判断成功；
- 输出每个 reward component。

---

# 17. 基准方法

必须实现至少两个非 RL 基准：

## 17.1 Excel 基准

使用已有电流、速度 PID，位置环和 DOBC 使用保守初始化。

## 17.2 黑盒优化基准

使用一种传统优化方法，例如：

```text
scipy.optimize.differential_evolution
```

在同一仿真和同一 reward 下搜索 11 个参数。

比较：

```text
RL vs Excel/传统初值 vs 黑盒优化
```

这样才能判断 RL 是否真正有价值。

---

# 18. 输出结果

训练结束后生成：

```text
outputs/exported_parameters/best_parameters.json
outputs/exported_parameters/best_parameters.yaml
outputs/exported_parameters/best_parameters.csv
```

格式：

```json
{
  "kppos": 0.0,
  "kipos": 0.0,
  "kdpos": 0.0,
  "kpspeed": 0.0,
  "kispeed": 0.0,
  "kdspeed": 0.0,
  "kgspeed": 0.0,
  "tauspeed": 0.0,
  "kpcurr": 0.0,
  "kicurr": 0.0,
  "kdcurr": 0.0,
  "source": "SAC",
  "plant_variant": "speed_500mA",
  "metrics": {},
  "assumptions": []
}
```

还需生成：

```text
三环跟踪曲线
负载扰动响应
控制量与电流曲线
训练 reward 曲线
各 reward component 曲线
参数变化轨迹
频域指标对比表
时域指标对比表
```

---

# 19. 单元测试要求

至少测试：

1. NPZ key 和 shape；
2. 频率 Hz 到 rad/s 换算；
3. 传递函数频响；
4. 延迟缓冲；
5. PID anti-windup；
6. DOBC 关闭时输出为 0；
7. 参数映射可逆；
8. 同一 seed 仿真可复现；
9. 不稳定时环境终止；
10. reward 无 NaN/Inf；
11. 动作被正确裁剪；
12. 训练/测试对象不泄漏。

执行：

```bash
pytest -q
```

必须通过。

---

# 20. 实现里程碑

## M1：数据与频响

完成：

```text
data_loader.py
inspect_data.py
transfer_models.py
validate_frequency_response.py
```

交付实测与辨识模型 Bode 对比图。

## M2：确定性三环仿真

完成：

```text
PID
DOBC
multi-rate simulator
metrics
baseline run
```

先不接 RL。

## M3：参数化和环境

完成：

```text
parameterization.py
env.py
reward
domain randomization
tests
```

随机动作能够稳定运行多个 episode。

## M4：SAC 训练

完成：

```text
train_sac.py
checkpoints
logs
validation
```

## M5：评估与导出

完成：

```text
evaluate_policy.py
export_params.py
comparison reports
```

---

# 21. 验收标准

代码必须满足：

1. 能加载提供的 NPZ；
2. 能重现实测电流/速度辨识频响；
3. 能明确报告位置模型验证误差；
4. 能运行完整三环时域仿真；
5. 能设置并关闭 DOBC；
6. 能计算时域、频域和抗扰指标；
7. 能运行 Gymnasium 环境；
8. 能训练 SAC；
9. 能导出 11 个参数；
10. 所有缺失信息和假设有日志；
11. 不使用 Excel 中无效的 99999 指标；
12. 不把频率点当独立监督样本；
13. 不伪造位置 PID 和 DOBC 标签；
14. 训练结果可复现。

---

# 22. Codex 第一轮任务

先只执行 M1，不要直接生成整个项目的所有复杂代码。

第一轮请完成：

1. 创建项目目录；
2. 编写 `data_loader.py`；
3. 编写 `transfer_models.py`；
4. 编写 `scripts/inspect_data.py`；
5. 编写 `scripts/validate_frequency_response.py`；
6. 编写对应测试；
7. 运行测试；
8. 生成模型与实测频响对比图；
9. 输出发现的问题；
10. 不开始 SAC 训练。

完成 M1 并确认模型解释正确后，再继续 M2。
