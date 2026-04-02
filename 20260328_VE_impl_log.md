# VE 实现日志

日期：2026-03-28

## 背景

目标是把 consistency/distillation 路径中的**连续坐标分支**从原来的 VP/DDPM 形式改成 VE 形式，同时尽量复用现有 PocketXMol 的 teacher / student / eval 基础设施。

最初的直接改法是：

- 位置前向噪声改成 `x_t = x_0 + sigma_t * eps`
- 位置回推改成基于 `sigma_{t-1} / sigma_t` 的确定性插值
- 离散 node / edge 仍保留 categorical diffusion

但很快发现一个关键问题：

- 原始 teacher checkpoint 是按 VP 位置噪声训练的
- 如果直接把 VE 状态喂给 teacher，teacher 输入分布会失配

因此后续实现改成：

- **位置状态本身保持 VE**
- **模型输入前做 SNR/缩放对齐**


## 核心思路

### 1. 状态空间保持 VE

连续坐标状态使用：

```python
x_t = x_0 + sigma_t * eps
```

这样 consistency 的位置分支在状态定义上已经是 VE。

### 2. 模型输入对齐回 VP 形式

为了兼容现有 `x0-prediction` teacher，将送入模型的坐标做缩放：

```python
scale = 1 / sqrt(1 + sigma^2)
x_in = x_t * scale
```

原因是：

```python
x_t = x_0 + sigma * eps
x_in = x_t / sqrt(1 + sigma^2)
     = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
alpha_bar = 1 / (1 + sigma^2)
```

这正好映射回原始 VP teacher 更熟悉的输入形式。

这一步不是完整 EDM preconditioning，而是一个**teacher-compatible 的 SNR 对齐层**。


## 代码改动

### 1. `models/transition.py`

`ContigousTransition` 已从 beta 驱动改成 sigma 驱动：

- `__init__` 现在直接接收 `sigmas`
- `add_noise` 改成 `x + sigma * randn`
- `get_prev_from_recon` 改成确定性 VE rollback
- `sample_init` 改为依赖 `self.sigmas.device`

位置回推公式为：

```python
r = sigma_{t-1} / sigma_t
x_prev = r * x_t + (1 - r) * x_recon
```

当 `t == 0` 时直接返回 `x_recon`。


### 2. `utils/consistency.py`

`build_transitions` 现在的语义是：

- `sigmas` 用于位置 VE 分支
- `betas = sigmas_to_betas(sigmas)` 只给离散 node / edge 分支使用

新增：

```python
get_pos_snr_scale(transitions, t_graph, node_batch)
```

它返回：

```python
1 / sqrt(1 + sigma^2)
```

用于把 VE 坐标状态缩放到 teacher / student 模型的输入空间。


### 3. `scripts/train_consistency.py`

训练路径做了两类关键修改。

#### 3.1 避免污染原始 batch

teacher / ema / student 各自使用浅拷贝 batch：

- `teacher_batch`
- `ema_batch`
- `student_batch`

这样不会反复覆盖同一个 `batch["pos_in"] / batch["node_in"] / batch["halfedge_in"]`。

#### 3.2 位置输入增加 SNR 对齐

训练中：

- `pos_high` 先按 VE 生成
- 喂 teacher / student 前乘 `get_pos_snr_scale(...)`
- `pos_low_hat` 作为 VE rollback 结果
- 喂 EMA student 前，按 `t_low_graph` 对应的 sigma 再做一次缩放

即：

- 状态转移空间是 VE
- 模型输入空间是经过 SNR 对齐的 teacher-compatible 形式


### 4. `scripts/eval_consistency.py`

采样时每一步：

1. 当前 `node_state / pos_state / edge_state` 保持在 VE/categorical 状态空间
2. 对 `pos_state` 用当前 step 的 sigma 做 SNR 缩放
3. 再送入模型
4. 模型输出的 `pred_pos_x0` 仍视为数据空间坐标
5. 用 `renoise_from_pred_x0` 进入下一步

这样 eval 闭环和训练闭环对齐。


### 5. `scripts/generate_consistency.py`

与 `eval_consistency.py` 同样接入了位置 SNR 缩放，保证生成脚本与评估脚本使用一致的采样语义。


### 6. `configs/distill/distill_pxm.yaml`

改动较小：

- `wandb.id` 改为空字符串，避免复用旧 run id


### 7. 文档

已同步更新：

- [`VE_modify.md`](/shared/healthinfolab/phz24002/PocketXMol/VE_modify.md)

它现在区分了：

- 必须改
- 可选改
- 暂不改
- 验证项


## 为什么没有直接上完整 EDM

当前没有做这些内容：

- `sigma_data`
- `c_skip / c_out / c_in / c_noise`
- 明确的 noise-level embedding 接口
- 完整 EDM output parameterization

原因很直接：

- 当前 PocketXMol 模型接口没有显式 sigma 输入
- 原始 teacher 也不是按 EDM 形式训练的
- 直接硬接 EDM 会同时改状态定义、模型输入语义、模型输出语义，风险过高

所以当前版本的目标不是“全面 EDM 化”，而是：

**先把位置状态迁移到 VE，再用 SNR 缩放把模型输入对齐回 teacher 可接受的形式。**


## 验证与运行

### 已完成

- Python 语法检查：
  - `py_compile` 通过
- `gpu32` 上做过短时 smoke 训练
- 确认训练能真正进入 step 循环，不是只停在初始化

### 当前正式训练

已在 `gpu32` 上启动在线 wandb 训练：

- 机器：`gpu32`
- 进程：`PID 2723281`
- 输出目录：
  - [`outputs_distill_ve/consistency_distill_20260328_024114`](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_ve/consistency_distill_20260328_024114)
- wandb run 目录：
  - [`outputs_distill_ve/consistency_distill_20260328_024114/wandb/run-20260328_024138-m7pid9i8`](/shared/healthinfolab/phz24002/PocketXMol/outputs_distill_ve/consistency_distill_20260328_024114/wandb/run-20260328_024138-m7pid9i8)

日志中已确认：

- `Initializing new wandb run.`
- 已跑到数百 step

示例日志：

```text
step=50  total=0.059235
step=200 total=0.049456
step=350 total=0.010225
```


## 当前边界与后续建议

### 已知边界

1. 这不是完整 EDM

当前只是：

- VE 状态
- VP-compatible 模型输入缩放

2. evaluation 仍分两层

- `scripts/eval_consistency.py` 适合快速趋势检查
- 正式指标对比仍建议接：
  - `evaluate/evaluate_sdf_standard.py`
  - 必要时再跑 vina

3. teacher compatibility 仍是经验性方案

SNR 缩放是一个合理对齐方法，但仍需要靠后续采样质量来验证，而不是只看训练 loss。


### 后续建议

1. 训练到第一个 checkpoint 后，先做小规模评估

建议：

- `sample_steps = 4 8 16`
- `num_mols = 100 ~ 200`

2. 若趋势正常，再做标准评估

使用：

- `scripts/generate_consistency.py`
- `evaluate/evaluate_sdf_standard.py`

3. 如果 VE + SNR 方案仍明显不稳，再讨论下一步

优先级建议：

- 先考虑显式 sigma embedding
- 再考虑 EDM 输出参数化
- 最后才考虑更复杂的 loss / preconditioning 设计


## 一句话总结

本次实现不是“把 PocketXMol 改成完整 EDM”，而是：

**把位置分支迁移到 VE 状态空间，并用基于 SNR 的输入缩放，把原有 VP teacher / student 框架尽量无缝接到这套状态定义上。**
