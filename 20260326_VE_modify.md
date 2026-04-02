# VE 修改方案（可实施版本）

目标：把 consistency 路径中的**连续坐标分支**从当前的 VP/DDPM 风格，改成 **VE 风格的 sigma 调度与采样**；同时保持离散的 node / edge 分支继续沿用现有 beta-parameterized categorical diffusion。

这份文档只保留当前仓库里**能直接落地**、且接口上自洽的改动。凡是需要额外模型接口设计、训练目标重定义、或缺少明确理论约束的部分，统一降级为“可选改”或“暂不改”。


## 一、必须改

以下改动是把“位置分支改成 VE”所必需的最小闭环；不做这些，当前 consistency 代码仍然只是“用 sigma 网格生成 betas”，并不是真正的 VE 位置扩散。

### 1. `models/transition.py`

#### 1.1 重写 `ContigousTransition` 的连续坐标逻辑

当前实现仍然是标准 VP / DDPM 形式：

- `__init__` 接收 `betas`
- `add_noise` 使用 `sqrt(alpha_bar) * x + sqrt(1 - alpha_bar) * eps`
- `get_prev_from_recon` 使用 DDPM posterior 的 `mu + sigma * randn`

对位置分支的目标改法：

- `ContigousTransition` 改为直接接收 `sigmas`
- 内部持有升序 sigma 网格 `sigma[0] ... sigma[T-1]`
- `sample_init` 不再依赖 `self.betas`
- 连续坐标分支不再构造 `alphas_bar / coef_x0 / coef_xt / std`

建议接口：

```python
class ContigousTransition(nn.Module):
    def __init__(self, sigmas, num_classes=None, scaling=1.0):
        ...
```

#### 1.2 `add_noise` 改为 VE 形式

把：

```python
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
```

改为：

```python
x_t = x_0 + sigma_t * eps
```

要求：

- `time_step == 0` 时仍返回干净样本，保持当前 consistency 代码对 `t=0` 的语义
- 支持按 graph-level `time_step` + atom-level `batch` 索引 sigma
- 保留 `scaling` 逻辑，避免破坏现有接口

#### 1.3 `get_prev_from_recon` 改为确定性 VE 步进

把当前随机一步：

```python
x_prev = mu + std * randn
```

改为基于 sigma 比例的确定性更新。建议使用文档中已有思路：

```python
r = sigma_{t-1} / sigma_t
x_prev = r * x_t + (1 - r) * x_recon
```

要求：

- `t == 0` 时直接返回 `x_recon`
- 不再向该步额外注入随机噪声
- 明确只用于 consistency distillation 中的 teacher one-step rollback，不声称它是完整的 VE reverse SDE / ODE 求解器

#### 1.4 `sample_init` 修正

当前 `sample_init` 依赖 `self.betas.device`。改成 sigma 版后，应改为依赖 `self.sigmas.device`。


### 2. `utils/consistency.py`

#### 2.1 `build_transitions` 中位置分支真正使用 `sigmas`

当前文件已经有：

- `karras_sigmas`
- `log_uniform_sigmas`
- `sigmas_to_betas`

但位置分支仍然是：

```python
"pos": ContigousTransition(betas)
```

必须改为：

```python
"pos": ContigousTransition(sigmas.detach().cpu().numpy())
```

保留：

- node transition: `GeneralCategoricalTransition(betas, ...)`
- edge transition: `GeneralCategoricalTransition(betas, ...)`

即：

- **位置**走纯 sigma / VE
- **离散 node / edge** 继续走由 `sigmas_to_betas` 派生的 categorical diffusion

#### 2.2 `normalize_pred_x0` 先保持“恒等输出语义”

当前实现：

```python
pred_pos_x0 = outputs["pred_pos"]
```

这一点在“最小可实施版本”里**先不要改成 EDM preconditioning 反组合**。原因：

- 当前模型没有显式 `sigma` / `c_noise` 输入接口
- 当前 `pred_pos` 的训练语义就是“直接预测数据空间坐标”
- 如果现在强行加 `c_skip / c_out`，会改掉坐标头语义，影响 teacher / EMA / student 一致性

因此：

- `normalize_pred_x0` 对 node / edge / pos 继续保持当前恒等映射
- 最多只允许补充注释，说明“这里还不是 EDM parameterization”

#### 2.3 `renoise_from_pred_x0` 保持现有离散逻辑，只替换位置加噪后端

当前：

- `pos_next = transitions["pos"].add_noise(pred_pos_x0, t_next, ...)`
- node / edge 用 `q_vt_sample`

这个框架可保留。只要位置 transition 已改为 sigma 版，这里就自动变成 VE 风格的 re-noise。


### 3. `scripts/train_consistency.py`

#### 3.1 避免原地污染 `batch`

当前代码反复覆盖同一个 `batch["node_in"] / batch["halfedge_in"] / batch["pos_in"]`。这虽然能跑，但在 teacher / EMA / student 共用同一个 batch 对象时很脆弱。

必须改为：

- `teacher_batch = copy.copy(batch)` 或明确的浅拷贝字典
- `ema_batch = copy.copy(batch)`
- `student_batch = copy.copy(batch)`

要求：

- 不复制大张量数据本体，只复制映射关系
- 每个分支只写自己的 `node_in / halfedge_in / pos_in`
- 不改动原始监督字段，如 `node_type / node_pos / halfedge_type`

#### 3.2 训练闭环保持“teacher high -> low_hat -> EMA target；student high -> predict”

最小可实施版本里，训练语义保持如下：

1. 从 `t_high ~ Uniform({1, ..., T-1})`
2. 对 node / edge / pos 构造 high-noise 状态
3. teacher 从 high-noise 状态预测 `x0`
4. 用位置 VE `get_prev_from_recon` 得到 `pos_low_hat`
5. 用离散 posterior 得到 `node_low_hat / edge_low_hat`
6. EMA student 在 low-hat 状态上预测
7. student 在 high-noise 状态上预测
8. 用现有 consistency loss 对 student / EMA 结果做匹配

这里不引入额外 `c_in / c_skip / c_out`。

#### 3.3 明确 shape 约束

文档里提到过 `view(-1, 3)` 风险。这条保留为实现约束：

- 所有传入位置 transition 的坐标张量都必须是 `(N, 3)`
- `teacher_pred["pred_pos_x0"]`、`pos_high`、`pos_low_hat` 如有潜在额外维度，进入 transition 前应显式整理

但只有在实际出现 shape mismatch 时才加 reshape；不要无条件到处 `view(-1, 3)`。


### 4. `scripts/eval_consistency.py` 与 `scripts/generate_consistency.py`

#### 4.1 采样逻辑保持当前框架

当前采样闭环可继续使用：

1. 从 prior 初始化 `node_state / pos_state / edge_state`
2. 逐步调用 model 得到 `pred_x0`
3. 若未到最后一步，则调用 `renoise_from_pred_x0`
4. 最后一步直接取 `pred_x0`

只要位置 transition 已改为 sigma 版，这个闭环就已经是“VE position + categorical discrete”的版本。

#### 4.2 同样避免原地污染输入 batch

建议在每个采样 step 里使用：

```python
input_batch = copy.copy(batch)
```

然后写入：

- `input_batch["node_in"]`
- `input_batch["pos_in"]`
- `input_batch["halfedge_in"]`

避免后续 post-process 或 debug 时混淆 batch 内真实字段与临时采样状态。

#### 4.3 打印 transition/schedule 日志

建议补一条日志，至少包含：

- number of steps
- schedule type
- sigma_min
- sigma_max

方便确认当前运行的是哪套 grid。


### 5. `configs/distill/distill_pxm.yaml`

配置上必须改的只有一条：

- `wandb.id` 不应固定复用旧 run id；默认应为空字符串，除非明确想 resume 同一个 run

其余超参不属于“必须改”：

- `num_steps`
- `sigma_max`
- `batch_size`
- `max_steps`

这些需要实验验证，不能作为算法实现的一部分直接写死。


## 二、可选改

这些改动可能有价值，但不属于当前仓库下“位置 VE 最小闭环”的必要条件。

### 1. 增加 `schedule_type` 以外的 schedule 诊断输出

例如在 `build_transitions` 后打印：

- 前 3 个 sigma
- 后 3 个 sigma
- 对应 derived betas 的范围

用于排查异常 schedule。


### 2. 为 `ContigousTransition` 增加辅助函数

可选增加：

- `get_sigma(t, batch)`
- `get_sigma_pair(t, batch)` 返回 `(sigma_t, sigma_prev)`

让 `add_noise` 和 `get_prev_from_recon` 的索引逻辑更清晰。


### 3. 给 `normalize_pred_x0` 增加注释或扩展接口占位

可以把函数签名扩成：

```python
def normalize_pred_x0(outputs, batch=None, transitions=None, t=None, pos_orig=None):
    ...
```

但在当前版本中：

- 不使用 `transitions`
- 不使用 `t`
- 不做 EDM `c_skip / c_out` 组合

仅作为未来接口占位，避免后续再改调用点。


### 4. 配置实验建议

可以作为实验分支尝试，但不要在主文档中写成“应当如此”：

- `num_steps: 50 -> 100`
- `sigma_max: 80 -> 1`
- `batch_size: 8 -> 32`
- `max_steps: 50000 -> 100000`

这些都要以数据尺度、显存、收敛曲线为准。


## 三、暂不改

以下内容先明确**不纳入本轮实施**，避免把文档写成“大而全但落不了地”的状态。

### 1. 暂不接完整 EDM preconditioning

包括但不限于：

- `sigma_data`
- `get_precondition(sigma)`
- `c_skip`
- `c_out`
- `c_in`
- `c_noise`

原因：

- 当前模型接口没有显式 noise-level conditioning
- 当前 `PMAsymDenoiser` 并未接收 `sigma` / `timestep embedding`
- 在没有统一 teacher / EMA / student 输入语义前，强行接入只会造成训练目标不一致

只有在模型结构层面明确“如何把 sigma 输入网络”后，才应单独开新文档讨论 EDM parameterization。


### 2. 暂不采用“teacher 不乘 c_in，EMA/student 乘 c_in”的混合策略

这条在旧 patch 摘要里出现过，但当前没有足够理由证明它是正确设计，而不是历史遗留不一致。

本轮实现要求：

- teacher / EMA / student 的位置输入语义保持一致
- 都直接在“真实坐标尺度”下工作


### 3. 暂不修改 consistency loss 的目标定义

当前 loss：

- pos: MSE
- node: KL
- edge: KL

先保持不变。位置分支改成 VE transition，不等于必须同时改 loss parameterization。


### 4. 暂不加入在线评测自动化 patch

例如：

- 周期性 subprocess 调 `eval_consistency`
- 跑 Vina
- 读取 `summary.csv`
- 回写 wandb

这些属于训练基础设施，不属于 VE 核心改造。


## 四、验证项

实现完成后，至少做以下验证。

### 1. 单元级验证

#### 1.1 `ContigousTransition.add_noise`

检查：

- `t=0` 时输出等于输入
- `t>0` 时噪声强度随 sigma 增大而增大
- 支持 graph-level timestep + atom-level batch gather

#### 1.2 `ContigousTransition.get_prev_from_recon`

检查：

- `t=0` 时输出等于 `x_recon`
- `sigma_prev < sigma_t` 时，`x_prev` 比 `x_t` 更靠近 `x_recon`
- 同一输入下结果 deterministic


### 2. 训练 smoke test

至少跑一个很小的训练：

- 1 到 10 step
- 单卡
- 小 batch

确认：

- 无 shape error
- 无 device error
- loss 为有限值
- teacher / EMA / student 前向都能通过


### 3. 采样 smoke test

分别对：

- `scripts/eval_consistency.py`
- `scripts/generate_consistency.py`

做最小样本采样，确认：

- 能完整跑完
- 输出 SDF 正常写出
- 最终 `pred_pos_x0` 无 NaN / Inf


### 4. 与当前版本的最小对照

至少记录：

- 训练初期 loss 曲线是否稳定
- 1-step / 4-step 采样是否退化
- 生成分子重建失败率是否明显恶化

如果 VE 版在这些最基础指标上明显更差，应先排查 sigma 网格、采样步进公式和位置尺度，而不是继续叠加 EDM 改动。


## 五、实施顺序

建议按以下顺序提交：

1. `models/transition.py`
   - 把位置分支改成纯 sigma / VE
2. `utils/consistency.py`
   - 让位置 transition 真正吃 sigmas
3. `scripts/train_consistency.py`
   - 清理 batch 污染，接上新的位置 rollback
4. `scripts/eval_consistency.py` / `scripts/generate_consistency.py`
   - 清理采样时的 batch 写入方式
5. `configs/distill/distill_pxm.yaml`
   - 只改 `wandb.id`


## 六、一句话边界

本轮目标不是“把 PocketXMol 全面改成 EDM 模型”，而是：

**把 consistency 路径中的连续坐标扩散从 VP 后验采样，收敛成一个接口自洽、可训练、可采样的 VE 版本。**
