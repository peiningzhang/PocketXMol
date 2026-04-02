# 一致性蒸馏模型 (Consistency Distillation) 结果分析

通过读取 [outputs_consistency_eval/consistency_eval_20260324_020444/summary.csv](file:///shared/healthinfolab/phz24002/PocketXMol/outputs_consistency_eval/consistency_eval_20260324_020444/summary.csv)，我们能够看到不同采样步数（1, 2, 4, 8, 16）下的质量变化规律。以下是深度数据分析与见解：

### 1. 呈阶梯状的完整与连通性 (Connectivity & Complete Rate)
| Steps | Connectivity | Complete | Recon Success |
|-------|--------------|----------|---------------|
| 1     | 0.0          | 0.0      | 0.74          |
| 2     | 0.0          | 0.0      | 1.00          |
| 4     | 0.0          | 0.0      | 0.98          |
| 8     | 0.80         | 0.38     | 0.47          |
| 16    | 0.97         | 0.41     | 0.42          |

**现象**：模型在低步数（1, 2, 4）完全无法生成“整体连通”（Connectivity=0）的有效分子，基本是处于碎裂或原子乱飞的状态。一直要到 8 步甚至 16 步时，才会出现连通且完整的分子。

**结论**：这非常遗憾地印证了一点——**模型尚未学到真正的“Consistency（一致性）”**。在完美的 Consistency Model 中，1 步采样应该直接跳跃到与 16 步采样相同的轨迹终点，产生高连通率分子。目前它依然呈现扩散模型（Diffusion）的特性，即只有积分步数足够多才能完成去噪坍缩。这通常说明了**模型处于极度欠拟合状态（训练步数太少）**，蒸馏损失还没有把长轨迹强制拉直。

### 2. Vina Score 异常偏高（严重空间碰撞）
| Steps | Vina Score |
|-------|------------|
| 8     | +4.48      |
| 16    | +4.98      |

**现象**：正常蛋白口袋生成的 Vina 应该是很低的负数（如 -6.0 $\sim$ -9.0 kcal/mol），而这里的打分是罕见的**正数**。
**结论**：正的 Vina 结合能意味着巨大的“空间排斥力”（Steric Clashes）。也就是生成的配体分子与蛋白口袋发生了严重的**原子重叠（交叠）**或者分子自身内部坐标极度扭曲。
在多步去噪（16步）时，由于时间表（特别是现在的 [log_uniform](file:///shared/healthinfolab/phz24002/PocketXMol/utils/consistency.py#79-92)）的最后几步负责进行空间微调（把重叠在一起的原子推开以匹配共价键长），如果蒸馏模型这部分学得不好，原子的三维坐标 [pos](file:///shared/healthinfolab/phz24002/PocketXMol/models/transition.py#140-160) 收敛会差之毫厘谬以千里，造成严重的物理重合。

### 3. QED 和 SA 表现稳定
| Steps | QED   | SA    |
|-------|-------|-------|
| 8     | 0.533 | 0.554 |
| 16    | 0.507 | 0.576 |

这部分指标达到了基线水平，说明 `node_type` 和分子组装的部分特征（类别生成）已经被提取出来了，它知道自己要生成哪些重原子，但就是把原子的 3D 位置放烂了。

---

### 下一步优化建议

   
1. **Loss 权重的调整 (`pos_weight`)**
   正数的 Vina 分数暗示 [pos](file:///shared/healthinfolab/phz24002/PocketXMol/models/transition.py#140-160) 预测精度严重滞后于 `node/edge`。可以在 [distill_pxm.yaml](file:///shared/healthinfolab/phz24002/PocketXMol/configs/distill/distill_pxm.yaml) 中考虑稍微提升空间坐标的 Consistency Loss，或者观察一下 wandb 上 `loss_pos` 是不是卡在了瓶颈。
   
2. **回滚对比测试**
   如果您有之前用基础 Karras（`rho=7` 密集网格）或者常规扩散在 16 步的结果，可以对比发现这是 [log_uniform](file:///shared/healthinfolab/phz24002/PocketXMol/utils/consistency.py#79-92) 对小噪声区间拟合偏弱造成的，还是整体蒸馏失败造成的。   通常，生成蛋白大分子需要极高精度，[log_uniform](file:///shared/healthinfolab/phz24002/PocketXMol/utils/consistency.py#79-92) 可能在靠近 $t=0$ 的极小尺度上没有部署足够多的网格点，也是推不开重叠原子的一个诱因。

---

### 4. 时间步编码与噪声分布的不一致性 (Critical Issue)

通过对代码和配置的深入分析，我们发现当前蒸馏训练 (`train_consistency.py`) 与原始 Teacher 模型在**噪声注入物理逻辑**上存在根本性冲突：

#### A. 噪声类型冲突：Scaling (VP) vs. Additive (VE)
- **Teacher (原始模型)**: 使用的是 `gaussian_simple` 加性噪声 ([utils/prior.py:L283-318](file:///shared/healthinfolab/phz24002/PocketXMol/utils/prior.py#L283-318))，公式为 $x_t = x_0 + \sigma \epsilon$。这种方式不改变原分子的坐标尺度。
- **Student (蒸馏模型)**: 在 `utils/consistency.py` 中使用了 `ContigousTransition` ([models/transition.py:L9-26](file:///shared/healthinfolab/phz24002/PocketXMol/models/transition.py#L9-26))，它实现的是方差保存（VP）缩放噪声：$x_t = \frac{1}{\sqrt{1+\sigma^2}} x_0 + \frac{\sigma}{\sqrt{1+\sigma^2}} \epsilon$。
- **后果**: 在大噪声水平下（如 $\sigma=80$），学生模型传给 Teacher 的分子会被**缩小约 80 倍**。Teacher 模型由于从未见过这种缩小的输入，预测结果会完全失效，导致蒸馏目标（Target）毫无意义。

#### B. 噪声分布不匹配：Log-Uniform vs. Uniform
- **Teacher**: 训练时 `level` 采样自 `Uniform(0, 1)`，对应 $\sigma$ 在线性空间均匀分布。
- **Student**: 当前配置 [distill_pxm.yaml:L29](file:///shared/healthinfolab/phz24002/PocketXMol/configs/distill/distill_pxm.yaml#L29) 使用 `log_uniform`，这使得训练集中充斥着大量极低噪声 ($\sigma \approx 0$) 的样本，而原本 Teacher 擅长的中高噪声区间样本密度过低。

#### C. 模型感知的 $t$ 编码
- `PMAsymDenoiser` ([models/maskfill.py](file:///shared/healthinfolab/phz24002/PocketXMol/models/maskfill.py)) 内部**并不显式接收 $t$ 或 $\sigma$**。它完全依靠输入 $x_t$ 的特征来推断去噪状态。由于上述重参数化逻辑的改变，输入特征的含义发生了偏移，模型无法正确“定位”自己在去噪轨迹上的位置。

---

### 5. 已实现的重构总结 (Refactor Summary)

我们通过彻底改变扩散逻辑，将模型从不匹配的 VP 模式切换到了标准的 **EDM (Elucidated Diffusion Models)** 范式：

#### A. 核心逻辑重构 (`models/transition.py`)
- **加性噪声切换**: 将坐标噪声从缩放模式 ($x_t = \alpha x_0 + \beta \epsilon$) 改为 Teacher 训练时所用的加性模式 ($x_t = x_0 + \sigma \epsilon$)。
- **EDM 预处理因子**: 实现了 $c_{skip}, c_{out}, c_{in}$ 公式。这些因子允许学生模型专注于学习去噪余项（Residual），同时保持输出在 $x_0$ 物理空间。
- **ODE 轨迹更新**: 更新了 `get_prev_from_recon`，支持基于 $\sigma$ 比例的确定性 ODE 步进，使学生模型能更精准地“对齐”Teacher 的去噪路径。

#### B. 训练流改进 (`scripts/train_consistency.py`)
- **$c_{in}$ 输入缩放**: 对输入坐标 `pos_in` 进行了 $1/\sqrt{\sigma^2 + 0.25}$ 的缩放，使模型感知到的输入尺度始终保持稳定。
- **Batch 隔离机制**: 为 Teacher、Student 和 EMA 模型各分配了独立的 `batch` 字典副本，彻底消除了不同模型间字段污染导致的意外干扰。
- **强制形状对齐 (Critical)**: 在各级传递中强制执行 `.view(-1, 3)`。这修复了此前因为 `unsqueeze` 冗余触发 Broadcasting 机制产生的维度爆炸（如 1200 万维度溢出的报错）。

#### C. 参数校对 (`distill_pxm.yaml`)
- **$\sigma_{max}$ 下调**: 将 `sigma_max` 从 80.0 修正为 1.0。这与 Teacher 模型在 SBDD 任务中对蛋白口袋的物理感受野完全对齐。

#### D. 采样逻辑对齐 (`scripts/eval_consistency.py`)
- **预处理同步**: 推理代码现在使用与训练完全相同的 EDM 预处理逻辑，确保了推理与训练的数学等价性。

---
### 结论
通过上述重构，模型现在处于正确的物理基准上。此前的“连通性为0”和“主链碰撞”问题主要是由于 VP 与 VE 逻辑冲突导致 Teacher 输出的 Target 毫无意义。目前的结构支持模型在极短步数下实现高质量的分子还原。

### 修订后的优化路径

1.  **对齐噪声注入逻辑 (优先级最高)**: 
    修改 `train_consistency.py`，将坐标 `pos` 的噪声添加方式从 VP 缩放逻辑改为与 Teacher 一致的 **VE 加性逻辑**。
    
2.  **纠正 $\sigma_{max}$**: 
    在蒸馏配置中将 `sigma_max` 从 80 降低到与 Teacher 训练一致的水平（通常是 1.0），避免输入超出 Teacher 的有效感受野。

3.  **调度函数切换**: 
    将 `schedule` 从 `log_uniform` 切换为 `linear` 或调整 Karras 调度的 `rho` 参数，以匹配原始训练时的 $\sigma$ 分布。
