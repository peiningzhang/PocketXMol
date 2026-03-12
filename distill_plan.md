# 📝 研究计划：PocketXMol 的混合状态物理感知一致性蒸馏 (Physics-Aware Mixed-State Consistency Distillation)

## 1. 研究背景与核心动机 (Motivation)

* **当前痛点**：PocketXMol 等全原子 3D 分子扩散模型展现了极高的准确性（统一 13 个任务），但其依赖数百至上千步的马尔可夫迭代去噪，导致推理极慢，难以胜任工业级高通量虚拟筛选 (HTVS)。
* **传统方法的局限**：现有的加速采样方法（如 DDIM 或少步渐进式蒸馏）在跳步幅度过大时，会打破原子坐标（连续变量）与原子类型（离散变量）在多步迭代中的“协商机制”，导致严重的物理规则崩塌（如空间位阻、化合价异常）。
* **我们的目标**：提出首个针对 3D 分子混合状态的**直接一致性蒸馏 (Direct Consistency Distillation)** 框架。通过一步映射预测干净数据，并引入联合物理约束，实现 1-4 步极速采样的同时，保持与 1000 步扩散相媲美的物理合法性与亲和力。

## 2. 核心理论创新 (NeurIPS 卖点)

本计划的核心突破在于提出 **“物理感知的混合一致性模型 (Physics-Aware Mixed Consistency Model)”**。

一致性模型 (Consistency Models) 的核心思想是学习一个映射函数 $f_\theta(x_t, t)$，使得同一条 ODE 轨迹上的任意点都映射到同一个起点 $x_0$。
由于我们的模型在**任意时间步 $t$ 都直接预测最终的“干净状态” $(\hat{X}_0, \hat{C}_0)$**，这赋予了我们一个传统扩散模型所不具备的巨大优势：**我们可以直接在一致性蒸馏的训练循环中，对预测结果施加确切的物理与几何惩罚。**

这完美解决了“独立计算连续 Loss 和离散 Loss 导致深度脱节”的问题。

## 3. 算法设计与损失函数 (Algorithm & Objectives)

### 3.0 PocketXMol 的“混合状态一致性”定义 (Mixed-State Consistency)

PocketXMol 的生成状态是混合变量：原子坐标为连续变量，而原子/键类型为离散变量。为避免把图像领域“连续时间 PF ODE + score”叙述生搬硬套到离散变量上，我们定义一致性模型的目标为：给定任意噪声水平的输入图状态 \((X_t, C_t)\) 与时间 \(t\)，学生模型直接预测干净终点态的**同一表征**：

- **坐标头**：输出 \(\hat{X}_0\)（或等价的去噪坐标表征）
- **类型头**：输出 \(\hat{C}_0\)（原子/键类型 logits 或概率分布）

教师模型用于生成**相邻时间步对** \((\hat{x}^\phi_{t_n}, x_{t_{n+1}})\) 的“轨迹监督”。其中连续部分可通过一步 ODE solver 近似 PF ODE 轨迹点；离散部分则以教师在对应时间步的 logits/分布作为蒸馏目标（KL/CE），不强行引入不自然的离散 ODE。

### 3.1 一致性蒸馏基础损失 ($L_{CD}$)

在直接一致性蒸馏中，我们使用预训练的 PocketXMol 作为教师网络来估计 ODE 轨迹，并训练一个学生网络 $f_\theta$。目标是最小化相邻时间步的一致性误差：

* **连续变量 (3D 坐标)**：

$$L_{pos} = \mathbb{E} \left[ \left\| f^{pos}_\theta(x_{t_{n+1}}, t_{n+1}) - f^{pos}_{\theta^-}(\hat{x}_{t_n}, t_n) \right\|^2_2 \right]$$


* **离散变量 (原子/键类型)**：

$$L_{type} = \mathbb{E} \left[ D_{KL} \left( f^{type}_{\theta^-}(\hat{x}_{t_n}, t_n) \parallel f^{type}_\theta(x_{t_{n+1}}, t_{n+1}) \right) \right]$$



*(注：$\theta^-$ 为学生网络的指数移动平均 (EMA) 权重，$\hat{x}_{t_n}$ 为教师模型从 $t_{n+1}$ 单步退化得到的状态)*

### 3.1.1 学生模型接口约定 (Engineering Interface)

为保证直接实现并复现实验，约定学生模型（及其 EMA 副本）输出统一的终点态预测：

- 输入：`(graph_state_at_t, t)`，其中 `graph_state_at_t` 包含 `node_type/halfedge_type/node_pos` 等 PocketXMol 现有字段（与采样时 batch 的字段保持一致）
- 输出：`pred_x0` 字典，至少包含：
  - `pred_node_logits_x0`（原子类型 logits 或概率）
  - `pred_halfedge_logits_x0`（键类型 logits 或概率）
  - `pred_pos_x0`（去噪坐标）

这样一致性损失可以统一写成“终点态预测之间的差异”，采样时也能直接用 \(\hat{x}_0\) 作为一步/少步生成的输出。

### 3.2 深度融合：联合物理惩罚损失 ($L_{Physics}$)

为了防止 1-4 步生成时发生空间位阻和几何崩塌，直接利用学生网络预测的终点态 $(\hat{X}_0, \hat{C}_0)$ 计算物理惩罚：

```python
def compute_joint_physics_loss(pred_pos_x0, pred_node_type_c0):
    """
    pred_pos_x0: 模型预测的最终 3D 坐标 (连续)
    pred_node_type_c0: 模型预测的最终原子类型分布 (离散)
    """
    # 1. 将连续的坐标转为成对距离矩阵
    distance_matrix = torch.cdist(pred_pos_x0, pred_pos_x0)
    
    # 2. 将离散的原子类型转换为预期的范德华半径 (VdW Radii)
    # 利用 Gumbel-Softmax 或直接 Softmax 期望保持可导性
    expected_vdw = torch.matmul(torch.softmax(pred_node_type_c0, dim=-1), VDW_RADII_TABLE)
    
    # 3. 计算预期的最小安全距离矩阵 (r_i + r_j)
    safe_distance_matrix = expected_vdw.unsqueeze(1) + expected_vdw.unsqueeze(2)
    
    # 4. 空间位阻惩罚 (Steric Clash Penalty): 惩罚实际距离小于安全距离的原子对
    clash_mask = (distance_matrix < safe_distance_matrix) & (distance_matrix > 0)
    clash_penalty = torch.sum(clash_mask * (safe_distance_matrix - distance_matrix)**2)
    
    return clash_penalty

```

**总损失函数**：


$$L_{total} = L_{pos} + \lambda_1 L_{node\_type} + \lambda_2 L_{edge\_type} + \lambda_{phys} L_{Physics}$$

## 4. 实施阶段计划 (Implementation Phases)

### 阶段 1：一致性蒸馏框架搭建

* **文件**: `scripts/train_consistency.py`
* **任务**:
1. 加载预训练的 PocketXMol (Teacher，冻结参数)。
2. 初始化 Student 模型及其 EMA (Exponential Moving Average) 副本。
3. 参考https://arxiv.org/pdf/2303.01469，复用 PocketXMol 的噪声日程与 time embedding 方式，固定一套时间网格 \(\{t_n\}\)（Karras 公式，\(\rho=7\)，给定 \(\epsilon, T\)），并实现“一步 teacher 轨迹点”生成：
   - 连续坐标：用 Euler 或 Heun 对教师诱导的 PF ODE 做一步更新，得到 \(\hat{X}^\phi_{t_n}\)
   - 离散类型：直接使用教师在 \(t_n\) / \(t_{n+1}\) 的 logits 分布作为蒸馏监督（KL/CE）



### 阶段 2：混合损失与物理评估模块

* **文件**: `models/consistency_loss.py`
* **任务**:
1. 实现连续变量的 MSE Loss 和离散变量的 KL Div Loss。
2. **核心代码**：实现 `compute_joint_physics_loss`，构建 VdW 半径查找表，确保梯度可以通过物理距离计算反向传播至原子类型预测层和坐标预测层。


### 阶段 3：单阶段直接蒸馏训练

* **任务**:
1. 固定一个离散时间步数 \(N\)（例如 10/20/50；先小后大做对照），按照 Karras 等人的公式一次性生成时间网格 \(\{t_n\}_{n=1}^N\)，在整个训练过程中 **不再动态增加或调整时间步**。
2. 每个 iteration **随机采样** 一个时间步索引 \(n \sim \mathcal{U}\{1,\dots,N-1\}\)（或按噪声权重采样），从数据分布采样 \(x\)，并按 SDE 转移采样 \(x_{t_{n+1}}\)。
3. 用教师模型构造相邻对：根据一步 ODE solver 得到连续部分 \(\hat{X}^\phi_{t_n}\)，并取教师在对应时间步的离散 logits 作为类型监督，形成 \((\hat{x}^\phi_{t_n}, x_{t_{n+1}})\)。
4. 用一致性损失训练单个一致性模型 \(\mathcal{C}_\theta\)（配合 EMA \(\theta^-\)），并加入 \(L_{\text{Physics}}\) 做联合约束。重点记录：训练稳定性（loss 曲线/梯度范数）、以及少步采样的有效性指标。


### 阶段 4：极速采样与全面评估

* **文件**: `scripts/eval_consistency.py`
* **任务**:
1. 使用训练好的学生模型执行 **1步、2步、4步、8步** 采样。
2. 对比指标（形成“步数-质量-速度”对照表）：
* **速度指标**：wall time / samples-per-second / GPU util（至少报告 wall time 与吞吐）。
* **基础指标**：Vina Score (亲和力), QED, SA。
* **验证指标**：recon_success / complete / validity / connectivity / steric clashes（与你现有 `evaluate_sdf_standard.py` 对齐）。


3. **Ablation Study (消融实验)**：对比去除 $L_{Physics}$ 的传统独立一致性蒸馏，证明“深度融合”在极速采样下的不可替代性。



## 5. 预期成果 (Expected Outcomes)

1. **性能突破**：实现只需 **2-8 步** 即可生成合法且高亲和力的 3D 分子，推理速度比原始 PocketXMol 提升 **10-50 倍**。
2. **理论自洽**：通过实验强有力地证明，在混合变量模型的一致性映射中，跨模态物理约束 ($L_{Physics}$) 是防止极速生成发生结构坍塌的必要条件。
3. **论文交付**：产出一篇具备完整理论推导、针对“AI for Science 极速采样痛点”的 NeurIPS 强竞争力论文。

## 6. 配置示例
### configs/distill/distill_pxm.yaml
distill:
  teacher_checkpoint: "data/trained_models/pocketxmol.ckpt"
  teacher_steps: 1000
  student_steps: [50, 100]  # 依次训练
  iterations: 3  # 每个步数迭代蒸馏轮数
  
  loss_weights:
    pos: 1.0
    node: 1.0
    halfedge: 1.0
    
  optimization:
    lr: 1e-5
    batch_size: 8
    epochs_per_iter: 10
eval:
  test_steps: [1, 5, 10, 25, 50, 100]  # 评估时测试的采样步数
  metrics: ["vina", "validity", "diversity", "sa", "qed"]


## 7. 执行流程
### 1. 运行轨迹蒸馏
python scripts/distill_trajectory.py --config configs/distill/distill_pxm.yaml
### 2. 迭代评估 (自动测试不同采样步数)
python scripts/evaluate_distilled.py \
    --distill_dir outputs_distill/ \
    --test_steps 1 5 10 25 50 100 \
    --is_ar ""  # 非AR模式
### 3. 生成对比报告
python scripts/compare_results.py \
    --baseline data/trained_models/pocketxmol.ckpt \
    --distilled outputs_distill/ \
    --output comparison_report.csv

## 8.现有的none AR模式的采样和评估
### noAR的采样

```
conda_shared
conda activate PocketXMol
cd /shared/healthinfolab/phz24002/PocketXMol/
CUDA_VISIBLE_DEVICES=1 python scripts/sample_drug3d.py --config_task configs/sample/test/sbdd_csd/simple.yml --outdir outputs_test/sbdd_csd_noAR/
```

评估：

```
(pxm_vina) [phz24002@gpu39 PocketXMol]$ python evaluate/evaluate_vina_sdf.py   --sdf_dir outputs_test/sbdd_csd_noAR/simple_pxm_20260311_150536/SDF/   --gen_info outputs_test/sbdd_csd_noAR/simple_pxm_20260311_150536/gen_info.csv   
--split_by_name_path /shared/healthinfolab/phz24002/AliDiff/data/split_by_name.pt   --test_set_root /shared/healthi
nfolab/phz24002/AliDiff/data/test_set   --mode score_only   --n_workers 16   --quiet
---
Saved 10000 results to outputs_test/sbdd_csd_noAR/simple_pxm_20260311_150536/vina.csv
  vina_score mean: -5.76
  vina_score median: -5.99
---
python evaluate/evaluate_sdf_standard.py   --sdf_dir outputs_test/sbdd_csd_
noAR/simple_pxm_20260311_150536/SDF/   --gen_info outputs_test/sbdd_csd_noAR/simple_pxm_20260311_150536/gen_info.cs
v   --split_by_name_path /shared/healthinfolab/phz24002/AliDiff/data/split_by_name.pt   --test_set_root /shared/healthinfolab/phz24002/AliDiff/data/test_set
---
[2026-03-11 21:20:00,184::evaluate_sdf_standard::INFO] mol_stable:      NA
[2026-03-11 21:20:00,191::evaluate_sdf_standard::INFO] atm_stable:      NA
[2026-03-11 21:20:00,199::evaluate_sdf_standard::INFO] recon_success:   0.9027
[2026-03-11 21:20:00,203::evaluate_sdf_standard::INFO] eval_success:    0.8742
[2026-03-11 21:20:00,204::evaluate_sdf_standard::INFO] complete:        0.8861
[2026-03-11 21:20:00,267::evaluate_sdf_standard::INFO] JS bond distances:
[2026-03-11 21:20:00,272::evaluate_sdf_standard::INFO] JSD_6-6|4:       0.3959
[2026-03-11 21:20:00,279::evaluate_sdf_standard::INFO] JSD_6-6|1:       0.3503
[2026-03-11 21:20:00,287::evaluate_sdf_standard::INFO] JSD_6-8|1:       0.2976
[2026-03-11 21:20:00,295::evaluate_sdf_standard::INFO] JSD_6-7|1:       0.3004
[2026-03-11 21:20:00,300::evaluate_sdf_standard::INFO] JSD_6-8|2:       0.3925
[2026-03-11 21:20:00,304::evaluate_sdf_standard::INFO] JSD_6-6|2:       0.2754
[2026-03-11 21:20:00,311::evaluate_sdf_standard::INFO] JSD_6-7|4:       0.2043
[2026-03-11 21:20:00,315::evaluate_sdf_standard::INFO] JSD_6-7|2:       0.3037
[2026-03-11 21:20:00,902::evaluate_sdf_standard::INFO] JS pair distances:
[2026-03-11 21:20:00,903::evaluate_sdf_standard::INFO] JSD_CC_2A:       0.3136
[2026-03-11 21:20:00,904::evaluate_sdf_standard::INFO] JSD_All_12A:     0.0859
[2026-03-11 21:20:00,904::evaluate_sdf_standard::INFO] Atom type JS: 0.0669
[2026-03-11 21:20:00,914::evaluate_sdf_standard::INFO] QED:   Mean: 0.520 Median: 0.527
[2026-03-11 21:20:00,920::evaluate_sdf_standard::INFO] SA:    Mean: 0.740 Median: 0.760
[2026-03-11 21:20:00,933::evaluate_sdf_standard::INFO] Vina Score: Mean: -5.760 Median: -5.985
[2026-03-11 21:20:00,943::evaluate_sdf_standard::INFO] Vina Min:   Mean: -6.697 Median: -6.617
[2026-03-11 21:20:00,952::evaluate_sdf_standard::INFO] ring size: 3 ratio: 0.010
[2026-03-11 21:20:00,961::evaluate_sdf_standard::INFO] ring size: 4 ratio: 0.020
[2026-03-11 21:20:00,972::evaluate_sdf_standard::INFO] ring size: 5 ratio: 0.459
[2026-03-11 21:20:00,979::evaluate_sdf_standard::INFO] ring size: 6 ratio: 0.787
[2026-03-11 21:20:00,989::evaluate_sdf_standard::INFO] ring size: 7 ratio: 0.128
[2026-03-11 21:20:00,999::evaluate_sdf_standard::INFO] ring size: 8 ratio: 0.011
[2026-03-11 21:20:01,005::evaluate_sdf_standard::INFO] ring size: 9 ratio: 0.002
[2026-03-11 21:20:02,035::evaluate_sdf_standard::INFO] Saved metrics to outputs_test/sbdd_csd_noAR/simple_pxm_20260311_150536/SDF/eval_results_standard/metrics_sdf_0-to-9999.pt
[2026-03-11 21:20:02,036::evaluate_sdf_standard::INFO] Done.
```