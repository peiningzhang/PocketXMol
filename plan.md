# stage_1：冻结 PocketXMol 的 Test-Time Scaling 实验计划

状态：计划；不因写入本文自动启动实验。  
分支：`codex/test-time-scaling`。  
预计周期：3–5 个工作日，包含运行、方法适配和分析。  
资源上限：单卡累计 48 GPU-hours；到达上限先汇总，不自动扩大搜索。  
范围：原始 teacher、SBDD、现有 CSD 100 个口袋；不训练、不修改 checkpoint。

## 1. Stage 1 要回答的问题

给冻结的 PocketXMol 更多推理预算，预算应该用于更多去噪步骤、更多独立候选，还是对已有候选进行修正／轨迹筛选？

本阶段交付一条可复现的质量—计算量曲线，并确定值得在 stage_2 研究的计算分配机制。优先检验现有方法，暂不把内部 message-passing looping 作为主方法，也不扩大到其他 backbone。

需要区分三个结论：

1. **候选覆盖**：更多独立样本是否增加找到好分子的机会？
2. **选择能力**：置信度或物理信号能否可靠识别好样本？
3. **生成改善**：refinement／粒子重采样是否在相同预算下优于独立采样加筛选？

## 2. 已有证据与待验证假设

已完成的 20-pocket pilot 和 100-pocket 历史复评见 `20261011_test_time_scaling.md`。

| 观察 | 本阶段验证方式 |
|---|---|
| Validity 较高，但仍有几何、构象能量和口袋兼容性问题 | 使用联合物理指标及逐项失败统计 |
| 物理筛选 Best-of-N 有收益 | 扩展至全部口袋、多个独立 seed、N=1/2/4/8 |
| 置信度收益不随 N 单调增加 | 比较排序质量、置信度校准和候选池中的可达上限 |
| 200/400 步和原生 refinement 在小 pilot 中没有优势 | 做相同实际 NFE 的多 seed 对照；不提前宣称方法无效 |
| 几何筛选后仍有能量不合格样本 | 将能量作为独立评估，检查收益是否延伸到未用于筛选的指标 |

历史 100-pocket first-candidate 的物理合格率约 77%，联合物理＋Vina 达标产出率约 25%；几何筛选 Best-of-4 分别约 83%、35%。这些数字用于说明 headroom，不作为新实验的配对基线。

## 3. 指标与统计口径：长实验开始前冻结

### 两个共同主指标

**P_phys：物理合格率**

```text
sanitizable
AND single_fragment
AND five_geometry_checks_pass
AND pocket_no_clash
AND energy_pass
```

五项几何检查：键长、键角、内部碰撞、芳香环平面性、双键平面性。沿用已安装 PoseBusters `mol` 配置；构象能量采用其 50-conformer ensemble 和 energy-ratio threshold=100；口袋碰撞采用仓库现有 heavy-atom 检查、cutoff=0.75。

这是明确定义的联合物理代理指标，不称为完整 PoseBusters pass，也不代表实验验证的分子可用性。

**P_joint：物理合格且原位 Vina ≤ −7 的产出率**

```text
P_phys_pass AND as_generated_vina_score <= -7
```

−7 是固定的探索性打分阈值。主结果不使用 minimize／redock 后的坐标，不在看到结果后调整阈值。另报告 −6／−8 阈值的敏感性曲线，不能据此替换主指标。

### 必须同时报告的指标

- Validity、complete，以及几何检查、能量检查、口袋碰撞的逐项通过率。
- 在 complete 样本中的几何／能量通过率，区分化学重建和三维结构的贡献。
- Energy ratio 中位数、分位数和计算 coverage；缺失值单列原因。
- 原位 Vina 均值、中位数、分位数、coverage；同时报告 heavy-atom count 和 ligand efficiency，排查靠增大分子获得低分。
- QED、SMILES／scaffold 多样性、每口袋候选池内的重复率和 fingerprint diversity。
- 口袋碰撞数、单位重原子碰撞数，以及采样失败、价态失败、芳香键失败的数量。
- 实际 graph-NFE、生成时间、选择器时间、verifier 时间、每单位搜索时间的合格输出数。

所有主指标的分母均为全部计划输出：包括 bad、重建失败、无合格候选的口袋。不能删除难口袋，不能通过额外补采样改变预算。缺失评估在联合通过率中计失败，同时报告 coverage；系统性评估故障先修复后比较。

原评估中的参考配体身份／`stereo` 不作为 de-novo SBDD 优化目标。SDF 被标为 2D 但有非零 Z 坐标时应读取真实坐标；不得静默重新 embedding 后代替原生成构象。

### 独立评估与选择偏差

默认几何筛选器不使用能量或 Vina，因此这两项是独立于该排序器的评估信号。Vina 排序的 Vina 收益是优化目标上的收益，不作为独立验证。若探索能量排序，应单列为 energy-guided 方法，明确能量不再是独立指标。

### 统计

- 固定 5 个基础 seed：`2026, 12026, 22026, 32026, 42026`；候选 r 使用基础 seed+r，避免不同批次候选流重叠。
- 相同口袋、基础 seed、预算构成配对单位；共享 seed 不等于不同采样方法具有相同噪声轨迹。
- 先在每口袋内平均各 seed，再比较口袋间结果。不能把 100×5 次运行当作 500 个独立靶标。
- 主比较报告配对差值及按口袋聚类的 bootstrap 95% CI，至少 5000 次重采样；保留所有失败记录。
- 重复候选和嵌套 N 的结果相关；不把各个 N 的比较当成独立重复证据。
- 100 个现有口袋中前 20 个作为开发子集，其余 80 个作为预留确认子集。历史汇总已涵盖这些口袋，不能声称这 80 个是从未见过的独立测试集。

## 4. 计算预算与对照矩阵

主预算为每个最终输出 `B ∈ {100, 200, 400, 800}` 次 graph-NFE。选择器 CPU／GPU 开销另计，因此相同 NFE 不等于相同总运行成本。

| 方法 | 配置 | 输出与预算 |
|---|---|---|
| 更多步骤 | 原生 no-AR，100/200/400/800 steps | 每口袋、每 seed 输出 1 个候选 |
| 独立候选池 | 每候选 100 steps，最多 8 个候选 | 同一个池取前 N=1/2/4/8；B=100N |
| 随机选择 | 从前 N 个候选固定 seed 随机选择 | 对照覆盖效应；不看质量指标 |
| 置信度 Best-of-N | 仓库 trajectory confidence | 输出 1 个；不额外补采样 |
| 几何＋置信度 Best-of-N | 沿用 pilot 的排序定义 | 输出 1 个；计入所有被打分候选的物理 verifier 开销 |
| Vina Best-of-N | qualified-first，再按 Vina 排序 | 搜索参考线；包含 receptor preparation／Vina 成本 |
| 原生局部 refinement | `base.yml`，额外轮数 1/3/7 | 预计 B=200/400/800；以实测 NFE 为准 |

排序定义在首次长实验前写入 manifest。随机选择、置信度选择不预先过滤无效分子；几何和 Vina 选择的过滤规则显式记录。无候选满足门槛时仍输出预定义 fallback，失败计入分母。

可以计算 candidate-pool oracle，例如“池中至少存在一个物理合格候选”的比例，以及所选候选相对 oracle 的差距。Oracle 仅用于定位覆盖／选择瓶颈，不算可部署方法。

仅在 B=800 仍有明确上升趋势时，追加 N=16／B=1600 的小规模扩展。不能只给搜索方法更大预算而缺少相应步数／refinement 对照。

## 5. 分批执行

### Batch 0：协议和运行器补齐，约半天

已有脚本可以采样、评估和计数，但 `run_test_time_scaling.sh` 仍固定 seed=2026、最多 4 个候选和 3 轮 refinement；不能直接作为正式多 seed runner。

长实验前完成：

1. 保存 100 个口袋 ID 列表及开发／确认子集；支持指定口袋列表和 seed。
2. 长实验 driver 支持 steps、候选数、refinement 轮数、已完成任务跳过、失败任务单独重试。
3. 增加随机选择、候选池 oracle、diversity、逐口袋统计和配对 CI。
4. 保存 checkpoint/config/code hash、环境版本、GPU、batch size、全部实际调用计数和计时口径。
5. 检查能量 ensemble 的随机性并固定评估随机源；若组件不能完全固定，记录其行为，复查少量样本对最终 pass 的敏感性。
6. 正式 runtime 比较使用相同 batch size，模型 warm-up 后计时。独立记录启动、数据读取、receptor preparation 和 verifier 时间；不能将 pilot 的摊销时间直接称为生产延迟。
7. 用 2–5 个口袋检查普通采样、N=2 选择和一次 refinement；验证分母、NFE、缓存协议和无合格候选时的行为。

入口已有：`scripts/sample_test_time_scaling.py`、`scripts/evaluate_test_time_scaling.py`、`scripts/evaluate_tts_energy.py`、`scripts/summarize_test_time_scaling.py`。

### Batch 1：完整基线曲线，优先跑过夜

对全部 100 个口袋、5 个基础 seed 跑第 4 节矩阵。

每个 seed 的实际生成任务：

- 100 steps × 8 个候选 × 100 个口袋；同一个池产生所有 N 的选择结果。
- 200／400／800 steps × 各 1 个候选 × 100 个口袋；100-step 基线复用候选池的第一个候选。
- 原生 refinement 额外轮数 1／3／7 × 各 1 个候选 × 100 个口袋。

预计共 7000 个新候选、约 180 万次 graph-NFE。依据 pilot，采样约 6–12 A100 GPU-hours；端到端还需评估和文件读取时间。这是规划估计，第一批实际计时后更新剩余预算。

执行顺序：先全部 seed 的 B≤400，再跑 B=800，避免先在单个 seed 上耗尽预算。每完成一个 seed 汇总，但不根据单个 seed 的结果更改阈值或删减不利对照。

Batch 1 必须给出：两个主指标的预算曲线、组件失败分布、confidence 与物理／Vina 的相关性、oracle-gap、以及计入 verifier 的时间曲线。

### Batch 2：现有 TTS 思路的分子适配，1–2 天

仅在开发 20 个口袋上开发，基础 seed 先用前 3 个。候选方法最多保留两类。

**A. FK 风格粒子重采样**

- 在原生 PocketXMol 采样链中保持同口袋的 P=4／8 个粒子，生成过程在指定位置重采样。
- 先用当前 clean prediction 上的低成本几何／口袋信号，不使用能量或 Vina；禁止把带噪状态直接当最终 SDF 打分。
- 第一轮只检查两个重采样间隔（20／50 次模型调用）和两个势强度；记录 reward、权重、ESS、祖先数量与粒子多样性。
- 差分势要连同父粒子的 reward 历史一起复制，使用稳定的 log-weight 归一化；全体打分失败时使用明确的均匀 fallback。
- 必须复制完整的 node/edge/position、口袋条件和图结构元数据，禁止混合不同口袋。若需固定各粒子原子数，相同原子数策略也用于对应 Best-of-N 对照。
- P×100 的模型预算匹配 Best-of-P；中间 verifier 调用全部计入总成本。
- 这是原生采样链上的 FK 风格适配；在核对转移与势函数前，不声称严格复现原论文算法。

**B. Iterative partial refinement 适配**

- 原生 refinement 已是基线；新增一个更可解释的部分重加噪／重生成对照。
- 先比较随机局部选择与置信度局部选择，固定选择比例 25%，固定一次 refinement 的噪声起点和回程步数。
- 明确更新坐标、原子类型、哪些内部／跨区域键；未选择区域的保持规则必须可检查。
- 优先使用 teacher 训练支持的 mixed-noise／maskfill 路径。普通 no-AR 模型的固定标记和 projector 不保证自动保持上下文，不能只改 fixed mask 就假定实现了局部修复。
- 先比较 1／3 轮；记录改动区域、修复／损害的变量、价态和碰撞变化。将实际调用数匹配同预算的更多步骤和 Best-of-N。
- 文档明确与原生 refinement 的区别；不将简单重复采样或同一策略改名当成新方法。

两类方法先做状态保持和口袋隔离检查，再做质量对照。若基础基线已经耗尽预算，优先完成一类适配，不同时扩展多种 reward。

### Batch 3：冻结配置后确认，约一天

开发阶段最多选出 2 个配置，先写出配置与选择依据，再在预留 80 个口袋、全部 5 个基础 seed 上确认。

- 每个新方法至少匹配 B=400 和一个相邻预算；优先 B=800，必要时选择 B=200。
- 同时保留对应 Best-of-N、更多步骤和原生 refinement；这些基线结果复用 Batch 1，不重新搜索。
- 选择器不使用的能量／Vina 是主要独立检查；报告分子大小、diversity、coverage 的变化。
- 对成功修复和新引入错误分别取样，检查收益发生在坐标、原子类型还是键分配。
- 将开发子集与确认子集分开报告，另给全 100 个口袋的汇总。

如果新方法在开发集已明显无优势，也完成其最小确认或记录失败机制；不通过继续调参拖延 stage_1 结束。

## 6. GPU、调度与长期运行

GPU 任务必须通过 `ssh gpuXX` 在 GPU 节点运行。共享文件系统和用户环境可复用，但节点的 CUDA／驱动／可用 GPU 需要重新检查。

当前可参考 gpu35、A100 和既有 Slurm 分配；不能硬编码历史作业仍有效，也不能因 `nvidia-smi` 空闲就使用未分配 GPU。

执行前检查有效 Slurm allocation，再通过该作业的 `srun` 获取正确 CUDA_VISIBLE_DEVICES。SSH 系统配置权限异常时沿用已验证的 `ssh -F /dev/null`。

```bash
ssh -F /dev/null gpu35
cd /shared/healthinfolab/phz24002/PocketXMol
# STAGE1_JOB_ID 必须是当前有效、属于本用户的分配。
srun --jobid="$STAGE1_JOB_ID" --overlap -n 1 \
  /shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/bin/python \
  scripts/sample_test_time_scaling.py \
  --outdir outputs_test_time_scaling/stage_1/base100_n8_s2026 \
  --steps 100 --repeats 8 --num_pockets 100 --seed 2026 --batch_size 4
```

此命令是首个候选池的当前可用入口，不是尚未实现的完整 stage_1 driver。物理评估使用 reed_molecule，Vina 使用 pxm_vina；不为本计划修改共享环境。

- 默认一张已分配 GPU，batch size=4；CPU workers 不超过作业分配。并行 GPU 批次必须使用额外有效分配。
- 每个采样／评估批次独立日志、独立输出目录；checkpoint／配置／原始结果不覆盖。
- 无需保存完整轨迹；仅保存可解释诊断子集。保留全部 SDF、gen_info、候选评分和方法选择表。
- 资源上限 48 GPU-hours；结果目录超过 20 GB 时先清理可重建的临时文件和未要求的轨迹，不删除原始候选或失败记录。
- 每个批次完成更新进度和剩余预算；不因运行时间长而重复启动已有任务。

### 失败与恢复规则

- SSH／Slurm 无分配：更换有有效分配的节点或等待调度，不改用登录节点 GPU。
- OOM：仅重试失败批次，降低 batch size，并在 runtime 比较中重新统一 batch size；不更改口袋或候选数量。
- 评估失败：保留 SDF 和错误明细；先区分化学失败与环境故障，再重跑相应 evaluator，禁止重新生成来替换失败样本。
- 中断：依据 manifest 和 DONE 状态恢复；现有 sampler 不支持目录内逐 batch 自动续跑，driver 完成前只允许完整新任务目录重试，旧目录保留并标记 abandoned。
- 缓存复用要求候选、checkpoint、评估配置一致；参数变化必须生成新版本，不能混用旧阈值结果。
- 所有方法无新收益时仍按第 7 节结束并交付负结果。

## 7. 推进／停止标准

Stage 1 的成功首先是完成可靠比较，不要求每一种 TTS 方法都胜出。

**进入 stage_2 的条件**：在预留确认子集、相同实际 NFE 且计入 verifier 成本后，至少出现以下一种结果：

1. 新方法相对对应几何 Best-of-N，在 P_phys 或 P_joint 上提升至少 5 个百分点，配对 CI 下界大于 0；另一主指标没有超过 2 个百分点的点估计退化。
2. 与强基线达到相近质量（两个主指标点估计差距不超过 2 个百分点，CI 如实报告），搜索成本减少至少 20%，且重复计时能确认成本优势。
3. 发现可重复的变量／噪声阶段差异，有明确修复对象和至少一个能验证它的有效对照，足以支持 stage_2 的内部计算分配实验。

额外 guardrails：validity／complete 不下降超过 3 个百分点；不能仅由分子大小增加、coverage 下降或多样性明显坍缩解释收益。触发 guardrail 时先给出分层分析，再决定是否继续，不能只用一个均值掩盖退化。

**停止扩大某个方法**：开发子集的多个 seed 均无优势，或收益完全来自候选数／更昂贵筛选，或新增计算持续引入非法价态、粒子坍缩和口袋碰撞。保留最小确认与失败诊断，不无限调参。

**Stage 1 结束条件**：核心基线曲线完成；所选适配完成确认或明确停止；统计、计时、失败模式和配置均可复查。即使只有 Best-of-N 有效，也将其作为确定的基线交付。

## 8. 结果目录与交付

```text
outputs_test_time_scaling/stage_1/
  protocol.json                 # 指标、预算、seed、口袋列表和配置版本
  progress.csv                  # 任务状态、耗时、NFE、失败和重试
  <method>_<budget>_<seed>/      # SDF、gen_info、run.json、评估／选择结果
  tables/                       # 逐口袋结果、配对差值、coverage、oracle gap
  figures/                      # 两个主指标的 NFE／时间曲线及组件失败分析
  report.md                     # 结论、边界、负结果、stage_2 建议
```

最终交付：

- 可复跑的批量 runner、冻结协议和所有方法的实际配置。
- P_phys／P_joint 对 NFE 和搜索成本的曲线，含配对 CI。
- confidence／几何／Vina 排序及 candidate-pool oracle 的比较。
- 几何、能量、口袋碰撞、价态／键级和 diversity 的失败分析。
- 明确说明 stage_2 应研究候选搜索、局部修复、还是内部异质变量计算分配；若证据不足，列出一个最小验证实验，不直接扩张 backbone。

参考方法资料：[FK steering](https://proceedings.mlr.press/v267/singhal25b.html)、[IPR](https://arxiv.org/abs/2605.19317)、[PoseBusters](https://posebusters.readthedocs.io/en/latest/)。
