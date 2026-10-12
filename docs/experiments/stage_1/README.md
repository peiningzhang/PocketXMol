# Stage 1：PocketXMol Test-Time Scaling

本目录归档 2026-10-11 的实验结果与讨论。任务是 **CrossDocked 上的 SBDD**：给定蛋白口袋，生成原子类型、键与三维坐标。仓库数据标识 `csd` 指 CrossDocked，见 [数据说明](../../data_processing.md#crossdocked-csd)。

模型为原始 PocketXMol teacher，未训练或修改 checkpoint。核心比较覆盖 100 个口袋、5 个 seed、7000 个候选；新适配在开发20口袋选参、预留80口袋确认。历史分析已涵盖这些口袋，不能宣称新的外部测试泛化。

## 阅读入口

- [实验计划](../../../plan.md)：stage_1 的矩阵、指标、预算和停止条件。
- [最终报告](report.md)：完整结果、配对置信区间、成本与下一阶段判断。
- [扩展指标表](metrics_overview.md)：Validity、几何、能量、口袋碰撞、Vina、QED与大小。
- [讨论记录](discussion.md)：Best-of-N 如何选、FK 是什么、能量指标是什么。
- [方法与实现边界](method_notes.md)、[原论文资料](method_references.md)。
- [复跑与复查入口](runbook.md)、[冻结协议](protocol.json)。
- [预算统计 CSV](tables/budget_curves.csv)、[配对比较](tables/paired_comparisons.csv)、[预算增量](tables/paired_budget_differences.csv)。
- [质量—计算量图](figures/all_nfe_curves.png)、[统一计时图](figures/confirmation_controlled_time_curves.png)。

## 主要结果

几何 Best-of-4 相对原始单候选，在4倍模型调用预算下，将 P_phys 从72.0%提高到89.0%，P_joint从19.6%提高到33.0%。这是更多候选加筛选的收益，不是模型参数或单候选生成分布改善的证明。

几何 Best-of-8 的 P_phys/P_joint 为90.4%/35.0%；相比 Best-of-4 的增益分别只有1.4/2.0个百分点，配对CI包括零。当前结果不支持继续扩大到 N16。

FK 相比单候选有效，能量指标也有较好表现；但相比同预算几何 Best-of-N，未得到可靠的整体优势。当前局部全变量重生成明显损害化学合法性。保留几何 Best-of-N 强基线，不将这轮 FK／局部适配称为新的有效算法突破。

## 归档范围

GitHub 归档包括正式实验与分析代码、结果表、图、协议、选择 manifest、修复例子和完成核验。约24 MiB的结果文件不含模型权重、完整原始 SDF 池、运行临时文件或完整历史源码快照。它们在原工作区 `outputs_test_time_scaling/stage_1/` 保留。

运行脚本中的本地环境／数据路径应按机器调整。GPU任务通过 SSH 进入GPU节点；历史作业ID和运行时资源授权不是其他会话的有效分配。复跑前检查当前调度与资源权限，不直接复用旧GPU状态。

结果文件来源哈希见 [artifact_manifest.json](artifact_manifest.json)，冻结代码哈希见 [源文件记录](source_hashes.json)。表中的均值须结合coverage读取；通过率中的缺失计失败。完整语义核验见 [completion_evidence_review.json](completion_evidence_review.json)。
