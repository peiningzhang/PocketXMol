# Stage 1 讨论记录

日期：2026-10-11。此文件记录任务解释、指标和结论的澄清；正式数值见 [扩展表](metrics_overview.md) 与 [最终报告](report.md)。

## 任务与指标

这是 CrossDocked 的 structure-based drug design（SBDD），仓库简称 `csd`。生成小分子的类型、连接和坐标，因此合法、连通、物理合理、口袋适配、化学品质和多样性应分别检查。

Validity只能判断化学可解析性。当前 P_phys 要求 sanitizable、single fragment、五项几何检查、无口袋碰撞和能量检查联合通过；P_joint另要求原生成构象 Vina≤−7。两者分母都是全部计划输出，−7是探索性评分阈值，不是实验结合证据。

应结合连续Vina、评分coverage、能量长尾、QED、原子数、ligand efficiency、骨架／指纹多样性解释联合通过率。SA、训练集novelty和参考配体相对表现是可补充方向，本轮尚未提供这些结果，不能在报告中当作已经验证。

## “Best”如何比较

Best-of-N先在同一口袋独立生成N个候选，每候选100步，再按预先固定规则选一个。N1/2/4/8使用同一池的嵌套前缀，不补采样替换失败候选。

几何＋置信度排序依次比较：

1. `qualified`：化学有效、单片段、五项几何通过、无口袋碰撞。
2. `geometry_score`：五项检查通过数量。
3. `complete`：有效且单片段。
4. 模型trajectory confidence，越高越优先。
5. 仍相同时选较早生成的候选。

这是按优先级逐项排序，不是权重加总。**此处 qualified 不含能量；与正式联合物理指标 P_phys 不同。** 几何排序不使用能量、Vina、QED或SA。

置信度排序只用trajectory confidence，不先过滤无效分子。Vina排序先优先qualified，再按Vina从低到高。随机排序不看质量。Pool oracle仅描述池中存在好候选的机会，不是可部署选择器。

## 不使用Vina排序，Vina也会改善吗

本轮观察到：单候选、几何Best-of-4、Best-of-8平均原位Vina分别约−5.54、−6.33、−6.70。几何和碰撞信号可能与Vina相关，但这不能证明仅筛Validity足以获得同样收益。

平均重原子数约22.41、22.85、22.75，ligand efficiency约0.266、0.284、0.313；没有明显依赖分子增大。但Vina coverage约82.8%、91.4%、91.6%，成功打分样本组成有变化，均值不替代全部输出上的联合达标率。

“仅Validity／几何／几何＋置信度”同池消融、相同可打分配对上的Vina比较可以进一步拆分贡献；本轮未完成该Validity-only消融。4→8的连续Vina点估计改善不等于已经得到配对显著性结论。

## 能量指标是什么

使用安装版本PoseBusters的UFF力场：

```text
energy_ratio = 原生成构象的UFF能量 / 同一分子参考构象ensemble的平均UFF能量
```

参考ensemble为50个ETKDG构象，经UFF优化，随机源42、单线程；能量比≤100计通过。保留原生成构象，不能将优化后的结构替换用于正式评价。失败或不能完成ensemble的结果保留原因和coverage。

能量比衡量分子自身构象的合理性；Vina衡量生成配体与受体的评分。两者不是同一量。能量比在不同分子上使用不同参考ensemble，不能解释为不同分子间的绝对能量差，也不能直接解释为同化合物构象应变能。

本轮能量中位数：单候选4.69、几何Best-of-8为3.98、FK B800为3.70。能量通过率分别82.4%、90.8%、92.6%。后续可补同一化学身份下优化前后能量差及按重原子数归一化的应变指标；本轮没有这些补充结果。

## FK是什么、来自哪里

FK是Feynman–Kac steering：同时运行多个生成粒子，在生成途中评分，按权重重采样完整父状态，再继续生成。好的路径可以被复制，差的路径可能淘汰；复制路径后续随机噪声不同，可以分化。区别于所有候选生成完后才选择的Best-of-N。

采用的方法来源为Singhal等人的ICML2025论文 [A General Framework for Inference-time Scaling and Steering of Diffusion Models](https://proceedings.mlr.press/v267/singhal25b.html)，[arXiv预印本](https://arxiv.org/abs/2501.06848)。原文验证图像与文本；本实验是在PocketXMol原生生成链上的分子图适配，不声称原论文直接提供了CrossDocked实现。

本次冻结FK配置使用interval50、strength5。中途根据clean prediction的键长异常、非键接近和口袋接近软惩罚构造差分势；未使用能量或Vina。重采样复制完整图、口袋条件及父reward历史，记录ESS、祖先和多样性。评分失败有uniform fallback。P8粒子由两个物理batch4 forward执行。

## FK算不算有效

相对B100单候选，B800 FK提高P_phys和P_joint，能量检查也有较好表现，**可以称为有效的推理方法**。但该比较同时增加8倍模型调用预算，不能把全部收益归因于FK重采样本身。

与同预算几何Best-of-8相比，全100口袋FK的P_phys91.2% vs90.4%，P_joint34.4% vs35.0%；确认80口袋上P_phys差+2.25pp，95%CI[−1.75,+6.25]pp，P_joint差−0.5pp、CI包含零，计时成本约高5%。能量通过率2个百分点左右的优势只是点估计，尚未单独验证其统计可靠性。

所以“没有可靠增益”特指**尚未可靠超过同预算强基线**，并不是“FK相对原始单候选没有效果”。B400 FK的确认P_joint还比几何基线低5pp，CI[−9.25,−0.75]pp。保留这种分层结论，不将一个能量指标较好等同于整体胜出。

## 下一步定位

确定的基线是几何Best-of-N，2–4个候选有较明确收益，4→8边际收益开始饱和。当前FK和全变量局部重生成没有通过stage_2推进门槛。

若继续研究内部变量计算分配，先做小规模固定化学身份的坐标修复对照，检验能否降低构象问题而不破坏类型、键和口袋适配；不要仅凭少量修复例子扩大训练或backbone。对CrossDocked还应补SA、novelty和口袋相对参考指标，以区分物理合理性与药物设计价值。
