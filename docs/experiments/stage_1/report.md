# 冻结 PocketXMol：Stage 1 最终报告

完成范围：核心 40 个生成任务、7000 个计划候选，按全部gen_info逐行求和的实际 graph-NFE 为1,783,600（包含原生 refinement 提前停止）；两类有界开发；冻结两配置的 80 口袋×5 seed×B400/B800 确认；补齐所选方法的开发五 seed；统一重复计时与独立能量评估。无训练、无 checkpoint 修改。失败候选、废弃整任务重试和旧版条件保持诊断保留。

100 个 CSD 口袋的前 20 个用于开发，其余 80 个预留确认。历史汇总已包含这些口袋，因此是模型已见口袋上的推理预算研究，不能称为全新外部测试。五 seed 为 2026/12026/22026/32026/42026。结论限定于这一 teacher、任务及口袋集合。

P_phys = sanitizable、single fragment、五项几何检查、heavy-atom 口袋碰撞 cutoff0.75、energy ratio≤100 的联合通过率。能量 ensemble50、随机源42、单线程；P_joint 再要求原生成坐标 score-only Vina≤−7。所有比例以全部计划输出为分母，缺失计失败，另报告 coverage。未将 minimization/redocking 坐标替换生成构象。物理指标是代理检查，不代表实验可用性。

统计先在每个口袋内平均 seed，再按口袋 bootstrap5000 次得到 95% CI；比较使用配对差值。同一候选池的 N1/2/4/8 精确嵌套，预算点和重复前缀具有相关性。下表为百分比，括号内为 CI。

## 全 100 口袋

|方法|名义预算|实际NFE均值|P_phys|P_joint|
|---|---|---|---|---|
|confidence|100|100.0|72.0 [67.2, 76.6]|19.6 [14.6, 25.0]|
|confidence|200|200.0|75.8 [72.0, 79.6]|26.2 [20.4, 32.4]|
|confidence|400|400.0|75.8 [71.8, 79.8]|29.0 [23.0, 35.2]|
|confidence|800|800.0|78.4 [74.2, 82.4]|30.6 [24.4, 37.2]|
|fk_i50_b5|400|400.0|90.4 [87.4, 93.2]|28.2 [21.8, 34.8]|
|fk_i50_b5|800|800.0|91.2 [88.0, 94.0]|34.4 [27.4, 41.6]|
|geometry_confidence|100|100.0|72.0 [67.2, 76.6]|19.6 [14.6, 25.0]|
|geometry_confidence|200|200.0|87.2 [83.4, 90.6]|28.8 [22.2, 35.4]|
|geometry_confidence|400|400.0|89.0 [85.8, 92.0]|33.0 [26.2, 39.8]|
|geometry_confidence|800|800.0|90.4 [87.6, 92.8]|35.0 [28.0, 42.0]|
|more_steps|200|200.0|68.8 [63.4, 73.8]|17.2 [12.2, 22.6]|
|more_steps|400|400.0|69.0 [63.8, 74.0]|13.6 [9.4, 18.2]|
|more_steps|800|800.0|63.8 [58.6, 68.8]|12.2 [8.2, 16.6]|
|native_refinement|200|200.0|74.2 [69.6, 78.8]|20.4 [15.4, 26.0]|
|native_refinement|400|397.6|77.0 [72.0, 81.4]|23.8 [18.2, 29.8]|
|native_refinement|800|769.6|80.4 [75.6, 84.8]|25.8 [20.0, 32.2]|
|partial_confidence|400|400.0|55.2 [50.4, 60.0]|15.0 [11.0, 19.2]|
|partial_confidence|800|800.0|59.6 [54.4, 64.8]|17.6 [12.8, 23.0]|
|random|100|100.0|72.0 [67.2, 76.6]|19.6 [14.6, 25.0]|
|random|200|200.0|74.6 [69.8, 79.2]|21.2 [16.4, 26.4]|
|random|400|400.0|74.0 [69.4, 78.2]|23.4 [18.0, 29.0]|
|random|800|800.0|69.6 [64.6, 74.4]|19.6 [14.6, 24.8]|
|vina_oracle|100|100.0|72.0 [67.2, 76.6]|19.6 [14.6, 25.2]|
|vina_oracle|200|200.0|88.4 [84.8, 91.8]|32.2 [25.4, 39.6]|
|vina_oracle|400|400.0|92.0 [89.4, 94.6]|42.8 [35.0, 50.4]|
|vina_oracle|800|800.0|92.4 [90.0, 94.6]|51.4 [43.8, 59.2]|

## 开发 20 口袋

|方法|名义预算|实际NFE均值|P_phys|P_joint|
|---|---|---|---|---|
|confidence|100|100.0|73.0 [64.0, 82.0]|18.0 [9.0, 28.0]|
|confidence|200|200.0|73.0 [65.0, 81.0]|27.0 [16.0, 38.0]|
|confidence|400|400.0|72.0 [66.0, 79.0]|32.0 [20.0, 45.0]|
|confidence|800|800.0|81.0 [74.0, 87.0]|34.0 [22.0, 46.0]|
|fk_i50_b5|400|400.0|91.0 [84.0, 97.0]|32.0 [18.0, 47.0]|
|fk_i50_b5|800|800.0|87.0 [81.0, 93.0]|38.0 [24.0, 53.0]|
|geometry_confidence|100|100.0|73.0 [63.0, 82.0]|18.0 [9.0, 28.0]|
|geometry_confidence|200|200.0|87.0 [81.0, 93.0]|30.0 [18.0, 42.0]|
|geometry_confidence|400|400.0|86.0 [80.0, 91.0]|36.0 [23.0, 50.0]|
|geometry_confidence|800|800.0|92.0 [86.0, 97.0]|39.0 [26.0, 54.0]|
|more_steps|200|200.0|71.0 [62.0, 80.0]|14.0 [7.0, 22.0]|
|more_steps|400|400.0|75.0 [66.0, 84.0]|13.0 [6.0, 22.0]|
|more_steps|800|800.0|76.0 [67.0, 85.0]|13.0 [7.0, 19.0]|
|native_refinement|200|200.0|80.0 [72.0, 88.0]|20.0 [11.0, 31.0]|
|native_refinement|400|388.0|83.0 [75.0, 90.0]|29.0 [16.0, 44.0]|
|native_refinement|800|780.0|86.0 [79.0, 92.0]|33.0 [20.0, 47.0]|
|partial_confidence|200|200.0|53.3 [40.0, 66.7]|16.7 [6.7, 30.0]|
|partial_confidence|400|400.0|55.0 [46.0, 64.0]|18.0 [9.0, 27.0]|
|partial_confidence|800|800.0|62.0 [52.0, 72.0]|20.0 [9.0, 33.0]|
|random|100|100.0|73.0 [63.0, 82.0]|18.0 [9.0, 28.0]|
|random|200|200.0|70.0 [60.0, 79.0]|22.0 [13.0, 32.0]|
|random|400|400.0|66.0 [55.0, 77.0]|24.0 [14.0, 34.0]|
|random|800|800.0|72.0 [65.0, 80.0]|25.0 [14.0, 38.0]|
|vina_oracle|100|100.0|73.0 [63.0, 82.0]|18.0 [9.0, 27.0]|
|vina_oracle|200|200.0|91.0 [85.0, 96.0]|38.0 [24.0, 54.0]|
|vina_oracle|400|400.0|94.0 [88.0, 98.0]|51.0 [34.0, 68.0]|
|vina_oracle|800|800.0|95.0 [90.0, 99.0]|63.0 [46.0, 79.0]|

## 确认 80 口袋

|方法|名义预算|实际NFE均值|P_phys|P_joint|
|---|---|---|---|---|
|confidence|100|100.0|71.8 [66.2, 77.0]|20.0 [14.2, 26.2]|
|confidence|200|200.0|76.5 [72.2, 80.7]|26.0 [19.0, 33.5]|
|confidence|400|400.0|76.7 [72.0, 81.2]|28.2 [21.5, 35.2]|
|confidence|800|800.0|77.8 [72.5, 82.5]|29.8 [22.5, 37.2]|
|fk_i50_b5|400|400.0|90.2 [86.8, 93.2]|27.3 [20.5, 34.5]|
|fk_i50_b5|800|800.0|92.2 [88.5, 95.3]|33.5 [25.2, 42.0]|
|geometry_confidence|100|100.0|71.8 [66.2, 77.0]|20.0 [14.0, 26.2]|
|geometry_confidence|200|200.0|87.3 [83.0, 91.2]|28.5 [21.0, 36.5]|
|geometry_confidence|400|400.0|89.8 [86.0, 93.2]|32.2 [24.8, 40.2]|
|geometry_confidence|800|800.0|90.0 [87.0, 92.8]|34.0 [26.2, 42.0]|
|more_steps|200|200.0|68.2 [62.0, 74.2]|18.0 [12.0, 24.5]|
|more_steps|400|400.0|67.5 [61.5, 73.2]|13.8 [9.0, 19.0]|
|more_steps|800|800.0|60.8 [54.8, 66.5]|12.0 [7.0, 17.2]|
|native_refinement|200|200.0|72.8 [67.5, 78.0]|20.5 [14.8, 26.7]|
|native_refinement|400|400.0|75.5 [69.5, 80.7]|22.5 [16.2, 29.2]|
|native_refinement|800|767.0|79.0 [73.8, 84.0]|24.0 [17.2, 31.0]|
|partial_confidence|400|400.0|55.2 [49.5, 60.7]|14.2 [9.8, 19.0]|
|partial_confidence|800|800.0|59.0 [53.0, 64.8]|17.0 [11.5, 22.8]|
|random|100|100.0|71.8 [66.2, 77.0]|20.0 [13.8, 26.2]|
|random|200|200.0|75.8 [70.2, 81.0]|21.0 [14.8, 27.3]|
|random|400|400.0|76.0 [71.0, 80.7]|23.2 [17.2, 29.8]|
|random|800|800.0|69.0 [62.8, 74.5]|18.2 [12.8, 24.2]|
|vina_oracle|100|100.0|71.8 [66.2, 77.0]|20.0 [14.0, 26.2]|
|vina_oracle|200|200.0|87.7 [83.2, 91.8]|30.8 [22.7, 38.8]|
|vina_oracle|400|400.0|91.5 [88.2, 94.5]|40.8 [32.5, 49.5]|
|vina_oracle|800|800.0|91.8 [89.0, 94.5]|48.5 [39.8, 57.2]|

## 覆盖、选择和生成改善

几何筛选为本轮强基线：全 100 口袋从 B100 的 P_phys72.0% / P_joint19.6%，升至 B400 的89.0% /33.0%，B800为90.4% /35.0%。增加独立候选并进行几何筛选，比给单条轨迹增加步骤更有效。该结论来自五 seed，历史 pilot 数字未用作配对基线。

B800 池中至少一个合格候选的 oracle 为 P_phys99.4%、P_joint55.2%；几何选择分别为90.4%、35.0%，gap9.0pp/20.2pp；confidence为78.4%/30.6%，gap21.0pp/24.6pp。Vina 排序为92.4%/51.4%，其联合指标收益使用了目标本身，不能当作独立验证。Oracle 是诊断上限，不可部署。

置信度仅是排序信号。跨正式多候选池的平均池内 Spearman 与 P_phys约0.096、P_joint约0.261、Vina约−0.316、energy ratio约−0.172；只统计相关性可定义的池，数量见表。sigmoid 显示值的固定十区间可靠性表未拟合概率模型，不是校准后的成功概率，也不是独立样本验证。

原生 refinement 可能提前停止，名义 B400/B800 的全口袋实际 NFE 均值397.6/769.6。额外回顾性控制逐项匹配全部1500个原生输出的真实 NFE，使用原始 N8 池前 N 个候选，N=原生实际NFE/100，含 N3/5/6/7。原生减匹配几何对照的 P_phys 差值为−13.0/−12.0/−10.0pp，P_joint为−8.4/−9.2/−8.6pp，三个预算各 CI 均低于零。这是依赖观察原生消耗的诊断控制，不是可部署预算策略；完整分层见 [严格匹配表](tables/native_actual_nfe_matched_comparisons.csv)。

B400→B800 几何筛选配对 P_phys+1.4pp [−1.4,+4.4]，P_joint+2.0pp [−0.8,+4.8]。未满足预声明的正 CI 下界门槛，不扩大至 N16/B1600。扩展脚本保留但未运行；该停止决定在完整核心矩阵后作出。

## 冻结适配的确认

开发仅搜索 FK 的2间隔×2强度×P4/P8，以及局部修复随机/低坐标置信度25%×1/3轮。按冻结 B400 规则每类选一配置：FK interval50、strength5；partial_confidence、比例25%、noise_start1。确认不继续调参。旧版局部条件保持不完整的10单元已重做，v1仅列诊断。完整图状态、口袋隔离、差分 reward 父历史复制与 uniform fallback 已核查。

|方法|预算|指标|差值pp|95%CIpp|实际NFE配对比例|
|---|---|---|---|---|---|
|fk_i50_b5|400|P_phys|+0.50|[-3.50,+4.50]|1.000|
|fk_i50_b5|400|P_joint|-5.00|[-9.25,-0.75]|1.000|
|fk_i50_b5|400|valid|+0.00|[+0.00,+0.00]|1.000|
|fk_i50_b5|400|complete|-0.25|[-0.75,+0.00]|1.000|
|fk_i50_b5|800|P_phys|+2.25|[-1.75,+6.25]|1.000|
|fk_i50_b5|800|P_joint|-0.50|[-5.50,+4.50]|1.000|
|fk_i50_b5|800|valid|+0.00|[+0.00,+0.00]|1.000|
|fk_i50_b5|800|complete|+0.00|[+0.00,+0.00]|1.000|
|partial_confidence|400|P_phys|-34.50|[-40.75,-28.00]|1.000|
|partial_confidence|400|P_joint|-18.00|[-23.75,-13.00]|1.000|
|partial_confidence|400|valid|-14.00|[-17.75,-10.25]|1.000|
|partial_confidence|400|complete|-19.25|[-23.25,-15.25]|1.000|
|partial_confidence|800|P_phys|-31.00|[-37.00,-25.25]|1.000|
|partial_confidence|800|P_joint|-17.00|[-22.25,-12.25]|1.000|
|partial_confidence|800|valid|-15.50|[-19.75,-11.50]|1.000|
|partial_confidence|800|complete|-21.00|[-26.00,-16.25]|1.000|

局部修复不是保持化学身份的构象优化；坐标、原子类型与选中区域相关键均可改变。原始未选状态逐步保持，最终重建的芳香性/键级归一化另行核对。96个相邻状态转换中87对均可化学解析，40对存在未选区域归一化化学键级变化；未选连接、类型、坐标变化为0。序列化 Kekulé 变化不可直接等同于化学键变化。

保存的物理修复例子中 ratio194.85→20.49、P_phys失败→通过，但 Vina−4.267→−3.875、P_joint仍失败；该例化学身份发生变化，每个图有自己的 ensemble，不能解释为同化合物绝对能量降低。成功与损害例子及未解析 SDF 见 [修复与损害](repair_examples.md)。少数修复不能抵消确认中广泛的 validity/complete 下降。

## Coverage、分子大小与多样性

|method|budget|valid|complete|n_atoms_mean|qed_mean|ligand_efficiency_mean|energy_ratio_coverage|vina_score_coverage|unique_smiles_fraction_valid|acyclic_fraction_complete|
|---|---|---|---|---|---|---|---|---|---|---|
|fk_i50_b5|400|1.0|0.9975|22.1775|0.5216|0.2827|0.99|0.915|0.9825|0.0301|
|fk_i50_b5|800|1.0|1.0|23.095|0.5197|0.2848|0.9925|0.91|1.0|0.005|
|geometry_confidence|400|1.0|1.0|22.77|0.5151|0.2787|0.9975|0.9075|0.995|0.025|
|geometry_confidence|800|1.0|1.0|22.8175|0.5171|0.3127|1.0|0.8975|0.99|0.015|
|partial_confidence|400|0.86|0.8075|22.095|0.5282|0.2332|0.85|0.7525|0.9971|0.0774|
|partial_confidence|800|0.845|0.79|22.095|0.5278|0.2577|0.8425|0.7575|1.0|0.0538|
|vina_oracle|400|1.0|1.0|24.1675|0.5162|0.2817|0.995|0.9225|1.0|0.02|
|vina_oracle|800|1.0|1.0|25.52|0.4983|0.3109|0.9975|0.9225|1.0|0.01|

FK 与同 seed 独立池使用相同 pocket_atoms_based 可变原子数先验，未另设固定粒子大小；全部38个 FK 运行的配置已核对。mean=0.05924407×pocket_atoms+0.1478300335619629，std6.303689641435993，min5。粒子重采样复制完整图，允许异质原子数；相同先验不保证重采样后大小分布一致。固定大小分箱≤20/21–30/31–40/41–50/>50和未知单列，详见 [分层质量](tables/size_stratified_quality.csv)。

Vina 排序的平均原子数随预算增加，必须结合 ligand efficiency 与大小分层解释其目标收益。SMILES唯一比例按有效输出计算；空 Murcko scaffold 表示无环图，单列 acyclic_fraction_complete，不将其混同于丢失样本。池内重复率和 fingerprint diversity 按实际候选集合计算；全失败池的重复率是未知而非100%。FK ESS/祖先轨迹可查 [ESS表](tables/fk_ancestor_ess_summary.csv)，重采样引入的祖先集中与最终分子重复不是同一量。

冻结 FK 与独立候选池的配对 guardrail（确认80口袋×5seed，每口袋先平均seed）：

|budget|metric|FK_mean|independent_mean|delta|ci_low|ci_high|defined_pairs|planned_pairs|defined_pockets|
|---|---|---|---|---|---|---|---|---|---|
|400|valid_candidate_coverage|0.92188|0.91625|0.00562|-0.01312|0.02438|400|400|80|
|400|duplicate_fraction|0.00708|0.00062|0.00646|0.00188|0.01125|400|400|80|
|400|fingerprint_diversity|0.84818|0.86009|-0.01192|-0.01877|-0.00537|395|400|80|
|400|mean_candidate_heavy_atoms|21.64312|22.06062|-0.4175|-0.84752|0.03691|400|400|80|
|400|atom_count_coverage|1.0|1.0|0.0|0.0|0.0|400|400|80|
|800|valid_candidate_coverage|0.91031|0.91375|-0.00344|-0.01625|0.00969|400|400|80|
|800|duplicate_fraction|0.00531|0.00098|0.00433|0.00129|0.00772|400|400|80|
|800|fingerprint_diversity|0.85871|0.86145|-0.00275|-0.00689|0.00156|400|400|80|
|800|mean_candidate_heavy_atoms|22.83531|22.17875|0.65656|0.3403|0.96782|400|400|80|
|800|atom_count_coverage|1.0|1.0|0.0|0.0|0.0|400|400|80|

这里比较整个实际粒子池，区别于上一表的最终所选输出。B400 fingerprint diversity差约−0.012，B800约−0.0027；最终分子重复率虽增加，绝对水平仍低，未观察到最终SMILES的严重坍缩。B800候选平均原子数增加约0.66，CI高于零，故不能以同先验断言输出大小相同，必须保留分层结果。未定义的fingerprint值只在该条件指标中保留缺失，主质量与候选coverage分母不删；defined_pairs列标明395/400或400/400，均覆盖80口袋。

## 失败分解与能量尾部

|method|budget|component|pass_all|pass_given_complete|failures|denominator|
|---|---|---|---|---|---|---|
|fk_i50_b5|400|bond_lengths|1.0|1.0|0|400|
|fk_i50_b5|400|bond_angles|0.9975|0.9975|1|400|
|fk_i50_b5|400|internal_steric_clash|1.0|1.0|0|400|
|fk_i50_b5|400|aromatic_ring_flatness|1.0|1.0|0|400|
|fk_i50_b5|400|double_bond_flatness|1.0|1.0|0|400|
|fk_i50_b5|400|energy_pass|0.92|0.9198|32|400|
|fk_i50_b5|400|pocket_no_clash|0.9825|0.985|7|400|
|fk_i50_b5|800|bond_lengths|1.0|1.0|0|400|
|fk_i50_b5|800|bond_angles|1.0|1.0|0|400|
|fk_i50_b5|800|internal_steric_clash|1.0|1.0|0|400|
|fk_i50_b5|800|aromatic_ring_flatness|1.0|1.0|0|400|
|fk_i50_b5|800|double_bond_flatness|1.0|1.0|0|400|
|fk_i50_b5|800|energy_pass|0.94|0.94|24|400|
|fk_i50_b5|800|pocket_no_clash|0.9825|0.9825|7|400|
|geometry_confidence|400|bond_lengths|1.0|1.0|0|400|
|geometry_confidence|400|bond_angles|1.0|1.0|0|400|
|geometry_confidence|400|internal_steric_clash|0.9975|0.9975|1|400|
|geometry_confidence|400|aromatic_ring_flatness|1.0|1.0|0|400|
|geometry_confidence|400|double_bond_flatness|1.0|1.0|0|400|
|geometry_confidence|400|energy_pass|0.915|0.915|34|400|
|geometry_confidence|400|pocket_no_clash|0.985|0.985|6|400|
|geometry_confidence|800|bond_lengths|1.0|1.0|0|400|
|geometry_confidence|800|bond_angles|1.0|1.0|0|400|
|geometry_confidence|800|internal_steric_clash|1.0|1.0|0|400|
|geometry_confidence|800|aromatic_ring_flatness|1.0|1.0|0|400|
|geometry_confidence|800|double_bond_flatness|1.0|1.0|0|400|
|geometry_confidence|800|energy_pass|0.905|0.905|38|400|
|geometry_confidence|800|pocket_no_clash|0.995|0.995|2|400|
|partial_confidence|400|bond_lengths|0.845|0.9814|62|400|
|partial_confidence|400|bond_angles|0.8175|0.9505|73|400|
|partial_confidence|400|internal_steric_clash|0.845|0.9845|62|400|
|partial_confidence|400|aromatic_ring_flatness|0.8575|0.9969|57|400|
|partial_confidence|400|double_bond_flatness|0.85|0.9907|60|400|
|partial_confidence|400|energy_pass|0.73|0.839|108|400|
|partial_confidence|400|pocket_no_clash|0.765|0.9009|94|400|
|partial_confidence|800|bond_lengths|0.8075|0.9525|77|400|
|partial_confidence|800|bond_angles|0.7825|0.9304|87|400|
|partial_confidence|800|internal_steric_clash|0.825|0.9842|70|400|
|partial_confidence|800|aromatic_ring_flatness|0.845|1.0|62|400|
|partial_confidence|800|double_bond_flatness|0.8425|0.9968|63|400|
|partial_confidence|800|energy_pass|0.7425|0.8797|103|400|
|partial_confidence|800|pocket_no_clash|0.78|0.9399|88|400|

几何/能量 complete 条件通过率与全部计划输出通过率同时报告，避免将化学重建损失归因于三维结构。多个失败组件可能重叠，不能相加当作独立失败总量。采样、价态、芳香键失败数量见 [化学失败](tables/chemical_failure_counts.csv)；逐候选错误、归一化口袋碰撞、能量 warning 和 missing 原因见 [候选表](tables/all_candidates.csv)。

能量 ratio 与 Vina 的均值、中位数、10/90分位数及 coverage、QED/ligand efficiency 分布完整保存在 [预算汇总](tables/budget_curves.csv)。能量有长尾，单个 ensemble 可计算一小时以上；本轮未用事后超时阈值删除这些候选。CPU时间实际推进才保持运行，终止任务仅在检查退出状态、候选哈希与完整协议后恢复 EVALUATED。能量不用于几何/FK/局部排序，评价时间与部署搜索成本分开。

确认输出的能量缺失原因（不是全池候选计数）：

|method|budget|energy_missing_cause|planned_outputs|
|---|---|---|---|
|fk_i50_b5|400|ensemble_or_generated_conformer_energy_unavailable|4|
|fk_i50_b5|800|ensemble_or_generated_conformer_energy_unavailable|3|
|geometry_confidence|400|ensemble_or_generated_conformer_energy_unavailable|1|
|partial_confidence|400|ensemble_or_generated_conformer_energy_unavailable|4|
|partial_confidence|400|invalid input|56|
|partial_confidence|800|ensemble_or_generated_conformer_energy_unavailable|1|
|partial_confidence|800|invalid input|62|

状态保持全量复查覆盖67个正式生成运行、7380条诊断记录；见 [逐运行核验](state_preservation_audit.json)。采样环境中的两条芳香修复超时传播测试与普通/AR采样 trace开关状态、预测、置信度、调用数、RNG等价测试通过，日志在 smoke/*_final.log。评价环境缺少 easydict 的尝试保留为失败日志，最终验证使用原采样环境，未安装或改变共享依赖。

## 搜索成本与计时

|method|budget|mean|median|std|min|max|count|
|---|---|---|---|---|---|---|---|
|fk_i50_b5|400|5.7261|5.7085|0.0353|5.7031|5.7668|3|
|fk_i50_b5|800|12.4659|12.372|0.3042|12.2197|12.8059|3|
|geometry_confidence|400|6.2051|6.0702|0.3163|5.9787|6.5665|3|
|geometry_confidence|800|11.8311|11.7891|0.123|11.7346|11.9695|3|
|more_steps|400|5.2902|5.1441|0.2544|5.1425|5.584|3|
|more_steps|800|10.0801|10.0631|0.0467|10.0442|10.1329|3|
|native_refinement|400|5.7043|4.7549|1.7009|4.6901|7.668|3|
|native_refinement|800|8.6809|9.2671|1.0821|7.4322|9.3434|3|
|partial_confidence|400|5.4711|5.4377|0.1364|5.3544|5.6211|3|
|partial_confidence|800|10.546|10.5593|0.0315|10.51|10.5686|3|
|vina_oracle|400|7.9438|7.841|0.2901|7.7191|8.2713|3|
|vina_oracle|800|14.1585|14.1092|0.1805|14.0078|14.3586|3|

以上为同 GPU39/A100、batch4、8固定确认口袋、三次重复的 trace-off 原型搜索秒/输出。FK P8 是八图粒子总体，模型调用分成两个 batch4；26 个确认、开发补齐、trace-off/trace-on FK 运行的 run.json 调用计数均核实为实际四图 forward（见 tables/fk_physical_batch_audit.json）。搜索包含生成、选择器、几何 verifier，Vina排序还含 receptor准备及原位评分，FK中间 verifier包含在运行成本中；独立能量评价另列，不拿来比较部署搜索延迟。启动、warm-up、数据/重建I/O独立记录，早期未知启动成本保留未知，未补造。时间是批量吞吐摊销值，不能称为单请求生产延迟。

质量来自确认80口袋×5seed，时间来自8口袋×3重复，实际NFE在各自来源表保留；native早停在两个范围不同。普通轨迹记录与共同关闭记录的对照都有三次重复，但GPU非确定性及重建耗时差异使其不能当作纯记录开销因果估计。CPU trace开关状态/RNG/调用数等价检查已通过。

主要图：[实际NFE](figures/all_nfe_curves.pdf)、[全量摊销搜索时间](figures/all_time_curves.pdf)、[统一计时的确认质量](figures/confirmation_controlled_time_curves.pdf)、[组件通过](figures/confirmation_component_pass.pdf)。质量误差为口袋CI，统一计时时间误差为重复trial标准差；配对差值CI见独立表，不由两条边际误差线推断。

## Stage 2 决策

|方法|预算|质量门槛1|成本门槛2|valid/complete守卫|成本比|
|---|---|---|---|---|---|
|fk_i50_b5|400|False|False|True|0.923|
|fk_i50_b5|800|False|False|True|1.054|
|partial_confidence|400|False|False|False|0.882|
|partial_confidence|800|False|False|False|0.891|

门槛1要求确认同实际NFE提升≥5pp且配对CI下界>0、另一主指标退化≤2pp；门槛2要求两主指标点差≤2pp、成本降低≥20%且三个同硬件重复确认优势。validity/complete不得下降>3pp；大小、coverage和多样性须共同解释。数值审计见 [门槛记录](tables/stage2_quantitative_gates.json)。

门槛3尚未得到支持。当前全噪声局部重生成可以偶尔修复，也持续造成类型/键/重建损伤；没有一个已验证有效的变量或噪声阶段控制支持内部异质计算分配。不能只因保存了修复例子就进入 Stage2，或将失败归咎于实现错误后无限调参。

现阶段交付几何 Best-of-N 作为确定基线，停止扩大已测 FK/全噪声局部配置。不直接扩展 backbone、不训练。若要验证内部变量分配，只做一个最小预注册实验：同4个开发口袋×3seed，单轮 B200，固定同父样本与25%局部区域，比较“仅坐标部分重加噪、类型/全部化学连接固定”与“当前全变量部分重生成”，并复用同预算独立N2与200步对照；检查原始状态保持、化学身份、P_phys/P_joint及新增损害。不得先观察确认集再调噪声/比例；该实验是建议，未在本轮执行。

## 复查与资源

全部生成尝试保守累计 15.796 GPU-hours，低于48小时上限。失败/旧版/重试计算均计入，CPU能量不作为GPU生成时间。完成归档时结果目录约119 MB（低于20 GB）；所有原始SDF、gen_info、评分、选择表、错误和诊断保留；没有补采样替换难例。

入口及命令见 [runbook](runbook.md)，冻结协议 [protocol](protocol.json)，配置/代码快照与环境见 source_snapshot*/environment。整个首次池重试保留旧目录；相同seed GPU重跑并非逐位相同，检查见 [重试记录](retry_reproducibility.json)。初始能量线程协议迁移后统一thread1，缓存须核对候选哈希与配置。

方法定位与论文差异见 [方法说明](method_notes.md) 和 [原始资料](method_references.md)。FK为本原生链上的粒子适配，IPR为分子图条件重生成适配，不声称严格复现论文。

完成判定由 [自动必要条件](completion_audit.json) 与逐项语义/图形复查共同组成。自动文件存在检查不能代替结论与guardrail解释。
