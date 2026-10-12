# 局部修复的具体例子

以下是保存的开发诊断子集中的相邻轮次，不是确认80口袋的统计结论。初始→当前的累计指标另保留在 CSV 中；本文使用前一轮→当前轮指标。2D 图片展示类型和键结构，3D 坐标变化以表中位移及原始 SDF 为准。

## 物理修复

运行 `partial_confidence_r3_s12026`，口袋 `csd_100001`，seed=12026，相邻轮次 1→2。选中图内原子编号：[5, 7, 9]。

[前一轮 SDF](/shared/healthinfolab/phz24002/PocketXMol/outputs_test_time_scaling/stage_1/development_conditioning_v2/partial_confidence_r3_s12026/diagnostic_states/round1/SDF/1.sdf) · [当前轮 SDF](/shared/healthinfolab/phz24002/PocketXMol/outputs_test_time_scaling/stage_1/development_conditioning_v2/partial_confidence_r3_s12026/diagnostic_states/round2/SDF/1.sdf)

| 指标 | 前一轮 | 当前轮 |
|---|---:|---:|
| P_phys | False | True |
| P_joint | False | False |
| valid | True | True |
| complete | True | True |
| geometry_pass | True | True |
| energy_pass | False | True |
| pocket_no_clash | True | True |
| n_atoms | 12 | 12 |
| energy_ratio | 194.85491288814376 | 20.48676298678341 |
| vina_score | -4.267 | -3.875 |

原始模型图的原子类型变化 2 处、半边类型变化 8 处；全部原子坐标 RMS 位移 2.030 Å，选中原子的 SDF RMS 位移 4.060 Å。未选坐标最大位移 0.000 Å；原始固定上下文与口袋断言均为 True/True。

SDF 原子类型变化编号：[5, 7]。未选内部连接关系增删：0。重构后的键级/芳香性变化与原始模型半边更新分别记录，不能混同。

规范含立体信息 SMILES 是否一致：False。

规范 SMILES 改变；前后能量比各自对应各自分子的50构象 ensemble。

![物理修复](/shared/healthinfolab/phz24002/PocketXMol/outputs_test_time_scaling/stage_1/repair_examples/P_phys_round_repair.svg)

## 新增物理错误

运行 `partial_confidence_r1_s12026`，口袋 `csd_100000`，seed=12026，相邻轮次 0→1。选中图内原子编号：[1, 2, 4, 5, 6, 9, 18, 19]。

[前一轮 SDF](/shared/healthinfolab/phz24002/PocketXMol/outputs_test_time_scaling/stage_1/development_conditioning_v2/partial_confidence_r1_s12026/diagnostic_states/round0/SDF/0.sdf) · [当前轮 SDF](/shared/healthinfolab/phz24002/PocketXMol/outputs_test_time_scaling/stage_1/development_conditioning_v2/partial_confidence_r1_s12026/diagnostic_states/round1/SDF/0.sdf)

| 指标 | 前一轮 | 当前轮 |
|---|---:|---:|
| P_phys | True | False |
| P_joint | True | False |
| valid | True | True |
| complete | True | True |
| geometry_pass | True | False |
| energy_pass | True | False |
| pocket_no_clash | True | True |
| n_atoms | 32 | 32 |
| energy_ratio | 13.251976344385348 | 189.9150603311228 |
| vina_score | -7.627 | -7.303 |

原始模型图的原子类型变化 5 处、半边类型变化 23 处；全部原子坐标 RMS 位移 3.711 Å，选中原子的 SDF RMS 位移 7.422 Å。未选坐标最大位移 0.000 Å；原始固定上下文与口袋断言均为 True/True。

SDF 原子类型变化编号：[2, 4, 6, 9, 19]。未选内部连接关系增删：0。重构后的键级/芳香性变化与原始模型半边更新分别记录，不能混同。

规范含立体信息 SMILES 是否一致：False。

规范 SMILES 改变；前后能量比各自对应各自分子的50构象 ensemble。

![新增物理错误](/shared/healthinfolab/phz24002/PocketXMol/outputs_test_time_scaling/stage_1/repair_examples/P_phys_round_damage.svg)

## 新增化学或重构失效

运行 `partial_confidence_r3_s2026`，口袋 `csd_100001`，seed=2026，相邻轮次 1→2。选中图内原子编号：[1, 10, 12, 13]。

[前一轮 SDF](/shared/healthinfolab/phz24002/PocketXMol/outputs_test_time_scaling/stage_1/development_conditioning_v2/partial_confidence_r3_s2026/diagnostic_states/round1/SDF/1.sdf) · [当前轮 SDF](/shared/healthinfolab/phz24002/PocketXMol/outputs_test_time_scaling/stage_1/development_conditioning_v2/partial_confidence_r3_s2026/diagnostic_states/round2/SDF/1-bad.sdf)

| 指标 | 前一轮 | 当前轮 |
|---|---:|---:|
| P_phys | True | False |
| P_joint | False | False |
| valid | True | False |
| complete | True | False |
| geometry_pass | True | False |
| energy_pass | True | False |
| pocket_no_clash | True | False |
| n_atoms | 15 | 15 |
| energy_ratio | 2.9798232605276205 | nan |
| vina_score | -5.076 | nan |

原始模型图的原子类型变化 3 处、半边类型变化 8 处；全部原子坐标 RMS 位移 3.004 Å，选中原子的 SDF RMS 位移 5.818 Å。未选坐标最大位移 0.000 Å；原始固定上下文与口袋断言均为 True/True。

SDF 原子类型变化编号：[1, 10, 13]。未选内部连接关系增删：0。重构后的键级/芳香性变化与原始模型半边更新分别记录，不能混同。

规范含立体信息 SMILES 是否一致：None。

化学清洗失败；原始 SDF 已保留供复查。
