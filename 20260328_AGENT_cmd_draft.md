# PocketXMol

```
conda_shared
conda activate PocketXMol
```



- Following  @README.md, download the model weight and run the 4. SBDD (Structure-Based Drug Design) benchemark. 
- If you need GPU and conda environment. Use ssh gpu14 anduse conda environment /shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol

### sample_drug3d

- python scripts/sample_drug3d.py --config_task configs/sample/test/sbdd_csd/base.yml --outdir outputs_test/sbdd_csd

- 数据流概要：测试集 → DataLoader → 扩散采样 (sample_loop3) → 解码 → RDKit 重建 → 按质量分类 (succ/incomp/bad) → 写出 SDF + gen_info.csv

- num_repeats: 100 表示：对同一个测试集整体再跑 100 遍。

  具体含义：

  - SBDD 测试集：100 个 protein pocket（batch_size: 101 时，第一个 batch 会装下这 100 个）

  - 每一轮（i_repeat）：用 test_loader 把 100 个 pocket 都过一遍，为每个 pocket 各生成 1 个分子

  - 100 轮后：每个 pocket 都对应 100 个不同的生成分子

- refine：

  while True 外循环：SBDD 的 is_ar='ar' 时，会反复进行多轮 refine。

  每一轮：跑一次完整的 100 步去噪，所以每一轮都会出现一个 Sampling steps 的 tqdm。

  结束条件：outputs2batch_ar 会把结果分成：

  - part1：已满足置信度阈值的原子 → 视为完成

  - part2：仍需 refine 的原子 → 进入下一轮

    对 part1、part2 分别加噪（sample_noise.py 约 991–1035 行）：

  - part1：part1_pert: small → 只加较轻噪声（level_dict_p1 偏向低噪声）

  - part2：按 level_dict_p2 加完整强噪声，等于把 part2 再「推回」扩散链中较高 t 的状态

  再跑 100 步去噪：

  - part1 在 additional_process 里用 fixed_node 保持，不被覆盖

  - part2 重新走一遍 100 步去噪

- 上述流程是is_ar=True情况下的设定，paper中明确区分了两者，即非AR的生成和AR的生成。

### 一个Naive的想法

**总结实施路径：**

1. 下载 PocketXMol 的预训练权重和部分测试数据集（如 CrossDocked2020 或 PDBbind 的一个小子集）。
2. 在 PyTorch 中将其大部分网络层设置为 `requires_grad = False`。
3. 注入 $E(3)$ 兼容的 LoRA 层。
4. 跑通一个小规模的离线一致性蒸馏 Pipeline，证明在 2-4 步内可以恢复原始模型 1000 步生成质量的 80% 以上。
5. 这就是一个完美的 NeurIPS 投稿基础。

### 当前范式

- 扫描这个项目，我希望知道他们Diffusion到底是怎么定义的？怎么训练的？是eps-prediction?x-prediction？

  | 方面      | 实现                                                         |
  | :-------- | :----------------------------------------------------------- |
  | 前向过程  | x_t = $√(ᾱ_t)·x₀ + √(1−ᾱ_t)·ε$（Gaussian）                   |
  | 预测目标  | x₀（pred_pos, pred_node, pred_halfedge）                     |
  | 训练 loss | MSE(pred_pos, node_pos), CE(pred_node, node_type), CE(pred_halfedge, halfedge_type) |
  | 预测形式  | x-prediction（预测干净数据），不是 eps-prediction            |
  | 采样方式  | 每步预测 x₀，用预测结果更新状态，再对下一步继续加噪—去噪     |

- 当前只使用100步，然后可以用



### 评估

在PocketXMol环境

```
(PocketXMol) [phz24002@gpu40 PocketXMol]$ python evaluate/evaluate_sbdd.py --result_root outputs_test/sbdd_csd --exp_name base_pxm_20260304_002527
```

结果包含stable等指标，但是打印不全面

```
[2026-03-05 14:23:36,443::eval::INFO] Computing validity metrics...
[2026-03-05 14:23:36,445::eval::INFO] Validity : {'validity': 0.9512, 'connectivity': 0.9881202691337259}
```

### 尝试Docking

原来的evaluate的vina模式完全不成功，环境都搭不好。

使用原来的pdb文件并不成功，生成pdbqt会失败，使用AliDiff他们预处理过的。

##### 1) 快速检查（几分钟）

```
python evaluate/evaluate_vina_sdf.py \
  --sdf_dir outputs_test/sbdd_csd/base_pxm_20260304_002527/SDF \
  --gen_info outputs_test/sbdd_csd/base_pxm_20260304_002527/gen_info.csv \
  --split_by_name_path /shared/healthinfolab/phz24002/AliDiff/data/split_by_name.pt \
  --test_set_root /shared/healthinfolab/phz24002/AliDiff/data/test_set \
  --mode score_only \
  --n_workers 16 \
  --max_mols 1000 \
  --quiet
```

Saved 1000 results to outputs_test/sbdd_csd/base_pxm_20260304_002527/vina.csv
  ==vina_score mean: -5.90==
  ==vina_score median: -6.21==
  Failed: 66

##### 2) 全量但相对可接受

```
python evaluate/evaluate_vina_sdf.py \
  --sdf_dir outputs_test/sbdd_csd/base_pxm_20260304_002527/SDF \
  --gen_info outputs_test/sbdd_csd/base_pxm_20260304_002527/gen_info.csv \
  --split_by_name_path /shared/healthinfolab/phz24002/AliDiff/data/split_by_name.pt \
  --test_set_root /shared/healthinfolab/phz24002/AliDiff/data/test_set \
  --mode score_only \
  --n_workers 16 \
  --quiet
```

Saved 10000 results to outputs_test/sbdd_csd/base_pxm_20260304_002527/vina.csv
  vina_score mean: -5.87
  vina_score median: -6.14
  Failed: 672



因此这里的vina score并不太好，但是胜在分子质量高，而且可能只需要100个step（存在争议）

### 重新实现targetdiff标准的evaluation

实现已完成，摘要如下：

`evaluate/evaluate_sdf_standard.py` 已完成

##### 功能概览

- 输入：`--sdf_dir`、`--gen_info`、`--split_by_name_path`、`--test_set_root`，以及可选参数
- 映射：SDF `filename` → `gen_info.data_id` → `split_by_name[test][i]` → `test_set_root` 下的 protein 路径
- 受体：优先使用已存在的 `*_rec.pdbqt`，避免重复 prepare

##### 评估内容（对齐 `eval_split`）

- 化学性质：QED、SA、LogP、Lipinski、环大小比例
- 键长与 pair-distance 的 JSD
- 原子类型分布 JSD
- Vina：默认 `vina_score`（score_only + minimize）

##### NA 字段

- `mol_stable`、`atm_stable`：依赖 trajectory 张量，在 SDF 模式下设为 NA

##### 运行示例

小规模测试（50 个分子）：

```
python evaluate/evaluate_sdf_standard.py   --sdf_dir outputs_test/sbdd_csd/base_pxm_20260304_002527/SDF   --gen_info outputs_test/sbdd_csd/base_pxm_20260304_002527/gen_info.csv   --split_by_name_path /shared/healthinfolab/phz24002/AliDiff/data/split_by_name.pt   --test_set_root /shared/healthinfolab/phz24002/AliDiff/data/test_set   --max_mols 500
```

全量（并行）：

python evaluate/evaluate_sdf_standard.py \

  --sdf_dir ... --gen_info ... --split_by_name_path ... --test_set_root ... \

  --n_workers 4

##### 输出文件

- `eval_results_standard/metrics_sdf_*.pt`：stability、bond_length、all_results、bond_js、pair_js、atom_type_js
- `--save_plot`：pair-distance 直方图

```
[2026-03-06 01:21:51,598::evaluate_sdf_standard::INFO] mol_stable:      NA
[2026-03-06 01:21:51,604::evaluate_sdf_standard::INFO] atm_stable:      NA
[2026-03-06 01:21:51,608::evaluate_sdf_standard::INFO] recon_success:   0.9483
[2026-03-06 01:21:51,612::evaluate_sdf_standard::INFO] eval_success:    0.9328
[2026-03-06 01:21:51,615::evaluate_sdf_standard::INFO] complete:        0.9370
[2026-03-06 01:21:51,698::evaluate_sdf_standard::INFO] JS bond distances:
[2026-03-06 01:21:51,712::evaluate_sdf_standard::INFO] JSD_6-6|4:       0.3875
[2026-03-06 01:21:51,722::evaluate_sdf_standard::INFO] JSD_6-6|1:       0.3333
[2026-03-06 01:21:51,730::evaluate_sdf_standard::INFO] JSD_6-8|1:       0.2763
[2026-03-06 01:21:51,739::evaluate_sdf_standard::INFO] JSD_6-7|1:       0.2743
[2026-03-06 01:21:51,748::evaluate_sdf_standard::INFO] JSD_6-8|2:       0.3768
[2026-03-06 01:21:51,757::evaluate_sdf_standard::INFO] JSD_6-6|2:       0.2761
[2026-03-06 01:21:51,763::evaluate_sdf_standard::INFO] JSD_6-7|4:       0.2027
[2026-03-06 01:21:51,770::evaluate_sdf_standard::INFO] JSD_6-7|2:       0.2955
[2026-03-06 01:21:52,427::evaluate_sdf_standard::INFO] JS pair distances:
[2026-03-06 01:21:52,435::evaluate_sdf_standard::INFO] JSD_CC_2A:       0.3192
[2026-03-06 01:21:52,438::evaluate_sdf_standard::INFO] JSD_All_12A:     0.0865
[2026-03-06 01:21:52,442::evaluate_sdf_standard::INFO] Atom type JS: 0.0782
[2026-03-06 01:21:52,457::evaluate_sdf_standard::INFO] QED:   Mean: 0.520 Median: 0.531
[2026-03-06 01:21:52,468::evaluate_sdf_standard::INFO] SA:    Mean: 0.771 Median: 0.800
[2026-03-06 01:21:52,484::evaluate_sdf_standard::INFO] Vina Score: Mean: -5.869 Median: -6.139
[2026-03-06 01:21:52,489::evaluate_sdf_standard::INFO] Vina Min:   Mean: -6.851 Median: -6.787
[2026-03-06 01:21:52,495::evaluate_sdf_standard::INFO] ring size: 3 ratio: 0.003
[2026-03-06 01:21:52,500::evaluate_sdf_standard::INFO] ring size: 4 ratio: 0.005
[2026-03-06 01:21:52,506::evaluate_sdf_standard::INFO] ring size: 5 ratio: 0.427
[2026-03-06 01:21:52,510::evaluate_sdf_standard::INFO] ring size: 6 ratio: 0.855
[2026-03-06 01:21:52,515::evaluate_sdf_standard::INFO] ring size: 7 ratio: 0.087
[2026-03-06 01:21:52,521::evaluate_sdf_standard::INFO] ring size: 8 ratio: 0.007
[2026-03-06 01:21:52,526::evaluate_sdf_standard::INFO] ring size: 9 ratio: 0.002
```

这里的SA极高，其他指标也并不差。

### 尝试蒸馏当前的模型

```
扫描代码，制定一个计划来基于现有的pretrained model，进行Diffusion的轨迹蒸馏，即自蒸馏得到一个可以快速采样（采样步较少）的模型。
具体来说，需要对连续变量和离散变量分别使用不同的公式。然后，构建的代码应该适合迭代，即每次蒸馏的结果都可以用原来的接口评估，并且需要记录不同采样步的结果。
另外，还需要明确当前的默认状态是AR模式还是非AR模式，一开始应该只考虑和非AR模式比较。
```

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

```bash
本次实现内容
基于 distill_plan.md，我新增了完整的 Distillation 训练+测试框架：

训练脚本

scripts/train_consistency.py
实现了：
teacher 加载与冻结
student + EMA student 初始化
Karras 时间网格（rho=7）及混合状态 transition
教师引导相邻状态构造：(state_{t+1}, \hat state_t)
连续：一步回推（get_prev_from_recon）
离散：后验 q(v_t | v_{t+1}, \hat v_0) 采样
一致性损失训练（对齐 EMA student，不直接 KD teacher logits）
checkpoint 保存/恢复、可选 wandb、在线评估钩子（调用 eval 脚本）
评估脚本

scripts/eval_consistency.py
支持：
1/2/4/8 等少步采样
混合状态多步闭环：predict x0 -> renoise -> next step
生成 SDF + gen_info.csv
汇总速度与基础指标（QED/SA/validity/connectivity/recon_success/complete）
可选 vina 评估入口
一致性损失模块

models/consistency_loss.py
实现 pos MSE + node KL + edge KL 的混合一致性损失
预留 compute_joint_physics_loss（当前按计划保留占位，不默认启用）
通用工具

utils/consistency.py
包含 Karras sigmas、betas 构造、transition 初始化、EMA 更新、采样步序、renoise 等复用逻辑
兼容入口脚本（按计划命名）

scripts/distill_trajectory.py（代理到 train_consistency.py）
scripts/evaluate_distilled.py（代理到 eval_consistency.py）
配置文件

configs/distill/distill_pxm.yaml
包含 teacher、Karras 网格、loss 权重、训练超参、wandb/online eval 配置
运行方式
训练

python3 scripts/train_consistency.py \
  --config configs/distill/distill_pxm.yaml \
  --outdir outputs_distill \
  --device cuda:0
或计划里的兼容命令：


python3 scripts/distill_trajectory.py --config configs/distill/distill_pxm.yaml
评估（1/2/4/8步）

python3 scripts/eval_consistency.py \
  --config_distill configs/distill/distill_pxm.yaml \
  --consistency_ckpt outputs_distill/<你的实验目录>/checkpoints/last.pt \
  --config_task configs/sample/test/sbdd_csd/simple.yml \
  --sample_steps 1 2 4 8 \
  --num_mols 1000 \
  --device cuda:0
或兼容命令：


7 Files Changed
distill_pxm.yaml
+58
consistency_loss.py
+85
distill_trajectory.py
+10
eval_consistency.py
+345
evaluate_distilled.py
+10
train_consistency.py
+390
consistency.py
---
概述：
这里确实是一个CM模型，它使用CM的蒸馏loss，基于50步的EDM scheduler。
```

##### resume

```
nohup python scripts/train_consistency.py   --config configs/distill/distill_pxm.yaml   --outdir outputs_distill --resume outputs_distill/consistency_distill_20260315_013053/checkpoints/last.pt&
```



### 如何评估新的样本：

```
2) 快速评估（不跑 vina，先看趋势）
先做 1/2/4/8 步，100~300 个分子就够看方向：


python3 scripts/eval_consistency.py \
  --config_distill configs/distill/distill_pxm.yaml \
  --consistency_ckpt "$CKPT" \
  --config_task configs/sample/test/sbdd_csd/simple.yml \
  --sample_steps 1 2 4 8 \
  --num_mols 200 \
  --device cuda:1 \
  --outdir outputs_consistency_eval
  3) 全评估（带 vina）
如果快评估OK，再加 --eval_vina（会慢很多）：
---
python3 scripts/eval_consistency.py   --config_distill configs/distill/distill_pxm.yaml   --consistency_ckpt outputs_distill/consistency_distill_20260312_235428/checkpoints/last.pt   --con
fig_task configs/sample/test/sbdd_csd/simple.yml   --sample_steps 1 2 4 8   --num_mols 200   --device cuda:1   --outdir outputs_consistency_eval
---

python3 scripts/eval_consistency.py \
  --config_distill configs/distill/distill_pxm.yaml \
  --consistency_ckpt "$CKPT" \
  --config_task configs/sample/test/sbdd_csd/simple.yml \
  --sample_steps 1 2 4 8 \
  --num_mols 1000 \
  --device cuda:1 \
  --outdir outputs_consistency_eval \
  --eval_vina \
  --test_df_path data/test/dfs/sbdd_csd.csv \
  --protein_root data/csd/files/proteins \
  --exhaustiveness 16
  
python3 scripts/eval_consistency.py    --config_distill outputs_distill/consistency_distill_20260315_221913/distill_pxm_sampling.yaml    --consistency_ckpt outputs_distill/consistency_distill_20260315_221913/checkpoints/last.pt    --config_task configs/sample/test/sbdd_csd/simple.yml    --sample_steps 1 2 4 8 16    --num_mols 1000    --device cuda:1 --outdir outputs_consistency_eval   --eval_vina   --test_df_path data/test/dfs/sbdd_csd.csv   --protein_root data/csd/files/proteins   --exhaustiveness 16
```

```
'recon_success': 0.47, 'complete': 0.06, 'qed_mean': 0.4148305026830255, 'sa_mean': 0.6430158730158729, 'connectivity': 0.1111111111111111, 'sample_steps': 4
'recon_success': 0.52, 'complete': 0.25, 'qed_mean': 0.47206688660258717, 'sa_mean': 0.5823636363636364,  'connectivity': 0.4727272727272727, 'sample_steps': 8
'recon_success': 0.5, 'complete': 0.45, 'qed_mean': 0.5211918238430812, 'sa_mean': 0.5413725490196079,  'connectivity': 0.9019607843137255, 'sample_steps': 16
'recon_success': 0.52, 'complete': 0.5, 'qed_mean': 0.5312876430151697, 'sa_mean': 0.5657692307692308,  'connectivity': 0.9615384615384616, 'sample_steps': 32
```

```
python3 scripts/generate_consistency.py \
>   --config_distill outputs_distill/consistency_distill_20260312_235428/distill_pxm.yaml \
>   --consistency_ckpt outputs_distill/consistency_distill_20260312_235428/checkpoints/last.pt \
>   --config_task configs/sample/test/sbdd_csd/simple.yml \
>   --sample_steps 1 2 4 8 16 \
>   --num_mols 1000 \
>   --device cuda:1 \
>   --outdir outputs_consistency_gen
```



```bash
conda_shared
conda activate PocketXMol
cd /shared/healthinfolab/phz24002/PocketXMol/
---
targetdiff标准的evaluation
---
conda_shared
conda activate pxm_vina
cd /shared/healthinfolab/phz24002/PocketXMol/
---
./scripts/batch_eval_consistency.sh outputs_consistency_eval/你的新文件夹名字
---
cat 
```


---
运行测试代码可能需要GPU，此时可以ssh gpuxx，gpu xx指的是某一个节点。需要用squeue -u $USER看占用了哪些节点，且这些节点可能并非我独占的，需要登录后用nvidia-smi确认是否可用。