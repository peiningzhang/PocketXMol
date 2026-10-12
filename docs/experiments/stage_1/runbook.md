# Stage 1 执行与复查

工作目录：`/shared/healthinfolab/phz24002/PocketXMol`。此实验不安装或修改共享环境，不训练模型。实际参数、来源、种子、硬件和调用次数以各目录的 run.json、sample.yml 及 ledger 为准。

## 环境

生成使用 `/shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/bin/python`，设置 `LD_LIBRARY_PATH=/shared/healthinfolab/phz24002/anaconda3/envs/PocketXMol/lib`。物理评估使用同目录下 `reed_molecule/bin/python`，Vina 使用 `pxm_vina/bin/python`。线程环境均设 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。

GPU 任务必须 SSH 到 GPU 节点，执行前检查分配、硬件和实际占用。当前会话授权空闲未分配 GPU；这项授权记录在 protocol.json。不能直接复制历史 GPU/PID 状态作为当前可用证据。

## 标准入口

先确认没有同一 runner 在运行；不要因观察超时启动重复任务。

- `scripts/run_stage1.py --retry_failed`：核心40项任务；完整失败重试进入新目录，保留旧原始结果。
- `scripts/run_stage1_adaptations.py`：固定开发矩阵。旧 partial conditioning-v1 仅诊断；修复以 `scripts/run_stage1_partial_conditioning_repair.py` 和 repair ledger 为准，当前生成已完成，不应重复启动。
- `scripts/evaluate_stage1.py --root <root> --watch`：依次完成几何、原构象 Vina、选择及全候选能量。root 可为 stage_1、development、development_conditioning_v2、confirmation 或 development_extensions。禁止两个扫描器重叠处理同一目录；本会话的独占移交详见日志和 folders 文件。
- `scripts/freeze_stage1_confirmation.py --check_only`：必须全部36个正式开发单元及对应开发基线已完整评估。移除 `--check_only` 只冻结一次不可变 manifest，禁止覆盖。
- `scripts/run_stage1_confirmation.py --retry_failed`：冻结后运行80口袋、五种子、B400/800、两个选定方法。
- `scripts/run_stage1_development_extensions.py --retry_failed`：仅补齐冻结方法的开发种子/预算，不用于重新选参；优先复用 conditioning-v2 修复目录。
- `scripts/benchmark_stage1.py` 与同命令加 `--trace_control`：冻结方法的同 GPU 三轮计时，基线已完成时复用。两组必须同一硬件、batch4；`--baseline_only` 可在选择前运行基线计时。

队列辅助脚本含本会话进程句柄，用于当前排队与独占移交。复跑实验时使用上述标准入口，重新确认活跃进程和独占目录，不复用历史 PID。

## 统计和诊断

以下使用 reed_molecule Python，均为 CPU 分析：

```text
scripts/analyze_stage1.py
scripts/analyze_stage1_budget_trends.py
scripts/analyze_stage1_guardrails.py
scripts/analyze_stage1_timing.py
scripts/analyze_stage1_timing.py --trace_control
scripts/analyze_stage1_repair_attribution.py
scripts/audit_stage1_completion.py
```

图表使用 `scripts/plot_stage1.py`，设置可写 `MPLCONFIGDIR=/tmp/pxm_stage1_mpl`。诊断评估入口 `scripts/run_stage1_diagnostics.py` 复用 EVALUATED 状态，不重新采样。`repair_damage_with_attribution_v2.csv` 将独立物理结果与相邻轮次原子/坐标/键归因合并；按配置轮数分别汇总，不将重复的第一轮作为独立样本。

统计和图表在矩阵未完成时明确标为 provisional。completion_audit.json 是必要检查，不能代替最终对报告解释、图表、失败机制和 stage_2 条件的逐项人工审查。N16 只有完整核心矩阵的预算趋势触发后才运行，不能从部分结果提前扩张。

确认矩阵及冻结方法三轮 trace-off 计时全部完成后，运行 `scripts/plot_stage1_controlled_timing.py`。它使用确认80口袋/五种子的质量和固定8口袋/三轮的原型吞吐；横向误差为计时标准差，纵向误差为口袋 bootstrap95CI。两个范围的实际 NFE 分别记录，尤其不能将原生精修子集的提前退出混同为相同计算量。

运行 `scripts/stage1_merged_metadata.py` 可核对并修复全部 N8 视图的分片初始化/预热汇总，原元数据另存备份。旧分片未记录的字段保持 null，不能用首分片的数值冒充总和；候选、配置、主采样调用计数和评分不变。

Conditional N16 extension only after `tables/n16_decision.json` records complete core and `bounded_N16_extension_warranted`: SSH GPU-node generation with `scripts/run_stage1_n16.py --retry_failed`; independent CPU evaluation with `scripts/evaluate_stage1_n16.py --watch`; then `scripts/analyze_stage1_n16.py` and `scripts/plot_stage1_n16.py`. These entrypoints reject an untriggered/incomplete gate. The original N8 prefix is reused verbatim for dev20/first3 seeds; only repeats8–15, 1600-step and native extra15 controls are newly generated. Extension statistics and NFE/time exports stay under `n16_extension`; they are exploratory development evidence, not full100 or confirmation results.

### 最后一个评估完成后的产物流程

`scripts/finalize_stage1_artifacts.py` 持有独占锁，等待冻结确认20单元全部 `EVALUATED`，再顺序执行最终统计、实际NFE严格匹配、guardrails、两套重复计时分析、Stage2量化门槛、图形及报告。运行记录为 `finalization_status.json` 和每步 `finalization_*.log`；完成标记为 `ARTIFACTS_RENDERED_PENDING_MANUAL_REVIEW`。这个标记仅证明生成流程完成，不代表用户目标完成。

最后必须读取报告、逐口袋表、门槛及完整性审计，核对计划逐项要求并查看实际渲染图形；检查调度父进程恢复链全部退出，并只清理属于本计划且身份核验通过的空闲watcher。`completion_audit.json` 为必要条件，人工证据复查是额外必要条件。

最终代码先保存在 `source_snapshot_final/`，报告的最后证据补充修订另存 `source_snapshot_final_revisions/`，其 manifest 指定对应文件的新哈希。固定六个采样文件仍由 confirmation_manifest 单独核对，不受报告修订影响。

冻结FK候选池guardrails的独立入口为：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /shared/healthinfolab/phz24002/anaconda3/envs/reed_molecule/bin/python scripts/analyze_stage1_pool_guardrails.py
```

它仅依赖完整geometry候选表，不依赖energy是否可计算；对确认80口袋×5seed与同seed独立P4/P8前缀做配对。缺少两有效分子时fingerprint diversity保持NaN；共同定义对的数量单列，主质量/coverage仍保留全部计划分母。最新两份分析/报告源文件的快照为 source_snapshot_final_guardrails/，其manifest优先于之前相同文件的快照。重新制作全部产物前先执行该入口。
