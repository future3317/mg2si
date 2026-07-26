# 模型卡

模型在固定的 `workflow_branch × tumor_cell_line × normal_cell_line` 范围内运行，用于提出下一轮实验，不用于临床、安全或因果结论。

主路径为 `p(Z|X, branch)`、`p(Y|Z,C)` 和后验抽样不确定性传播。默认后端为 BoTorch/GPyTorch；ExtraTrees 只用于快速 grouped-CV 基线。原 MultiBgolearn 不再是主内核。

每个范围单独检查完整目标行数、独立材料/实验组数和表征覆盖。未满足 `configs/experiment.yaml` 门槛时返回 `insufficient_evidence`。显式启用的直接基线会标记为 `direct_process_to_biology_transitional_baseline`。

验证必须按 `material_parent_id` 和 `experiment_id` 分组。当前历史数据的映射和表征覆盖不足，不能发布可信的纯度、DLS、PDI、Zeta 或氧化层目标区间。

