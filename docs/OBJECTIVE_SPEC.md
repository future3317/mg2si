# 目标与约束规范

- 效果目标：最小化 `y_tumor_viability_pct`。
- 安全约束：`P(y_normal_viability_pct >= 80) >= 0.80`，在 `configs/objectives.yaml` 中维护。
- 工艺约束：满足 `configs/design_space.yaml` 的分支字段、范围和禁用字段。
- 稳定性约束：待项目组确定 PDI、粒径漂移或储存稳定性阈值后启用。

`normal - tumor viability` 可作为描述性窗口。旧的 `(100-tumor)/(100-normal)` 不作为优化目标，因为正常存活率接近 100 时数值不稳定。

