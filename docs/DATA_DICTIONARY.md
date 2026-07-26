# 数据字典与字段角色

项目按 `X -> Z -> Y` 管理字段，机器可读定义位于 `configs/schema.yaml`。

| 角色 | 含义 | 示例 | 优化器可直接设置 |
|---|---|---|---|
| X | 合成、加工和应用前可控制参数 | Mg/Si、SHS 温度、球磨、PVP、浓度 | 是，受分支和设备边界限制 |
| Z | 合成后测得的材料状态 | 纯度、晶粒、DLS、PDI、Zeta、表面化学 | 否，由 `p(Z|X)` 预测 |
| C | 生物实验上下文 | 肿瘤/正常细胞系、浓度、暴露时间、批次 | 固定或按实验设计枚举 |
| Y | 生物响应 | 肿瘤和正常细胞存活率、ROS、凋亡 | 否 |
| B | 干扰和追溯 | 供应商、设备、日期、source sheet、mapping type | 否 |

数值字段应逐步配套保存 `value_raw`、`value_numeric`、`unit_raw`、`unit_standard`、`range_lower`、`range_upper`、`parser_rule`、`parser_version` 和 `quality_flag`。当前兼容导出仍保留既有宽表列名；新解析器已经保留范围边界。

`normal_cell_line`、`exposure_time_h` 和 `experiment_id` 是正式模型上下文字段。缺失时允许入库，但不得跨细胞系合并训练。

正常细胞结果还包含 `normal_measurement_group_id`。当同一组安全性测量被多个肿瘤实验复用时，所有相关记录共享该 ID，验证和不确定性估计不得把它们当作独立重复。

相对对照的细胞存活率允许超过 100%，此时标记为 `above_control_reference`，不自动裁剪。小于 0 或超过暂定生物合理上限 200 的值才阻止进入模型；上限需由实验负责人确认。
