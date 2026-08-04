# 全流程前瞻候选空间

项目中的候选空间分为两种，不能混用：

- `real_candidate_space`：已经登记并有材料身份的真实样品 × 已测剂量，用于实验回流和已知样品状态检查。
- `prospective_design_space`：按照工艺条件水平做笛卡尔积，用于未来实验设计、主动学习和 BO 候选池。

## 条件分支

合成路线 `synthetic` 包含管式炉阶段：Mg/Si 摩尔比、最高温度、保温时间、真空循环、真空时间、初始压力和保护气氛；之后继续展开球磨、后处理、粒径目标、PVP 修饰和剂量。

商品路线 `commercial` 不展开管式炉条件。相关字段保留为 `NaN`，并由 `synthesis_parameter_applicability=not_applicable` 明确标记；它仍然展开商品来源、球磨、超声/溶剂后处理、粒径目标、PVP 修饰和剂量。

## 全因子规模与物化策略

全因子规模由 [design_space.yaml](../configs/design_space.yaml) 中的 `factor_levels` 定义。命令默认最多物化 10000 个候选点：

```powershell
conda run --no-capture-output -n EGNN python -m mg2si.cli enumerate-design-space --max-points 10000
```

数据库中的 `prospective_design_space_summary` 保存完整笛卡尔积规模，`prospective_design_space` 保存当前预算内的候选点。候选点携带：

- `candidate_source=prospective_full_factorial`
- `full_factorial_count`
- `candidate_space_complete`
- `synthesis_parameter_applicability`
- `not_applicable_fields`

超过物化预算时不声称已经保存全部组合，而是保存固定随机种子产生的可复现子集。模型只有在对应字段有足够实测数据后才可以对这些候选做后验预测；候选本身不是实验验证结果。

实验批次默认设置为每轮 3 个候选，配置位于 `configs/experiment.yaml` 的 `recommendation.batch_size`。全因子空间是候选池，推荐器每轮只从中输出 3 个优先验证对象。
