# 材料编号、工艺与表征映射审计

更新时间：2026-07-27

## 结论

当前数据库已完成材料编号实体化和主外键关联：

| 检查项 | 结果 |
|---|---:|
| 生物响应记录 | 402 |
| 样品编号 | 70 |
| 已映射样品编号 | 70 |
| 未映射生物记录 | 0 |
| 材料登记 | 98 |
| 材料级可建模记录 | 51 |
| 工艺映射 | 536 |
| 具体材料级工艺映射 | 368 |
| 来源文档范围工艺证据 | 168 |
| 表征映射 | 17 |
| 工艺映射孤儿外键 | 0 |
| 表征映射孤儿外键 | 0 |

材料编号映射覆盖率为 100%。所有 TACE 样品编号都拥有对应材料实体，并通过 `bioassay.material_id` 关联到 `material.material_id`。

## 映射粒度

### 材料实体

- 主材料库已有编号保持原编号和已确认血缘。
- 研究者确认的溶剂后处理、粒径分级和索引号映射优先使用。
- 只出现在生物表中的规范样品编号登记为 `source_derived` 材料实体。
- 无已确认父材料的样品将自身作为当前材料实体，不猜测父材料。

### 工艺映射

`material_process_mapping` 使用长表结构保存材料、参数、数值、单位、映射状态、来源文件和来源记录号。

- `mapping_scope = material`：参数能够定位到具体材料。
- `mapping_scope = source_document`：参数只能够定位到协议或文档，原文没有足够证据唯一对应某个材料。

来源文档范围的 168 条工艺证据不是丢失数据。它们完整保留在数据库中，但不会在没有证据时被强行分配给材料或进入 BO 特征。

### 表征映射

`material_characterization_mapping` 将 XRD、纯度、晶粒、粒径、DLS、PDI、Zeta 和表面化学等已知值统一为材料级长表。当前共有 17 条非空表征映射，全部能够关联到材料主表。

## 对模型的影响

- `model_eligible_material` 从 42 条增加到 51 条。
- 肿瘤存活率交叉验证的材料分组数从 7 组增加到 10 组。
- 肿瘤目标 Spearman 从约 0.64 更新到约 0.71。
- 正常细胞目标 Spearman 约为 0.66。

当前商品路线只有 9 条材料级记录，仍属于证据不足范围。材料编号已经不是主要瓶颈，下一阶段重点是补齐协议参数对应的具体样品/批次，以及 DLS、PDI、Zeta、纯度等表征字段。

## 可重复构建顺序

```powershell
conda run --no-capture-output -n EGNN python -m mg2si.cli ingest
conda run --no-capture-output -n EGNN python -m mg2si.cli evaluate
conda run --no-capture-output -n EGNN python -m mg2si.cli enumerate-design-space
conda run --no-capture-output -n EGNN python scripts/generate_figures.py
```

上述顺序可从原始 Excel/Word 和配置重新构建 SQLite、模型评估、prospective 候选空间与全部汇报图。
