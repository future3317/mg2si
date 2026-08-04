# 维护、追因与迭代约定

## 1. 单一事实来源

- 原始 Excel、Word 和本地 SQLite 属于研究数据，不进入 Git。
- `configs/` 是字段、来源、目标和候选空间的唯一配置入口。
- `src/mg2si/io/` 负责读取与规范化；不得在根目录新增一次性解析脚本。
- `src/mg2si/data/` 负责建库、质量检查和数据库读写。
- `src/mg2si/models/` 只负责模型；候选筛选和实验约束放在 `optimization/`。
- `docs/assets/` 只保存从 SQLite 聚合出的汇报图，不保存逐样品中间表。

## 2. 推荐的调试顺序

1. 先运行 `mg2si validate-data`，确认数据库对象和质量问题。
2. 再查 `source_manifest`、`source_record` 和 `source_conflict`，确认原始来源、工作表、行号和冲突。
3. 查 `sample_mapping`、`supplement_material_lineage`，确认样品编号、父子材料和映射依据。
4. 查 `supplement_process_observation` 和 `material_input_coverage`，确认工艺字段是否真的进入模型输入。
5. 查 `bo_training`，确认分支、细胞系、浓度、双目标和模型资格标记。
6. 最后查看 `model_evaluation`、`prospective_design_space` 和 `recommendation`，区分模型证据、候选空间和本轮选择。

## 3. 标准运行入口

```powershell
conda activate EGNN
python -m pip install -e ".[test]"
python -m mg2si.cli ingest
python -m mg2si.cli validate-data
python -m mg2si.cli evaluate
python -m mg2si.cli recommend --branch synthetic --tumor-cell-line Huh-7 --normal-cell-line THLE --allow-direct-baseline
python scripts/generate_figures.py
```

`run_pipeline.ps1` 只是上述步骤的可重复封装。数据库重建后必须重新评估和生成图，不应手工修改 SQLite 或复制旧 CSV 覆盖结果。

## 4. 结果追踪要求

每轮推荐需要同时保留：数据源文件哈希、配置版本、模型路径、候选池大小、随机种子、候选生成方式、推荐角色和实验批次。若新增字段，先更新 `configs/schema.yaml` 和 `docs/DATA_DICTIONARY.md`，再修改解析器、建库视图和测试。

候选空间的“理论全因子规模”和“本轮物化候选池”必须分开记录。1024 个候选只是预算内的可复现候选池，不能表述为已经穷举全部组合。

## 5. 代码边界

- 不在 CLI 中实现数据解析、模型训练或指标计算。
- 不在图表脚本中重新定义数据清洗规则；图表只读取 SQLite。
- 不把 `allow-direct-baseline` 的结果写成正式的材料状态模型证据。
- 不把信息增益排序写成生物性能 Pareto 结论。
- 新的兼容导出需求应新增显式 CLI 子命令，而不是恢复根目录 CSV 生成脚本。
