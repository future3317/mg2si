# Mg2Si 材料中心逆向设计与主动学习

本项目围绕 `工艺参数 X -> 材料状态 Z -> 生物响应 Y` 建立科研闭环：根据目标肿瘤杀伤和正常细胞安全边界，学习材料需要达到的概率表征区间，再反推值得验证的合成、加工和应用条件。

## 科学边界

- `synthetic` 与 `commercial` 分别建模和推荐。
- 肿瘤与正常细胞系必须同时固定；样本不足时不自动混合。
- 浓度和暴露时间是应用上下文。
- 正常细胞安全性是概率约束；稳定性可配置为额外约束。
- 表征值是材料状态，优化器不能直接把 DLS、PDI、Zeta 或纯度当成旋钮。
- 推荐是下一轮实验建议，不是因果或安全结论。

## 架构

```text
配置化 Excel -> 身份映射与规范化
                       |
       X 工艺 -> Z 材料状态 -> Y 生物响应
                       |
        分组验证 + 安全概率约束
                       |
           连续候选 -> 实验回写/replay
```

核心配置在 `configs/`，实现位于 `src/mg2si/`，科学契约测试位于 `tests/`。原始 Excel 和处理后的数据库继续被 `.gitignore` 排除。

## 数据存储

默认只生成一个本地数据库：

```text
data/processed/mg2si.sqlite
```

数据库包含：

- `material`：材料、合成参数及表征；
- `bioassay`：TACE 汇总表形成的规范化浓度响应事实；
- `bioassay_source_audit`：汇总及全部阶段子表记录；
- `source_record`：两个原始工作簿所有子表的逐行 JSON 追溯；
- `source_conflict`：汇总与阶段子表不一致记录；
- `sample_mapping`：样品编号映射及依据；
- `quality_issue`：数据质量问题；
- `bo_training`：材料中心建模视图；
- `model_evaluation`、`recommendation`：评估与推荐结果。

阶段子表全部保留，但不会和“数据汇总”重复拼接训练。CSV 仅通过未来显式导出命令产生，不再作为默认流水线资产。

## EGNN 环境

```powershell
conda activate EGNN
python -m pip install -e ".[test]"
```

## 项目可视化

![材料中心主动学习闭环](docs/assets/01_material_centered_loop.png)

![当前数据就绪度](docs/assets/02_data_readiness.png)

![高斯过程剂量响应](docs/assets/05_gp_dose_response.png)

![低剂量锚定 GP 交叉验证](docs/assets/06_gp_cross_validation.png)

![双目标 Pareto 决策地图](docs/assets/07_multiobjective_landscape.png)

更多汇报图和讲解口径见 `docs/PITCH_BRIEF.md`。图表由以下命令从本地 SQLite 聚合生成，不包含原始实验明细：

```powershell
python scripts/generate_figures.py
```

BoTorch/GPyTorch 是主概率模型内核。MultiBgolearn 仅作为可选历史对照：

```powershell
python -m pip install -e ".[legacy]"
```

## CLI

```powershell
mg2si ingest
mg2si validate-data
mg2si build-dataset
mg2si evaluate
mg2si recommend --branch commercial --tumor-cell-line Huh-7 --normal-cell-line L02 --allow-direct-baseline
mg2si validate-candidates
mg2si clean-derived
mg2si replay
```

推荐必须显式指定分支和细胞系。`--allow-direct-baseline` 只用于表征尚未补齐时的研发对照；正式两阶段模型证据不足时默认返回 `insufficient_evidence`。

完整本地流水线：

```powershell
.\run_pipeline.ps1
```

`configs/design_space.yaml` 的范围是基于现有字段建立的初始工程占位边界。进入真实实验前必须由实验负责人确认设备边界、化学可行域和安全限制。

详细规范见 `docs/DATA_DICTIONARY.md`、`docs/MODEL_CARD.md`、`docs/OBJECTIVE_SPEC.md`、`docs/EXPERIMENT_LOOP.md` 和 `docs/ARCHITECTURE.md`。
