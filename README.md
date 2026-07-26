# Mg2Si 材料主动学习项目

本项目以材料为中心，建立 Mg2Si 材料组成、制备工艺、结构/表面表征与生物应用结果之间的可追溯数据链。目标不是只预测一个指标，而是回答可执行的科研问题：需要调整哪些材料或工艺参数、采用哪条制备路径、材料需要达到哪些表征和生物学指标，才能在安全边界内获得目标杀伤性能。项目代码只上传处理流程、建模逻辑和文档；原始实验数据及其派生表格不进入 Git 仓库。

## 项目边界

- 材料库负责提供合成、表征、表面化学和材料筛选信息。
- TACE 数据库负责提供材料对应的细胞学结果；其汇总表和子表均保留来源标记。
- `material_id`、`material_parent_id` 和 `sample_id` 用于连接样品、材料批次和细胞学记录。
- `workflow_branch` 区分 `synthetic`（需要 SHS）和 `commercial`（使用商品原料）两条制备路径。
- 只有目标指标和关键工艺字段满足完整性要求时，记录才进入 BO 训练或候选生成。

## 工艺分支

```text
synthetic:  原料配比 -> SHS 合成 -> 球磨 -> 粒径筛选 -> 后处理 -> PVP/细胞学实验
commercial: 商品原料 -> 球磨 -> 粒径筛选 -> 后处理 -> PVP/细胞学实验
```

商品原料不填充或虚构 SHS 温度、真空、保温时间等字段，而是将这些字段标记为 `not_applicable`。两条路径都要求球磨、目标粒径、球料比和球磨时间可执行，否则不生成虚拟候选。

## 目录与脚本

| 文件 | 用途 |
| --- | --- |
| `rebuild_material_cell_dataset.py` | 合并材料库子表与 TACE 汇总数据，完成样品编号映射并生成联合数据集 |
| `export_tace_subtables.py` | 展开 TACE 全部子表，保留 `source_sheet` 和跨表重复审计字段 |
| `prepare_bo_dataset.py` | 生成不覆盖主数据的通用规范化 BO 表（`bo_prepared_*`），用于独立审计 |
| `mobo_demo.py` | 在统一细胞系范围内运行 MultiBgolearn 多目标 BO demo，并生成条件工艺候选 |
| `validate_bo_contract.py` | 校验 BO 特征、双目标和候选工艺字段是否满足数据契约 |
| `run_pipeline.ps1` | 在 `EGNN` 环境中按顺序执行完整流水线并处理 Windows 编码问题 |
| `docs/ARCHITECTURE.md` | 数据分层、映射规则、特征契约和实验闭环说明 |
| `docs/DECISION_FRAMEWORK.md` | 将模型输出转化为材料合成与应用决策的实验规则 |
| `requirements-egnn.txt` | `EGNN` conda 环境中的 Python 依赖 |

## 环境安装

推荐使用已有的 `EGNN` conda 环境：

```powershell
conda activate EGNN
conda install -c conda-forge pygmo
python -m pip install -r requirements-egnn.txt
```

其中 `pygmo` 用于多目标指标相关计算，建议由 conda-forge 安装以减少二进制依赖问题。

## 数据处理与 demo

将两个原始 Excel 文件放在项目目录后，在 `EGNN` 环境中执行：

```powershell
conda run -n EGNN python rebuild_material_cell_dataset.py
conda run -n EGNN python export_tace_subtables.py
conda run -n EGNN python prepare_bo_dataset.py
conda run -n EGNN python mobo_demo.py
```

脚本默认读取：

- `Mg2Si材料库_已汇总SHS实验记录.xlsx`
- `TACE项目 材料品控 数据索引-260716.xlsx`

脚本生成的 CSV 包含实验联合表、来源展开表、BO 特征/目标表和 demo 候选表。`rebuild_material_cell_dataset.py` 生成的 `bo_joint_dataset.csv` 是条件分支 BO 的规范输入；`prepare_bo_dataset.py` 只生成带 `bo_prepared_` 前缀的独立审计输出，不会覆盖主输入。所有这些文件均被 `.gitignore` 排除，只保存在本地用于分析和复现实验。也可以直接执行：

```powershell
.\run_pipeline.ps1
```

## BO 数据契约

输入特征包括：

- 材料身份：`material_id`、`material_parent_id`、`sample_id`、原料类型；
- SHS 工艺（仅 `synthetic`）：Mg/Si 比、合成温度、保温时间、真空/压力条件；
- 通用工艺：球磨时间、球料比、目标粒径、浓度、后处理方式；
- 条件控制：`workflow_branch`、`synthesis_required`、`synthesis_feature_status`。

输出目标包括两个大目标列：杀伤/生物活性目标和安全性目标；后续可增加稳定性作为约束或第三目标。当前 demo 以 Huh-7 作为统一细胞系建模范围，并使用 EHVI 对两个目标进行联合推荐。扩展到其他细胞系前，应先确认目标定义、方向（最大化/最小化）和测量单位一致。

## 如何从推荐得到合成方向

每个推荐候选应同时读取 `workflow_branch`、材料组成、SHS 字段（如适用）、球磨与粒径字段、浓度和后处理方式，不能只看模型分数。推荐的使用顺序是：先剔除安全性不达标或工艺不可执行的候选，再比较 Pareto 前沿上的杀伤性能、选择性和稳定性，最后由实验人员确认设备边界并执行验证。具体参数优先级、指标门槛和实验轮次定义见 [`docs/DECISION_FRAMEWORK.md`](docs/DECISION_FRAMEWORK.md)。

## 数据质量与当前限制

- 数据映射采用精确样品号、别名、层级/父样品别名等可审计规则；未匹配记录不会被静默拼接。
- TACE 子表与“数据汇总”可能存在重复实验组，训练时应使用汇总表或明确去重后的注册表，不能直接把所有来源重复行当作独立样本。
- 当前数据中可生成的合成候选受球磨参数和目标粒径完整性限制；缺失值不会用中位数伪造工艺条件，因此合成候选可能为零。
- 当前 demo 只验证推荐流程和约束，不替代真实实验设计、统计验证或生物安全判断。
- 当前模型输出的是“下一组值得验证的材料-工艺组合”，不是因果结论；参数方向必须通过受控实验和后续回写数据确认。

## 数据不入库

`.gitignore` 明确排除 `*.xlsx`、`*.csv`、`bo_*`、`mobo_*`、`MultiBgolearn/` 及本地环境文件。提交前必须确认暂存区只包含代码、配置和文档，不包含任何原始或生成数据。

## 许可证与研究记录

本项目用于科研内部开发。数据的使用、共享和发表应遵循数据提供方授权及项目组约定。代码提交信息应说明处理规则或模型规则的变化，避免只提交不可追溯的结果文件。
