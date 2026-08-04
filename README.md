# Mg2Si 材料中心逆向设计与主动学习

本项目围绕“工艺参数 X -> 材料状态 Z -> 生物响应 Y”建立可追溯的科研闭环：

- 从材料库和 TACE 细胞学数据中建立统一 SQLite 数据库。
- 保留原始工作簿的子表、来源文件哈希、工作表、行号和样品映射证据。
- 区分 synthetic 与 commercial 工艺分支，避免给商品原料虚构 SHS 合成参数。
- 使用材料表征作为中间状态，逐步升级为 process -> state -> biology 的多目标模型。
- 在证据不足时输出 insufficient_evidence 或信息增益候选，不把预测当成实验结论。

## 项目结构

    configs/                  字段、数据源、目标和候选空间配置
    src/mg2si/io/             Excel/Word 来源读取、数值解析和来源清洗
    src/mg2si/data/           数据库构建、质量检查、SQLite 读写
    src/mg2si/mapping/        样品别名和映射规则
    src/mg2si/models/         GP、直接基线和材料中心模型
    src/mg2si/optimization/   候选空间、约束和批次推荐
    src/mg2si/evaluation/     分组交叉验证和评估服务
    scripts/                  审计和汇报图生成，不承载主数据清洗逻辑
    docs/                     架构、字段、模型、实验闭环和汇报文档
    tests/                    科学约束和数据契约测试
    data/processed/           本地 SQLite 和模型输出，不提交到 Git

根目录不再放置旧版 prepare、rebuild、mobo_demo 或 CSV 导出脚本。所有正式流程从 mg2si CLI 进入。

## 环境与安装

    conda activate EGNN
    python -m pip install -e ".[test]"

## 标准流程

    python -m mg2si.cli ingest
    python -m mg2si.cli validate-data
    python -m mg2si.cli evaluate
    python -m mg2si.cli recommend --branch synthetic --tumor-cell-line Huh-7 --normal-cell-line THLE --allow-direct-baseline
    python -m mg2si.cli validate-candidates
    python scripts/generate_figures.py

也可以运行：

    .\run_pipeline.ps1

allow-direct-baseline 只用于当前表征数据不足时的过渡性对照。正式的材料中心模型需要足够的工艺-表征-生物响应覆盖。

## 数据和结果边界

默认数据库为 data/processed/mg2si.sqlite。数据库包含材料主表、生物长表、子表审计、来源记录、样品映射、工艺观察、质量问题、训练视图、评估结果、候选空间和推荐结果。

原始 Excel、Word、CSV、SQLite、第三方 checkout 和 Python 缓存均由 .gitignore 排除。图表脚本只从 SQLite 读取并生成透明背景 PNG，不重新定义清洗规则。

理论候选空间和本轮物化候选池必须分开解释。当前推荐池的 1024 个点是预算内、固定种子可复现的候选池，不代表穷举全部理论组合。

## 文档入口

- 维护和追因：docs/MAINTENANCE.md
- 软件架构：docs/ARCHITECTURE.md
- 候选空间：docs/DESIGN_SPACE.md
- 汇报材料：docs/PITCH_BRIEF.md
