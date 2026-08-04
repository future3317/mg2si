# 软件架构

## 总体数据流

    原始 Excel / Word
            |
            v
    io.material_reader / io.biology_reader
            |
            v
    规范化记录 + 来源哈希 + 样品映射
            |
            v
    data.build_database
            |
            v
    data/processed/mg2si.sqlite
            |
            +--> data.quality ------> 质量状态
            +--> evaluation.service -> 分组评估
            +--> models ------------> 后验预测
            +--> optimization ------> 候选池与 3 个推荐
            +--> scripts ------------> 聚合图表

SQLite 是项目的唯一派生数据容器。CSV 只允许作为用户明确要求的互操作输出，不作为默认流水线资产。

## 模块职责

### configs/

配置包括数据源路径、字段角色、目标定义、候选空间、别名和补充映射。会改变科学含义的内容必须进入配置，不要硬编码在脚本中。

### src/mg2si/io/

- material_reader.py 读取材料主表及材料库子表。
- biology_reader.py 读取 TACE 汇总表和各子表，并展开为细胞实验长表。
- parsers.py 统一处理数值、范围、单位和 PVP 分子量。
- source_manifest.py 计算文件哈希并记录来源元数据。
- excel_reader.py 校验工作簿和配置的工作表是否一致。

这一层只负责“读到什么、如何规范化、来自哪里”，不负责模型筛选。

### src/mg2si/data/

build_database.py 负责把来源记录合并为材料主表、生物实验长表、工艺观察、映射、质量问题和 bo_training 视图。store.py 只负责连接、读写和清理历史派生 CSV。quality.py 负责数据库级状态判断。

### src/mg2si/models/

- botorch_surrogate.py：GP 后端和缺失值填补。
- direct_baseline.py：ExtraTrees 过渡基线。
- material_centered.py：p(Z|X) 与 p(Y|Z,C) 的材料中心模型。

模型模块不读取 Excel，也不自行猜测样品映射。

### src/mg2si/optimization/

- design_space.py：理论全因子空间、候选物化和连续采样。
- constraints.py：分支、字段适用性和设备边界。
- recommend.py：训练范围筛选、预测、信息增益和批次多样性选择。
- real_space.py：已登记真实样品的探索和回溯，不冒充前瞻候选。

### src/mg2si/evaluation/

grouped_splits.py 定义按材料父体或实验分组的无泄漏切分，metrics.py 定义指标，service.py 负责从 bo_training 读取数据并把评估写回 model_evaluation。CLI 不再承载评估实现。

### scripts/

只保留审计和图表生成。脚本可以编排包内服务，但不能复制解析、映射或模型逻辑。

## 可追溯性

追踪一条结果时按以下顺序：

1. recommendation：候选、角色、模型路径、概率和采样元数据。
2. prospective_design_space：候选是如何从理论空间物化的。
3. bo_training：模型实际看到了哪些输入和两个目标。
4. sample_mapping / supplement_material_lineage：材料编号和父子关系。
5. supplement_process_observation：工艺参数的来源和适用阶段。
6. source_record / source_manifest：原始文件、工作表、行号和文件哈希。
7. source_conflict / quality_issue：为什么某个字段没有进入训练或被标记为不可靠。

## 变更规则

新增字段时，按以下顺序修改：

1. configs/schema.yaml 和相关来源配置。
2. io 解析器及其来源元数据。
3. data.build_database 的规范化表或视图。
4. optimization 的候选约束和模型特征选择。
5. tests/ 中对应的契约测试。
6. DATA_DICTIONARY.md、模型卡和实验闭环。

禁止恢复根目录的一次性解析脚本。需要新的导出格式时，新增显式 CLI 子命令，并把来源和配置版本写入输出。
