# 发布与复现实验检查清单

## 提交前

- [ ] 只提交 Python、PowerShell、依赖和文档文件。
- [ ] 原始 Excel、派生 CSV、模型输出和本地 `MultiBgolearn/` 不在暂存区。
- [ ] 原始数据的授权范围允许本地处理；仓库不包含可逆推出实验数据的中间结果。
- [ ] `workflow_branch`、`synthesis_required` 和两个目标列的定义没有被本次改动破坏。

## 本地复现

```powershell
conda activate EGNN
python -m pip install -r requirements-egnn.txt
.\run_pipeline.ps1
```

流水线最后的 `validate_bo_contract.py` 必须输出 `status: ok`。如果合成候选为零，应检查材料库中的 SHS、球磨和粒径覆盖率，不要用统计填补替代真实工艺。

## 实验回写

- [ ] 候选编号已映射到实际样品编号和批次。
- [ ] 实际制备参数、表征值、细胞系、浓度和暴露时间已记录。
- [ ] 杀伤、安全和稳定性结果已按同一目标定义回写。
- [ ] 失败实验和设备限制也已记录，避免下一轮重复推荐不可执行方案。
