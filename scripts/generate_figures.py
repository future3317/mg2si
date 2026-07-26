from __future__ import annotations

from pathlib import Path
import sqlite3

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "processed" / "mg2si.sqlite"
OUTPUT = ROOT / "docs" / "assets"

INK = "#17324D"
BLUE = "#246BCE"
CYAN = "#2A9D8F"
GOLD = "#E9A23B"
ORANGE = "#E76F51"
PALE = "#F5F2EA"
PANEL = "#FFFFFF"
GRID = "#D9E1E8"
MUTED = "#61758A"


def configure() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(
        (name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS") if name in available),
        "DejaVu Sans",
    )
    plt.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": False,
        "figure.facecolor": PALE,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "text.color": INK,
        "savefig.facecolor": PALE,
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def rounded_box(ax, x, y, width, height, title, detail, color, title_size=13):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.8, edgecolor=color, facecolor=PANEL,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.64, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=color)
    ax.text(x + width / 2, y + height * 0.30, detail, ha="center", va="center", fontsize=9.5, color=MUTED, linespacing=1.4)


def arrow(ax, start, end, color=INK):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=color))


def project_loop() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.05, 0.93, "Mg2Si 材料中心主动学习闭环", fontsize=24, fontweight="bold")
    ax.text(0.05, 0.875, "目标：从期望生物学性能反推材料指标与可执行制备工艺", fontsize=12, color=MUTED)

    positions = [
        (0.03, 0.48, "统一数据资产", "合成参数｜材料表征\n细胞安全｜肿瘤杀伤", BLUE),
        (0.225, 0.48, "工艺 → 材料状态", "p(Z | X, branch)\n纯度｜粒径｜PDI｜Zeta", CYAN),
        (0.42, 0.48, "材料状态 → 生物响应", "p(Y | Z, C)\n细胞系｜剂量｜暴露时间", GOLD),
        (0.615, 0.48, "安全约束优化", "最大化肿瘤效果\n约束正常细胞安全概率", ORANGE),
        (0.81, 0.48, "下一轮实验", "优选｜探索\n边界学习｜重复锚点", BLUE),
    ]
    widths = [0.16, 0.16, 0.16, 0.16, 0.16]
    for (x, y, title, detail, color), width in zip(positions, widths):
        rounded_box(ax, x, y, width, 0.24, title, detail, color, title_size=11.5)
    for index in range(len(positions) - 1):
        start_x = positions[index][0] + widths[index]
        end_x = positions[index + 1][0]
        arrow(ax, (start_x + 0.005, 0.60), (end_x - 0.005, 0.60), MUTED)

    rounded_box(ax, 0.14, 0.16, 0.25, 0.17, "Synthetic 分支", "Mg/Si｜SHS 温度｜真空/压力\n保温｜球磨｜表面修饰", CYAN, 12)
    rounded_box(ax, 0.61, 0.16, 0.25, 0.17, "Commercial 分支", "商品原料不执行 SHS\n优化球磨｜PVP｜分散与应用条件", GOLD, 12)
    arrow(ax, (0.95, 0.48), (0.91, 0.30), BLUE)
    arrow(ax, (0.82, 0.17), (0.18, 0.13), BLUE)
    arrow(ax, (0.18, 0.13), (0.10, 0.47), BLUE)
    ax.text(0.50, 0.055, "实验结果回写数据库 → 更新模型 → 生成下一轮建议", ha="center", fontsize=11, fontweight="bold", color=BLUE)
    save(fig, "01_material_centered_loop.png")


def data_readiness(connection: sqlite3.Connection) -> None:
    total = connection.execute("SELECT COUNT(*) FROM bioassay").fetchone()[0]
    explicit = connection.execute("SELECT SUM(normal_cell_line_status='explicit') FROM bioassay").fetchone()[0]
    mapped = connection.execute("SELECT SUM(material_id IS NOT NULL) FROM bioassay").fetchone()[0]
    direct = connection.execute("SELECT SUM(model_eligible_direct) FROM bioassay").fetchone()[0]
    material = connection.execute("SELECT SUM(model_eligible_material) FROM bioassay").fetchone()[0]
    state_columns = ["mg2si_purity_pct", "grain_size_nm", "dls_size_nm", "pdi", "zeta_potential_mv"]
    state_rates = [
        connection.execute(f'SELECT AVG("{column}" IS NOT NULL) FROM material').fetchone()[0] or 0
        for column in state_columns
    ]
    metrics = [
        ("正常细胞身份明确", 100 * explicit / total),
        ("材料编号映射成功", 100 * mapped / total),
        ("直接基线可用记录", 100 * direct / total),
        ("材料中心可关联记录", 100 * material / total),
        ("核心材料表征平均覆盖", 100 * float(np.mean(state_rates))),
    ]
    labels, values = zip(*metrics)
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    y = np.arange(len(labels))
    colors = [BLUE, CYAN, GOLD, ORANGE, MUTED]
    bars = ax.barh(y, values, color=colors, height=0.58)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("占全部 402 个浓度响应点的比例（%）")
    ax.set_title("当前数据就绪度", loc="left", fontsize=22, fontweight="bold", pad=20)
    ax.text(0, 1.02, "真实数据库覆盖情况；材料表征补齐后才能启用正式两阶段模型", transform=ax.transAxes, color=MUTED, fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(min(value + 1.5, 94), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontweight="bold", color=INK)
    ax.text(0.99, -0.16, "Source: mg2si.sqlite｜2026-07-26", transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    save(fig, "02_data_readiness.png")


def baseline_validation(connection: sqlite3.Connection) -> None:
    frame = pd.read_sql_query(
        "SELECT target, mae, rmse, spearman, rows FROM model_evaluation WHERE target IS NOT NULL ORDER BY target",
        connection,
    )
    labels = ["正常细胞存活率", "肿瘤细胞存活率"] if len(frame) == 2 else frame["target"].tolist()
    x = np.arange(len(frame))
    width = 0.31
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    mae = ax.bar(x - width / 2, frame["mae"], width, label="MAE", color=BLUE)
    rmse = ax.bar(x + width / 2, frame["rmse"], width, label="RMSE", color=GOLD)
    ax.set_xticks(x, labels)
    ax.set_ylabel("预测误差（百分点）")
    ax.set_title("过渡基线的分组交叉验证", loc="left", fontsize=22, fontweight="bold", pad=20)
    ax.text(0, 1.02, "Synthetic × Huh-7 × THLE｜n=42｜按材料父级分组；数值越低越好", transform=ax.transAxes, color=MUTED, fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bars in (mae, rmse):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.1f}", ha="center", fontsize=10, fontweight="bold")
    for index, value in enumerate(frame["spearman"]):
        ax.text(index, -0.13, f"Spearman = {value:.2f}", ha="center", transform=ax.get_xaxis_transform(), color=ORANGE, fontsize=10)
    ax.text(0.99, -0.22, "研发基线，不代表模型已达到实验决策标准", transform=ax.transAxes, ha="right", fontsize=9.5, color=ORANGE, fontweight="bold")
    save(fig, "03_baseline_validation.png")


def roadmap() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.05, 0.92, "从数据资产到材料逆向设计", fontsize=24, fontweight="bold")
    ax.text(0.05, 0.865, "三阶段能力路线图：每一步都有明确的数据门槛和可交付结果", fontsize=12, color=MUTED)
    stages = [
        (0.05, "现在｜可信数据底座", BLUE, [
            "统一 SQLite 与来源哈希",
            "材料/细胞样品映射",
            "子表冲突与重复审计",
            "条件分支和安全约束框架",
        ]),
        (0.365, "补字段后｜材料中心模型", CYAN, [
            "工艺 → 纯度/粒径/PDI/Zeta",
            "表征 → 肿瘤杀伤与正常安全",
            "学习概率材料目标域",
            "输出参数区间与可信区间",
        ]),
        (0.68, "闭环后｜主动学习平台", ORANGE, [
            "安全约束 BoTorch 批量推荐",
            "探索/利用/边界/重复组合",
            "实验结果自动回写与重训",
            "历史 replay 量化迭代收益",
        ]),
    ]
    for x, title, color, items in stages:
        box = FancyBboxPatch((x, 0.20), 0.27, 0.54, boxstyle="round,pad=0.02,rounding_size=0.025", linewidth=2, edgecolor=color, facecolor=PANEL)
        ax.add_patch(box)
        ax.text(x + 0.025, 0.67, title, fontsize=14, fontweight="bold", color=color)
        for index, item in enumerate(items):
            y = 0.57 - index * 0.095
            ax.scatter([x + 0.035], [y], s=42, color=color)
            ax.text(x + 0.06, y, item, va="center", fontsize=10.5)
    arrow(ax, (0.325, 0.47), (0.355, 0.47), MUTED)
    arrow(ax, (0.64, 0.47), (0.67, 0.47), MUTED)
    ax.text(0.5, 0.105, "最终回答：需要把材料做到什么指标？应调整哪些合成/加工参数？下一轮最值得做哪组实验？", ha="center", fontsize=12, fontweight="bold", color=INK)
    save(fig, "04_capability_roadmap.png")


def main() -> None:
    if not DATABASE.exists():
        raise FileNotFoundError(f"Run `mg2si ingest` first: {DATABASE}")
    configure()
    project_loop()
    with sqlite3.connect(DATABASE) as connection:
        data_readiness(connection)
        baseline_validation(connection)
    roadmap()
    print({"status": "ok", "figures": sorted(path.name for path in OUTPUT.glob("*.png"))})


if __name__ == "__main__":
    main()
