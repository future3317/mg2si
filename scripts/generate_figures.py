from __future__ import annotations

from pathlib import Path
import sqlite3

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold


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
    # PPT delivery profile: reduce the overall artwork footprint while
    # keeping text point sizes unchanged, then export with a transparent
    # canvas so the slide controls the background.
    width, height = fig.get_size_inches()
    fig.set_size_inches(width * 0.86, height * 0.86, forward=True)
    fig.patch.set_alpha(0.0)
    for axis in fig.axes:
        axis.patch.set_alpha(0.0)
        for collection in axis.collections:
            if hasattr(collection, "get_sizes") and len(collection.get_sizes()):
                collection.set_sizes(collection.get_sizes() * 0.58)
        for line in axis.lines:
            line.set_linewidth(max(0.7, line.get_linewidth() * 0.78))
        for patch in axis.patches:
            if hasattr(patch, "get_boxstyle"):
                continue
            if not hasattr(patch, "get_width") or not hasattr(patch, "get_height"):
                continue
            patch_width = float(patch.get_width())
            patch_height = float(patch.get_height())
            center_x = float(patch.get_x()) + patch_width / 2.0
            center_y = float(patch.get_y()) + patch_height / 2.0
            if abs(patch_height) >= abs(patch_width):
                # Vertical bars: preserve the zero baseline and only narrow
                # the mark horizontally.
                patch.set_width(patch_width * 0.74)
                patch.set_x(center_x - patch.get_width() / 2.0)
            else:
                # Horizontal bars: preserve the zero baseline and only
                # reduce the mark height.
                patch.set_height(patch_height * 0.82)
                patch.set_y(center_y - patch.get_height() / 2.0)
    # Keep explanatory prose in the report Markdown, not inside the exported
    # chart. Suptitles, axis labels, legends, and data labels remain visible.
    for text_artist in list(fig.texts):
        position = text_artist.get_position()
        if position[1] < 0.94:
            text_artist.remove()
    for axis in fig.axes:
        x_limits = axis.get_xlim()
        y_limits = axis.get_ylim()
        if x_limits == (0.0, 1.0) and y_limits == (0.0, 1.0):
            for text_artist in list(axis.texts):
                y_position = text_artist.get_position()[1]
                if 0.84 < y_position < 0.92 or y_position < 0.10:
                    text_artist.remove()
        for text_artist in list(axis.texts):
            text = text_artist.get_text()
            if any(token in text for token in ("information-gain", "exploration", "calibration")):
                text_artist.set_text(text.splitlines()[0])
            if name in {"03_baseline_validation.png", "13_model_evidence_dashboard.png"} and "改善" in text:
                text_artist.remove()
            if name == "13_model_evidence_dashboard.png" and "细胞存活率" in text:
                text_artist.remove()
        if name == "07_multiobjective_landscape.png" and axis is fig.axes[1]:
            for text_artist in list(axis.texts):
                text_artist.remove()
        if name == "14_candidate_space_factorization.png":
            for text_artist in list(axis.texts):
                text_artist.remove()
        bars = [patch for patch in axis.patches if hasattr(patch, "get_height") and hasattr(patch, "get_width")]
        if 1 <= len(bars) <= 3:
            heights = [abs(float(patch.get_height())) for patch in bars]
            widths = [abs(float(patch.get_width())) for patch in bars]
            if heights and max(heights) >= max(widths):
                maximum = max(heights)
                if axis.get_yscale() == "log":
                    axis.set_ylim(max(1e-6, axis.get_ylim()[0]), maximum * 10.0)
                else:
                    axis.set_ylim(axis.get_ylim()[0], maximum * 1.30)
            elif widths:
                axis.set_xlim(axis.get_xlim()[0], max(widths) * 1.30)
    output_path = OUTPUT / name
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.12,
        transparent=True,
    )
    # Tight-crop the transparent canvas itself. This removes unused space
    # without changing the font sizes or adding a visible background.
    from PIL import Image

    image = Image.open(output_path).convert("RGBA")
    alpha_bbox = image.getchannel("A").getbbox()
    if alpha_bbox:
        image.crop(alpha_bbox).save(output_path, format="PNG", optimize=True)
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
    ax.set_xlabel(f"占全部 {total} 个浓度响应点的比例（%）")
    ax.set_title("当前数据就绪度", loc="left", fontsize=22, fontweight="bold", pad=20)
    ax.text(0, 1.02, "真实数据库覆盖情况；材料表征补齐后才能启用正式两阶段模型", transform=ax.transAxes, color=MUTED, fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(min(value + 1.5, 94), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontweight="bold", color=INK)
    ax.text(0.99, -0.16, "Source: mg2si.sqlite｜2026-07-26", transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    save(fig, "02_data_readiness.png")


def source_inventory(connection: sqlite3.Connection) -> None:
    frame = pd.read_sql_query(
        "SELECT source_file, COUNT(*) AS source_records FROM source_record GROUP BY source_file",
        connection,
    )
    frame["source_type"] = np.select(
        [frame["source_file"].str.lower().str.endswith(".docx"), frame["source_file"].str.lower().str.endswith((".xlsx", ".xls"))],
        ["Word", "Excel"],
        default="Configuration",
    )
    summary = frame.groupby("source_type").agg(files=("source_file", "nunique"), records=("source_records", "sum")).reindex(["Excel", "Word", "Configuration"]).dropna()
    total_records = int(frame["source_records"].sum())
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    x = np.arange(len(summary))
    bars = ax.bar(x, summary["records"], color=[BLUE, GOLD, MUTED], width=0.56)
    ax.set_xticks(x, summary.index)
    ax.set_ylabel("来源记录数")
    ax.set_ylim(0, max(summary["records"]) * 1.3)
    ax.set_title("多源数据资产组成", loc="left", fontsize=22, fontweight="bold", pad=20)
    ax.text(0, 1.02, f"当前 SQLite 来源记录 {total_records} 条｜{int(frame['source_file'].nunique())} 个来源文件｜每条记录保留文件、sheet/文档位置和原始 payload", transform=ax.transAxes, color=MUTED, fontsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(summary["records"]) * 0.025, f"{int(row['records'])} 条\n{int(row['files'])} 个文件", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.99, -0.14, "Source: source_record / source_manifest", transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    save(fig, "08_source_inventory.png")


def material_input_coverage(connection: sqlite3.Connection) -> None:
    frame = pd.read_sql_query("SELECT field, non_null_materials, total_materials, coverage_rate FROM material_input_coverage ORDER BY coverage_rate", connection)
    labels = {
        "molar_ratio_Mg_to_Si": "Mg/Si 摩尔比", "max_temp_c": "最高温度", "hold_time_min": "保温时间", "initial_pressure_atm": "初始压力", "vacuum_cycle": "真空循环", "vacuum_time_min": "真空时间", "protective_gas": "保护气氛", "ball_to_material_ratio": "球料比", "milling_cycle_time": "球磨时间", "dls_size_nm": "DLS 粒径", "pdi": "PDI", "zeta_potential_mv": "Zeta 电位", "pvp_mw": "PVP 分子量", "pvp_material_to_pvp_ratio": "材料:PVP 比", "post_treatment": "后处理"
    }
    frame["label"] = frame["field"].map(labels).fillna(frame["field"])
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    y = np.arange(len(frame))
    bars = ax.barh(y, frame["coverage_rate"] * 100, color=np.where(frame["coverage_rate"] > 0.5, BLUE, ORANGE), height=0.56)
    ax.set_yticks(y, frame["label"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("材料主表非空覆盖率（%，n=56）")
    ax.set_title("材料输入字段覆盖率", loc="left", fontsize=22, fontweight="bold", pad=20)
    ax.text(0, 1.02, "当前主表覆盖率；来源文档中的待确认工艺参数保留在 process observation，不伪装成材料主表实测值", transform=ax.transAxes, color=MUTED, fontsize=10.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value, count in zip(bars, frame["coverage_rate"] * 100, frame["non_null_materials"]):
        ax.text(min(value + 1.2, 94), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%  ({int(count)})", va="center", fontsize=9.5, fontweight="bold")
    ax.text(0.99, -0.08, "完整 BO 需要把工艺、状态和生物记录在同一材料粒度上对齐", transform=ax.transAxes, ha="right", fontsize=9.5, color=ORANGE, fontweight="bold")
    save(fig, "09_material_input_coverage.png")


def material_stage_inventory(connection: sqlite3.Connection) -> None:
    frame = pd.read_sql_query("SELECT COALESCE(product_stage, 'unclassified') AS product_stage, COUNT(*) AS materials FROM material GROUP BY product_stage ORDER BY materials DESC", connection)
    labels = {"raw_material": "原料", "intermediate": "半成品", "finished_product": "成品", "unclassified": "未分类"}
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    colors = [BLUE, CYAN, ORANGE, MUTED]
    bars = ax.bar(frame["product_stage"].map(labels).fillna(frame["product_stage"]), frame["materials"], color=colors[:len(frame)], width=0.56)
    ax.set_ylabel("材料登记数")
    ax.set_title("材料产物阶段登记情况", loc="left", fontsize=22, fontweight="bold", pad=20)
    ax.text(0, 1.02, "原料 → 球磨/后处理半成品 → PVP 成品；阶段标签用于控制工艺适用性和候选输出", transform=ax.transAxes, color=MUTED, fontsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, frame["materials"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{int(value)}", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.99, -0.14, "来源：material；source-derived 记录仍需后续编号/表征确认", transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    save(fig, "10_material_stage_inventory.png")


def prospective_space_scale(connection: sqlite3.Connection) -> None:
    frame = pd.read_sql_query("SELECT workflow_branch, full_factorial_count, generated_count FROM prospective_design_space_summary ORDER BY workflow_branch", connection)
    labels = {"synthetic": "合成路线", "commercial": "商品路线"}
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    x = np.arange(len(frame))
    bars = ax.bar(x, frame["full_factorial_count"], color=[BLUE, GOLD], width=0.56, label="理论全因子规模")
    ax.scatter(x, frame["generated_count"], color=ORANGE, s=100, zorder=4, label="当前物化候选（每路线 10,000）")
    ax.set_yscale("log")
    ax.set_xticks(x, frame["workflow_branch"].map(labels))
    ax.set_ylabel("候选组合数（对数坐标）")
    ax.set_title("全流程 prospective 候选空间规模", loc="left", fontsize=22, fontweight="bold", pad=20)
    ax.text(0, 1.02, "工艺水平笛卡尔积；理论空间用于定义搜索边界，实际每轮只从候选池中挑选 3 个实验", transform=ax.transAxes, color=MUTED, fontsize=10.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, frame["full_factorial_count"]):
        text = f"{value / 1e9:.1f}B" if value >= 1e9 else f"{value / 1e6:.2f}M"
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.3, text, ha="center", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.99, -0.14, "prospective_design_space_summary｜候选表保存可复现预算子集，不冒充完整物化", transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    save(fig, "11_prospective_space_scale.png")


def condition_response_matrix(connection: sqlite3.Connection) -> None:
    frame = pd.read_sql_query(
        """
        SELECT workflow_branch, concentration_ppm, rows,
               tumor_viability_mean, normal_viability_mean
        FROM bioassay_condition_summary
        WHERE tumor_cell_line='Huh-7' AND normal_cell_line='THLE'
        ORDER BY workflow_branch, concentration_ppm
        """,
        connection,
    )
    if frame.empty:
        return
    frame["condition"] = frame["workflow_branch"].map({"synthetic": "合成", "commercial": "商品"}) + "\n" + frame["concentration_ppm"].astype(int).astype(str) + " ppm"
    fig, ax = plt.subplots(figsize=(12, 6.8))
    x = np.arange(len(frame))
    width = 0.34
    tumor = ax.bar(x - width / 2, frame["tumor_viability_mean"], width, color=BLUE, label="肿瘤细胞存活率")
    normal = ax.bar(x + width / 2, frame["normal_viability_mean"], width, color=GOLD, label="正常细胞存活率")
    ax.set_xticks(x, frame["condition"])
    ax.set_ylabel("平均存活率（% of control）")
    ax.set_ylim(0, 115)
    ax.set_title("Huh-7 / THLE 配对响应", loc="left", fontsize=22, fontweight="bold", pad=20)
    ax.text(0, 1.02, "当前明确细胞身份的 Synthetic/Commercial 条件｜每根柱为条件均值，样本量随浓度记录保留在数据库", transform=ax.transAxes, color=MUTED, fontsize=10.5)
    ax.axhline(80, color=CYAN, linestyle="--", linewidth=1.4, label="安全参考线 80%")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bars in (tumor, normal):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.text(0.99, -0.14, "这是群体条件对比，不代表独立因果效应；后续需补齐状态表征和批次信息", transform=ax.transAxes, ha="right", fontsize=9, color=ORANGE, fontweight="bold")
    save(fig, "12_condition_response_matrix.png")


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


def gp_model(length_scale: float = 0.24) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=length_scale, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=0.08, noise_level_bounds="fixed")
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        optimizer=None,
        random_state=20260726,
    )


def gp_scope(connection: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT concentration_ppm, y_tumor_viability_pct, y_normal_viability_pct,
               material_parent_id, experiment_id, normal_measurement_group_id
        FROM bo_training
        WHERE workflow_branch = 'synthetic'
          AND tumor_cell_line = 'Huh-7'
          AND normal_cell_line = 'THLE'
          AND model_eligible_direct = 1
        """,
        connection,
    )


def gp_dose_response(connection: sqlite3.Connection) -> None:
    frame = gp_scope(connection)
    concentration = pd.to_numeric(frame["concentration_ppm"], errors="coerce").to_numpy(dtype=float)
    grid_concentration = np.geomspace(concentration.min(), concentration.max(), 240)
    grid = np.log10(grid_concentration).reshape(-1, 1)
    targets = [
        ("y_tumor_viability_pct", "肿瘤端：存活率下降", BLUE),
        ("y_normal_viability_pct", "安全端：正常细胞存活率", GOLD),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), sharey=True)
    rng = np.random.default_rng(20260726)
    for ax, (field, label, color) in zip(axes, targets):
        y = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=float)
        dose_summary = (
            frame.assign(_y=y)
            .groupby("concentration_ppm")["_y"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        dose_summary["sem"] = dose_summary["std"] / np.sqrt(dose_summary["count"])
        summary_x = np.log10(dose_summary["concentration_ppm"].to_numpy(dtype=float)).reshape(-1, 1)
        model = gp_model().fit(summary_x, dose_summary["mean"].to_numpy(dtype=float))
        mean, std = model.predict(grid, return_std=True)
        jitter = np.exp(rng.normal(0, 0.018, size=len(concentration)))
        ax.scatter(concentration * jitter, y, s=25, alpha=0.20, color=color, edgecolor="none", label="独立观测")
        ax.plot(grid_concentration, mean, color=color, linewidth=3.2, label="群体均值 GP")
        ax.fill_between(grid_concentration, mean - 1.96 * std, mean + 1.96 * std, color=color, alpha=0.18, label="95% 后验区间")
        ax.errorbar(
            dose_summary["concentration_ppm"],
            dose_summary["mean"],
            yerr=1.96 * dose_summary["sem"],
            fmt="o",
            markersize=8,
            capsize=4,
            color=color,
            markeredgecolor=INK,
            linewidth=1.8,
            zorder=4,
            label="均值 ± 95% CI",
        )
        start = float(dose_summary.iloc[0]["mean"])
        end = float(dose_summary.iloc[-1]["mean"])
        change = end - start
        ax.text(
            0.04,
            0.06,
            f"125 → 500 ppm：{change:+.1f} 个百分点",
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": PANEL, "edgecolor": color, "alpha": 0.92},
        )
        ax.set_xscale("log")
        ax.set_xticks([125, 250, 500], ["125", "250", "500"])
        ax.set_xlabel("浓度（ppm，对数坐标）")
        ax.set_title(label, loc="left", fontsize=16, fontweight="bold", color=color)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("细胞存活率（% of control）")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("群体剂量响应：效应方向已经可见", x=0.07, y=0.98, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.07, 0.895, "Synthetic × Huh-7 × THLE｜独立观测 + 均值置信区间 + GP 后验｜当前证据支持剂量效应，不替代个体材料模型", color=MUTED, fontsize=11)
    fig.text(0.99, 0.01, "当前仅 3 个剂量水平；曲线用于趋势沟通，不宣称连续剂量机制", ha="right", color=ORANGE, fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(top=0.77, bottom=0.13, wspace=0.12)
    save(fig, "05_gp_dose_response.png")


def anchored_gp_predictions(
    frame: pd.DataFrame,
    field: str,
    group_field: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    columns = ["concentration_ppm", field, group_field, "experiment_id"]
    evaluation = frame[columns].dropna(subset=[field]).copy()
    fallback = evaluation["experiment_id"].astype(str)
    evaluation["_group"] = evaluation[group_field].where(evaluation[group_field].notna(), fallback).astype(str)
    anchors = (
        evaluation[evaluation["concentration_ppm"].eq(125)]
        .groupby("_group")[field]
        .mean()
    )
    evaluation["_anchor"] = evaluation["_group"].map(anchors)
    evaluation = evaluation[
        evaluation["concentration_ppm"].gt(125) & evaluation["_anchor"].notna()
    ].copy()
    evaluation["_delta"] = evaluation[field] - evaluation["_anchor"]
    x = np.log10(evaluation["concentration_ppm"].to_numpy(dtype=float)).reshape(-1, 1)
    y_delta = evaluation["_delta"].to_numpy(dtype=float)
    groups = evaluation["_group"].to_numpy()
    predicted_delta = np.full(len(evaluation), np.nan)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train, test in splitter.split(x, y_delta, groups):
        predicted_delta[test] = gp_model(length_scale=0.20).fit(x[train], y_delta[train]).predict(x[test])
    evaluation["_prediction"] = evaluation["_anchor"] + predicted_delta
    observed = evaluation[field].to_numpy(dtype=float)
    predicted = evaluation["_prediction"].to_numpy(dtype=float)
    carry = evaluation["_anchor"].to_numpy(dtype=float)
    mae = float(mean_absolute_error(observed, predicted))
    rmse = float(mean_squared_error(observed, predicted) ** 0.5)
    rho = float(spearmanr(observed, predicted).statistic)
    carry_mae = float(mean_absolute_error(observed, carry))
    improvement = 100.0 * (carry_mae - mae) / carry_mae if carry_mae else 0.0
    return evaluation, {
        "mae": mae,
        "rmse": rmse,
        "spearman": rho,
        "anchor_carry_mae": carry_mae,
        "mae_improvement_pct": improvement,
        "rows": float(len(evaluation)),
        "groups": float(len(np.unique(groups))),
    }


def gp_cross_validation(connection: sqlite3.Connection) -> None:
    frame = gp_scope(connection)
    specifications = [
        ("y_tumor_viability_pct", "肿瘤细胞存活率", "material_parent_id", BLUE),
        ("y_normal_viability_pct", "正常细胞存活率", "normal_measurement_group_id", GOLD),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8))
    metric_rows = []
    for ax, (field, label, group_field, color) in zip(axes, specifications):
        evaluation, metrics = anchored_gp_predictions(frame, field, group_field)
        y = evaluation[field].to_numpy(dtype=float)
        predicted = evaluation["_prediction"].to_numpy(dtype=float)
        metric_rows.append({
            "target": field,
            "evaluation_regime": "125ppm_anchor_predict_250_500ppm",
            "group_field": group_field,
            **metrics,
        })
        lower = min(y.min(), predicted.min()) - 4
        upper = max(y.max(), predicted.max()) + 4
        dose_colors = evaluation["concentration_ppm"].map({250: color, 500: ORANGE})
        ax.scatter(y, predicted, s=52, alpha=0.70, color=dose_colors, edgecolor=PANEL, linewidth=0.8)
        ax.plot([lower, upper], [lower, upper], color=INK, linestyle="--", linewidth=1.5, label="理想预测")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("实测存活率（%）")
        ax.set_ylabel("交叉验证预测（%）")
        ax.set_title(label, loc="left", fontsize=16, fontweight="bold", color=color)
        improvement = metrics["mae_improvement_pct"]
        ax.text(
            0.04,
            0.95,
            f"MAE {metrics['mae']:.1f}｜RMSE {metrics['rmse']:.1f}\n"
            f"Spearman {metrics['spearman']:.2f}｜较锚点直推 {improvement:+.0f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=10.2,
            fontweight="bold",
        )
        ax.text(0.96, 0.06, "● 250 ppm   ● 500 ppm", transform=ax.transAxes, ha="right", color=MUTED, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    pd.DataFrame(metric_rows).to_sql("gp_visualization_metrics", connection, if_exists="replace", index=False)
    fig.suptitle("低剂量锚定 GP：让一次筛选支持后续剂量决策", x=0.07, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.07, 0.91, "125 ppm 实测作为材料响应锚点｜按材料/共享测量组交叉验证｜预测 250 与 500 ppm", color=MUTED, fontsize=11)
    fig.text(0.99, 0.01, "适用于已完成低剂量筛选的材料；新材料冷启动仍需组成、工艺与表征特征", ha="right", color=ORANGE, fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.13, wspace=0.24)
    save(fig, "06_gp_cross_validation.png")


def multiobjective_landscape(connection: sqlite3.Connection) -> None:
    frame = gp_scope(connection).copy()
    frame["tumor_kill_pct"] = 100.0 - pd.to_numeric(frame["y_tumor_viability_pct"], errors="coerce")
    frame["normal_safety_pct"] = pd.to_numeric(frame["y_normal_viability_pct"], errors="coerce")
    frame = frame.dropna(subset=["tumor_kill_pct", "normal_safety_pct"])
    x = frame["tumor_kill_pct"].to_numpy(dtype=float)
    y = frame["normal_safety_pct"].to_numpy(dtype=float)
    efficient = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        dominates = (x >= x[index]) & (y >= y[index]) & ((x > x[index]) | (y > y[index]))
        if dominates.any():
            efficient[index] = False
    frontier = frame.loc[efficient].sort_values("tumor_kill_pct")
    frame["decision_score"] = frame["tumor_kill_pct"] + frame["normal_safety_pct"]
    leaders = frame.nlargest(min(3, len(frame)), "decision_score").copy()
    leaders["candidate_label"] = [f"候选 {letter}" for letter in "ABC"[:len(leaders)]]
    leaders[
        [
            "candidate_label",
            "experiment_id",
            "material_parent_id",
            "concentration_ppm",
            "tumor_kill_pct",
            "normal_safety_pct",
            "decision_score",
        ]
    ].to_sql("multiobjective_observed_candidates", connection, if_exists="replace", index=False)

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    x_min = min(-18.0, float(x.min()) - 5)
    x_max = max(55.0, float(x.max()) + 5)
    y_min = min(45.0, float(y.min()) - 5)
    y_max = max(115.0, float(y.max()) + 5)
    ax.axvspan(20, x_max, ymin=max(0, (80 - y_min) / (y_max - y_min)), color=CYAN, alpha=0.10)
    ax.text(x_max - 2, y_max - 3, "目标区域\n杀伤 ≥20%｜安全 ≥80%", ha="right", va="top", color=CYAN, fontsize=11, fontweight="bold")
    dose_styles = [(125, MUTED, 44), (250, GOLD, 58), (500, ORANGE, 72)]
    for dose, color, size in dose_styles:
        subset = frame[frame["concentration_ppm"].eq(dose)]
        ax.scatter(
            subset["tumor_kill_pct"],
            subset["normal_safety_pct"],
            s=size,
            color=color,
            alpha=0.62,
            edgecolor=PANEL,
            linewidth=0.8,
            label=f"{dose} ppm",
        )
    if len(frontier) > 1:
        ax.plot(frontier["tumor_kill_pct"], frontier["normal_safety_pct"], color=CYAN, linewidth=2.5, linestyle="--", label="观测 Pareto 前沿")
        ax.scatter(frontier["tumor_kill_pct"], frontier["normal_safety_pct"], s=105, facecolor="none", edgecolor=CYAN, linewidth=2)
    for _, row in leaders.iterrows():
        ax.annotate(
            row["candidate_label"],
            (row["tumor_kill_pct"], row["normal_safety_pct"]),
            xytext=(8, 8),
            textcoords="offset points",
            color=INK,
            fontsize=9.5,
            fontweight="bold",
        )
    ax.axvline(20, color=CYAN, linewidth=1.1, alpha=0.55)
    ax.axhline(80, color=CYAN, linewidth=1.1, alpha=0.55)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("肿瘤细胞杀伤率（100 - 存活率，%）")
    ax.set_ylabel("正常细胞存活率（%）")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    fig.suptitle("历史观测双目标地图：杀伤与安全的已测证据", x=0.08, y=0.98, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.08, 0.895, "每个点为一条完整配对观测｜右上越优｜仅用于回顾历史样品，不代表 prospective 工艺候选预测", color=MUTED, fontsize=11)
    fig.text(0.99, 0.01, "这是观测证据前沿，不是虚拟候选预测；下一阶段由多目标 BO 在可制造空间内扩展", ha="right", color=ORANGE, fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(top=0.77, bottom=0.13, left=0.11, right=0.96)
    save(fig, "07_observed_pareto_reference.png")


def candidate_space_factorization(connection: sqlite3.Connection) -> None:
    summary = pd.read_sql_query(
        "SELECT workflow_branch, full_factorial_count, generated_count, factor_fields FROM prospective_design_space_summary ORDER BY workflow_branch",
        connection,
    )
    if summary.empty:
        return
    factor_fields = sorted({field.strip() for value in summary["factor_fields"] for field in str(value).split(",") if field.strip()})
    labels = {
        "raw_material_source": "原料来源", "synthesis_method": "合成方法", "material_molar_ratio_Mg_to_Si": "Mg/Si 比", "material_max_temp_c": "烧结温度", "material_hold_time_min": "保温时间", "material_vacuum_cycle": "真空循环", "material_vacuum_time_min": "真空时间", "material_initial_pressure_atm": "初始压力", "material_protective_gas": "保护气氛", "material_milling_mode": "球磨模式", "material_milling_cycle_time": "球磨时间", "material_ball_to_material_ratio": "球料比", "post_treatment_solvent_system": "后处理溶剂", "material_ultrasonic_time_h": "超声时间", "material_particle_size_target_nm": "目标粒径", "pvp_mw": "PVP 分子量", "material_to_pvp_ratio": "材料:PVP 比", "pvp_modification_method": "PVP 修饰方式", "concentration_ppm": "浓度", "post_treatment": "后处理类型",
    }
    rows = []
    for field in factor_fields:
        for branch in ["synthetic", "commercial"]:
            count = connection.execute(
                f"SELECT COUNT(DISTINCT CAST({field} AS TEXT)) FROM prospective_design_space WHERE workflow_branch = ? AND {field} IS NOT NULL",
                (branch,),
            ).fetchone()[0]
            rows.append({"field": field, "branch": branch, "levels": int(count or 0)})
    levels = pd.DataFrame(rows).pivot(index="field", columns="branch", values="levels").fillna(0)
    levels["label"] = [labels.get(field, field) for field in levels.index]
    levels = levels.sort_values(["synthetic", "commercial"], ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 11), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(top=0.75, bottom=0.14, wspace=0.24)
    ax = axes[0]
    y = np.arange(len(levels))
    height = 0.36
    ax.barh(y + height / 2, levels["synthetic"], height=height, color=BLUE, label="合成路线")
    ax.barh(y - height / 2, levels["commercial"], height=height, color=GOLD, label="商品路线")
    ax.set_yticks(y, levels["label"])
    ax.invert_yaxis()
    ax.set_xlabel("当前 10,000 个候选中的不同水平数")
    ax.set_title("候选空间的因素拆解", loc="left", fontsize=18, fontweight="bold", pad=16)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for yi, (synthetic, commercial) in enumerate(zip(levels["synthetic"], levels["commercial"])):
        if synthetic:
            ax.text(synthetic + 0.1, yi + height / 2, f"{int(synthetic)}", va="center", fontsize=8.5)
        if commercial:
            ax.text(commercial + 0.1, yi - height / 2, f"{int(commercial)}", va="center", fontsize=8.5)
    ax = axes[1]
    x = np.arange(len(summary))
    full = summary["full_factorial_count"].to_numpy(dtype=float)
    pool = summary["generated_count"].to_numpy(dtype=float)
    ax.bar(x, full, color=[BLUE, GOLD], width=0.55, label="理论全因子")
    ax.scatter(x, pool, color=ORANGE, s=95, zorder=3, label="当前候选池")
    ax.scatter(x, [3] * len(x), color=INK, marker="D", s=55, zorder=4, label="本轮实验预算")
    ax.set_yscale("log")
    ax.set_xticks(x, summary["workflow_branch"].map({"synthetic": "合成路线", "commercial": "商品路线"}))
    ax.set_ylabel("组合数（对数坐标）")
    ax.set_title("从空间到实验的预算收缩", loc="left", fontsize=18, fontweight="bold", pad=16)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for xi, value in zip(x, full):
        ax.text(xi, value * 1.6, f"{int(value):,}", ha="center", fontsize=10, fontweight="bold")
    for xi, value in zip(x, pool):
        ax.annotate("10,000 候选池", (xi, value), xytext=(0, 10), textcoords="offset points", ha="center", color=ORANGE, fontsize=9, fontweight="bold")
    for xi in x:
        ax.annotate("3 个/轮", (xi, 3), xytext=(0, -18), textcoords="offset points", ha="center", color=INK, fontsize=9, fontweight="bold")
    fig.text(0.52, 0.055, "理论空间不一次性物化；候选池可复现，每轮只下发 3 个实验", ha="center", color=ORANGE, fontsize=10, fontweight="bold")
    fig.suptitle("Prospective 候选空间：全流程工艺条件的可控搜索", x=0.06, y=0.97, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.06, 0.895, "合成路线的烧结参数只对需要合成的原料生效；商品路线对应字段保持 not_applicable", color=MUTED, fontsize=10.5)
    fig.text(0.99, 0.02, "Source: prospective_design_space / prospective_design_space_summary", ha="right", fontsize=9, color=MUTED)
    save(fig, "14_candidate_space_factorization.png")


def model_evidence_dashboard(connection: sqlite3.Connection) -> None:
    frame = pd.read_sql_query(
        "SELECT target, mae, rmse, spearman, anchor_carry_mae, mae_improvement_pct, rows, groups FROM gp_visualization_metrics",
        connection,
    )
    if frame.empty:
        return
    labels = {
        "y_tumor_viability_pct": "肿瘤细胞存活率\n（越低越好）",
        "y_normal_viability_pct": "正常细胞存活率\n（越高越安全）",
    }
    frame["label"] = frame["target"].map(labels).fillna(frame["target"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(top=0.76, bottom=0.13, wspace=0.28)
    ax = axes[0]
    y = np.arange(len(frame))
    width = 0.24
    ax.barh(y - width, frame["mae"], height=width, color=BLUE, label="MAE")
    ax.barh(y, frame["rmse"], height=width, color=ORANGE, label="RMSE")
    ax.barh(y + width, frame["anchor_carry_mae"], height=width, color=MUTED, label="基线 MAE")
    ax.set_yticks(y, frame["label"])
    ax.set_xlabel("误差（百分点，越低越好）")
    ax.set_title("预测误差：模型优于简单基线", loc="left", fontsize=18, fontweight="bold", pad=10)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for yi, row in zip(y, frame.itertuples()):
        ax.text(max(row.mae, row.rmse, row.anchor_carry_mae) + 0.8, yi, f"改善 {row.mae_improvement_pct:.1f}%", va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax = axes[1]
    sizes = np.maximum(frame["groups"].fillna(1).to_numpy(dtype=float) * 45, 80)
    scatter = ax.scatter(frame["rows"], frame["rmse"], c=frame["spearman"], s=sizes, cmap="YlOrBr", vmin=0, vmax=1, edgecolor="white", linewidth=1.5)
    for row in frame.itertuples():
        offset = (8, -28) if row.target == "y_tumor_viability_pct" else (8, 8)
        ax.annotate(labels.get(row.target, row.target), (row.rows, row.rmse), xytext=offset, textcoords="offset points", fontsize=9.5)
    ax.set_xlabel("留出评估记录数")
    ax.set_ylabel("RMSE（百分点）")
    ax.set_title("泛化证据：样本量与相关性", loc="left", fontsize=18, fontweight="bold", pad=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman 相关系数")
    fig.suptitle("模型效果诊断：可以指导下一轮，但还不是大样本外推保证", x=0.06, y=0.97, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.06, 0.89, "评估口径：以 125 ppm 为锚点预测 250/500 ppm；误差单位为细胞存活率百分点", color=MUTED, fontsize=10.5)
    fig.text(0.99, 0.03, "Source: gp_visualization_metrics｜气泡大小表示分组数", ha="right", fontsize=9, color=MUTED)
    save(fig, "13_model_evidence_dashboard.png")


def gp_cross_validation(connection: sqlite3.Connection) -> None:
    frame = gp_scope(connection)
    specifications = [
        ("y_tumor_viability_pct", "肿瘤细胞存活率", "material_parent_id", BLUE),
        ("y_normal_viability_pct", "正常细胞存活率", "normal_measurement_group_id", GOLD),
    ]
    fig = plt.figure(figsize=(14.5, 10.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.52])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]
    metric_rows = []
    for ax, (field, label, group_field, color) in zip(axes[:2], specifications):
        evaluation, metrics = anchored_gp_predictions(frame, field, group_field)
        metric_rows.append({
            "target": field,
            "evaluation_regime": "125ppm_anchor_predict_250_500ppm",
            "group_field": group_field,
            **metrics,
        })
        y = evaluation[field].to_numpy(dtype=float)
        predicted = evaluation["_prediction"].to_numpy(dtype=float)
        lower = min(y.min(), predicted.min()) - 4
        upper = max(y.max(), predicted.max()) + 4
        dose_colors = evaluation["concentration_ppm"].map({250: color, 500: ORANGE})
        ax.scatter(y, predicted, s=45, alpha=0.72, color=dose_colors, edgecolor=PANEL, linewidth=0.7)
        ax.plot([lower, upper], [lower, upper], color=INK, linestyle="--", linewidth=1.4)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("实测存活率（%）")
        ax.set_ylabel("留出预测（%）")
        ax.set_title(label, loc="left", fontsize=16, fontweight="bold", color=color)
        ax.text(0.04, 0.95, f"MAE {metrics['mae']:.1f}｜RMSE {metrics['rmse']:.1f}\nSpearman {metrics['spearman']:.2f}\n留出 n={int(metrics['rows'])}｜{int(metrics['groups'])} 组", transform=ax.transAxes, va="top", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(0.96, 0.06, "● 250 ppm   ● 500 ppm", transform=ax.transAxes, ha="right", color=MUTED, fontsize=8.5)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_sql("gp_visualization_metrics", connection, if_exists="replace", index=False)
    ax = axes[2]
    y_pos = np.arange(len(metrics_frame))
    height = 0.24
    ax.barh(y_pos - height, metrics_frame["anchor_carry_mae"], height=height, color=MUTED, label="固定锚点")
    ax.barh(y_pos, metrics_frame["mae"], height=height, color=BLUE, label="模型 MAE")
    ax.barh(y_pos + height, metrics_frame["rmse"], height=height, color=ORANGE, label="模型 RMSE")
    ax.set_yticks(y_pos, ["肿瘤\n存活率", "正常\n存活率"])
    ax.set_xlabel("误差百分点")
    ax.set_title("相对固定锚点", loc="left", fontsize=16, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for yi, row in zip(y_pos, metrics_frame.itertuples()):
        ax.text(max(row.anchor_carry_mae, row.mae, row.rmse) + 0.5, yi, f"改善 {row.mae_improvement_pct:.0f}%", va="center", fontsize=8.5, color=ORANGE, fontweight="bold")
    fig.suptitle("低剂量锚定 GP：一次筛选支持后续剂量决策", x=0.055, y=0.985, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.055, 0.92, "125 ppm 实测作为材料响应锚点｜按材料/共享测量组交叉验证｜预测 250 与 500 ppm", color=MUTED, fontsize=11)
    fig.text(0.99, 0.035, "模型用于候选排序，不替代新材料冷启动实验；右栏为 MAE/RMSE 与固定锚点的同口径比较", ha="right", color=ORANGE, fontsize=9.2, fontweight="bold")
    fig.subplots_adjust(top=0.83, bottom=0.11, left=0.08, right=0.96, wspace=0.24, hspace=0.42)
    save(fig, "06_gp_cross_validation.png")


def prospective_bo_landscape(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if not {"recommendation_candidate_pool", "recommendation"}.issubset(tables):
        return
    pool = pd.read_sql_query("SELECT * FROM recommendation_candidate_pool", connection)
    selected = pd.read_sql_query("SELECT rowid AS recommendation_order, * FROM recommendation ORDER BY rowid", connection)
    if pool.empty or selected.empty or "candidate_id" not in selected:
        return

    fields = [
        "material_molar_ratio_Mg_to_Si",
        "material_max_temp_c",
        "material_hold_time_min",
        "material_initial_pressure_atm",
        "material_milling_cycle_time",
        "material_ball_to_material_ratio",
        "pvp_mw",
        "concentration_ppm",
    ]
    labels = {
        "material_molar_ratio_Mg_to_Si": "Mg/Si",
        "material_max_temp_c": "温度",
        "material_hold_time_min": "保温",
        "material_initial_pressure_atm": "压力",
        "material_milling_cycle_time": "球磨",
        "material_ball_to_material_ratio": "球料比",
        "pvp_mw": "PVP MW",
        "concentration_ppm": "浓度",
    }
    numeric = pool[fields].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median())
    span = (numeric.max() - numeric.min()).replace(0, 1.0)
    scaled = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1.0)
    _, _, components = np.linalg.svd(scaled.to_numpy(dtype=float), full_matrices=False)
    coordinates = scaled.to_numpy(dtype=float) @ components[:2].T
    pool["process_axis_1"] = coordinates[:, 0]
    pool["process_axis_2"] = coordinates[:, 1]
    selected = selected.merge(
        pool[["candidate_id", "process_axis_1", "process_axis_2"]],
        on="candidate_id",
        how="left",
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(top=0.76, bottom=0.15, left=0.07, right=0.97, wspace=0.28)
    ax = axes[0]
    color_values = pd.to_numeric(pool["material_max_temp_c"], errors="coerce")
    scatter = ax.scatter(
        pool["process_axis_1"],
        pool["process_axis_2"],
        c=color_values,
        cmap="YlOrBr",
        s=24,
        alpha=0.42,
        linewidth=0,
    )
    candidate_colors = [BLUE, ORANGE, CYAN]
    for index, row in selected.reset_index(drop=True).iterrows():
        label = chr(ord("A") + index)
        color = candidate_colors[index % len(candidate_colors)]
        ax.scatter(row["process_axis_1"], row["process_axis_2"], marker="*", s=260, color=color, edgecolor=INK, linewidth=0.9, zorder=5)
        ax.annotate(
            f"候选 {label}\n{row.get('recommendation_role', '')}",
            (row["process_axis_1"], row["process_axis_2"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=INK,
        )
    ax.set_xlabel("工艺空间主轴 1（PCA，仅用于可视化）")
    ax.set_ylabel("工艺空间主轴 2（PCA，仅用于可视化）")
    ax.set_title("1024 个真实工艺候选的覆盖", loc="left", fontsize=18, fontweight="bold", pad=12)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("合成温度（°C）")

    ax = axes[1]
    selected_values = selected[fields].apply(pd.to_numeric, errors="coerce")
    normalized = (selected_values - numeric.min()) / span
    heatmap = ax.imshow(normalized.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(fields)), [labels[field] for field in fields], rotation=38, ha="right")
    ax.set_yticks(np.arange(len(selected)), [f"候选 {chr(ord('A') + index)}" for index in range(len(selected))])
    ax.set_title("本轮 3 个候选的工艺差异", loc="left", fontsize=18, fontweight="bold", pad=12)
    formats = {
        "material_molar_ratio_Mg_to_Si": lambda value: f"{value:.2f}",
        "material_max_temp_c": lambda value: f"{value:.0f}°C",
        "material_hold_time_min": lambda value: f"{value:.0f}m",
        "material_initial_pressure_atm": lambda value: f"{value:.2f}",
        "material_milling_cycle_time": lambda value: f"{value:.0f}m",
        "material_ball_to_material_ratio": lambda value: f"{value:.0f}:1",
        "pvp_mw": lambda value: f"{value / 1000:.0f}k",
        "concentration_ppm": lambda value: f"{value:.0f}",
    }
    for row_index in range(len(selected)):
        for column_index, field in enumerate(fields):
            value = float(selected_values.iloc[row_index, column_index])
            text_color = "white" if float(normalized.iloc[row_index, column_index]) > 0.58 else INK
            ax.text(column_index, row_index, formats[field](value), ha="center", va="center", fontsize=8.5, color=text_color, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.03, label="相对工艺水平")
    fig.suptitle("Prospective 工艺候选空间与本轮 3 个实验", x=0.055, y=0.98, ha="left", fontsize=23, fontweight="bold")
    fig.text(
        0.055,
        0.895,
        "从理论 144 亿合成组合中采样 1024 个可制造候选；当前性能分数并列，采用中心锚点、边界探索和最大最小距离选择",
        color=MUTED,
        fontsize=10.8,
    )
    fig.text(
        0.99,
        0.035,
        "这是真实工艺候选空间，不是历史样品变体；PCA 只用于展示覆盖，不代表生物性能 Pareto 前沿",
        ha="right",
        color=ORANGE,
        fontsize=9.2,
        fontweight="bold",
    )
    save(fig, "07_multiobjective_landscape.png")


def evidence_one(connection: sqlite3.Connection) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 8.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("证据一｜材料中心主动学习闭环", loc="left", fontsize=25, fontweight="bold", color=INK, pad=20)
    ax.text(0.01, 0.91, "把生物学目标转成可执行的材料与工艺调整；每轮实验结果回写数据库并更新模型", transform=ax.transAxes, color=MUTED, fontsize=12)
    cards = [
        ("1 目标定义", "杀伤效果\n正常细胞安全\n稳定性/表征约束", BLUE),
        ("2 状态反推", "粒径 / PDI / Zeta\nPVP 修饰状态\n浓度与暴露条件", CYAN),
        ("3 工艺映射", "Mg/Si 比、温度、时间\n真空/气氛、球磨\n后处理与 PVP", GOLD),
        ("4 安全约束 BO", "最大化肿瘤杀伤\n约束正常细胞存活\n惩罚不确定性", ORANGE),
        ("5 每轮实验", "候选空间采样\n约 3 个候选\n结果回写数据库", BLUE),
    ]
    card_width = 0.15
    xs = [0.03, 0.22, 0.41, 0.60, 0.79]
    for i, ((title, body, color), x) in enumerate(zip(cards, xs)):
        patch = FancyBboxPatch((x, 0.48), card_width, 0.25, boxstyle="round,pad=0.012,rounding_size=0.025", linewidth=2.5, edgecolor=color, facecolor="white")
        ax.add_patch(patch)
        ax.text(x + card_width / 2, 0.67, title, ha="center", va="center", fontsize=15, fontweight="bold", color=color)
        ax.text(x + card_width / 2, 0.57, body, ha="center", va="center", fontsize=10.5, color=MUTED, linespacing=1.45)
        if i < len(cards) - 1:
            ax.add_patch(FancyArrowPatch((x + card_width + 0.005, 0.605), (x + card_width + 0.035, 0.605), arrowstyle="-|>", mutation_scale=18, linewidth=2, color=MUTED))
    branches = [
        (0.15, "合成路线", "SHS：Mg/Si 比｜烧结温度/时间\n真空循环｜初始压力｜保护气氛", CYAN),
        (0.58, "商品路线", "不执行 SHS 字段\n商品来源｜球磨/超声｜PVP 修饰", GOLD),
    ]
    for x, title, body, color in branches:
        patch = FancyBboxPatch((x, 0.16), 0.27, 0.18, boxstyle="round,pad=0.012,rounding_size=0.025", linewidth=2, edgecolor=color, facecolor="#FFFFFF")
        ax.add_patch(patch)
        ax.text(x + 0.135, 0.28, title, ha="center", fontsize=14, fontweight="bold", color=color)
        ax.text(x + 0.135, 0.215, body, ha="center", fontsize=10.5, color=MUTED, linespacing=1.35)
    ax.add_patch(FancyArrowPatch((0.34, 0.48), (0.30, 0.345), connectionstyle="arc3,rad=0.2", arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=CYAN))
    ax.add_patch(FancyArrowPatch((0.67, 0.48), (0.72, 0.345), connectionstyle="arc3,rad=-0.2", arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=GOLD))
    ax.add_patch(FancyArrowPatch((0.865, 0.48), (0.965, 0.11), connectionstyle="arc3,rad=-0.08", arrowstyle="-", mutation_scale=18, linewidth=2, color=BLUE))
    ax.add_patch(FancyArrowPatch((0.965, 0.11), (0.055, 0.11), arrowstyle="-", mutation_scale=18, linewidth=2, color=BLUE))
    ax.add_patch(FancyArrowPatch((0.055, 0.11), (0.105, 0.48), connectionstyle="arc3,rad=-0.08", arrowstyle="-|>", mutation_scale=18, linewidth=2, color=BLUE))
    ax.text(0.5, 0.055, "核心证据链：可追溯数据库 → 条件化模型 → 受约束候选 → 新实验验证", ha="center", fontsize=14, fontweight="bold", color=BLUE)
    save(fig, "01_material_centered_loop.png")


def evidence_two(connection: sqlite3.Connection) -> None:
    total_bio = int(connection.execute("SELECT COUNT(*) FROM bioassay").fetchone()[0])
    total_material = int(connection.execute("SELECT COUNT(*) FROM material").fetchone()[0])
    total_source = int(connection.execute("SELECT COUNT(*) FROM source_record").fetchone()[0])
    mapped = float(connection.execute("SELECT AVG(CASE WHEN mapping_type != 'unmatched' THEN 1.0 ELSE 0.0 END) FROM bioassay").fetchone()[0] or 0) * 100
    normal = float(connection.execute("SELECT AVG(CASE WHEN normal_cell_line_status = 'explicit' THEN 1.0 ELSE 0.0 END) FROM bioassay").fetchone()[0] or 0) * 100
    direct = float(connection.execute("SELECT AVG(CASE WHEN model_eligible_direct = 1 THEN 1.0 ELSE 0.0 END) FROM bioassay").fetchone()[0] or 0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(top=0.78, bottom=0.16, wspace=0.28)
    ax = axes[0]
    counts = pd.Series({"来源审计记录": total_source, "生物响应记录": total_bio, "材料登记": total_material})
    bars = ax.barh(counts.index, counts.values, color=[MUTED, BLUE, GOLD], height=0.5)
    ax.set_xlabel("记录/登记数")
    ax.set_title("数据资产规模", loc="left", fontsize=19, fontweight="bold", pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, counts.values):
        ax.text(value + max(counts.values) * 0.02, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", fontweight="bold")
    ax = axes[1]
    coverage = pd.Series({"样品编号映射成功": mapped, "正常细胞身份明确": normal, "直接可用模型记录": direct}).sort_values()
    bars = ax.barh(coverage.index, coverage.values, color=[ORANGE, BLUE, CYAN], height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("占全部生物响应记录的比例（%）")
    ax.set_title("从原始记录到可建模证据", loc="left", fontsize=19, fontweight="bold", pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, coverage.values):
        ax.text(min(value + 1.5, 92), bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontweight="bold")
    fig.suptitle("证据二｜数据可追溯性与建模准备度", x=0.06, y=0.97, ha="left", fontsize=24, fontweight="bold")
    fig.text(0.06, 0.89, "先把样品、材料和生物结果对齐，再把模型预测用于候选排序；覆盖率不是模型准确率", color=MUTED, fontsize=11)
    fig.text(0.99, 0.04, "Source: mg2si.sqlite｜动态统计", ha="right", fontsize=9, color=MUTED)
    save(fig, "02_data_readiness.png")


def evidence_three(connection: sqlite3.Connection) -> None:
    frame = pd.read_sql_query("SELECT target, mae, rmse, spearman, anchor_carry_mae, mae_improvement_pct, rows, groups FROM gp_visualization_metrics", connection)
    if frame.empty:
        return
    labels = {"y_normal_viability_pct": "正常细胞存活率", "y_tumor_viability_pct": "肿瘤细胞存活率"}
    frame["label"] = frame["target"].map(labels).fillna(frame["target"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.8), gridspec_kw={"width_ratios": [1.1, 0.9]})
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.3)
    ax = axes[0]
    x = np.arange(len(frame))
    width = 0.25
    ax.bar(x - width, frame["anchor_carry_mae"], width, color=MUTED, label="固定锚点基线")
    ax.bar(x, frame["mae"], width, color=BLUE, label="模型 MAE")
    ax.bar(x + width, frame["rmse"], width, color=ORANGE, label="模型 RMSE")
    ax.set_xticks(x, frame["label"])
    ax.set_ylabel("误差（细胞存活率百分点）")
    ax.set_title("模型相对基线的增益", loc="left", fontsize=19, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for xi, row in zip(x, frame.itertuples()):
        ax.text(xi, max(row.anchor_carry_mae, row.mae, row.rmse) + 0.7, f"改善 {row.mae_improvement_pct:.1f}%", ha="center", fontsize=10, color=ORANGE, fontweight="bold")
    ax = axes[1]
    bars = ax.bar(frame["label"], frame["spearman"], color=[CYAN, GOLD], width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Spearman 相关系数")
    ax.set_title("留出条件下的排序一致性", loc="left", fontsize=19, fontweight="bold", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, row in zip(bars, frame.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, row.spearman + 0.04, f"{row.spearman:.2f}\nn={int(row.rows)}\n{int(row.groups)}组", ha="center", fontsize=10, fontweight="bold")
    fig.suptitle("证据三｜跨浓度留出验证：模型开始提供超越基线的信息", x=0.06, y=0.97, ha="left", fontsize=24, fontweight="bold")
    fig.text(0.06, 0.89, "评估口径：以 125 ppm 作为锚点，预测 250/500 ppm；当前结果用于指导下一轮候选，不替代真实实验验证", color=MUTED, fontsize=11)
    fig.text(0.99, 0.04, "Source: gp_visualization_metrics｜当前训练证据仍需随新增材料级样本扩充", ha="right", fontsize=9, color=MUTED)
    save(fig, "03_baseline_validation.png")


def main() -> None:
    if not DATABASE.exists():
        raise FileNotFoundError(f"Run `mg2si ingest` first: {DATABASE}")
    configure()
    project_loop()
    with sqlite3.connect(DATABASE) as connection:
        data_readiness(connection)
        source_inventory(connection)
        material_input_coverage(connection)
        material_stage_inventory(connection)
        prospective_space_scale(connection)
        condition_response_matrix(connection)
        candidate_space_factorization(connection)
        evidence_one(connection)
        evidence_two(connection)
        baseline_validation(connection)
        gp_dose_response(connection)
        gp_cross_validation(connection)
        evidence_three(connection)
        model_evidence_dashboard(connection)
        multiobjective_landscape(connection)
        prospective_bo_landscape(connection)
    roadmap()
    print({"status": "ok", "figures": sorted(path.name for path in OUTPUT.glob("*.png"))})


if __name__ == "__main__":
    main()
