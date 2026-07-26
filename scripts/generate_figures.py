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
    fig.suptitle("群体剂量响应：效应方向已经可见", x=0.07, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.07, 0.91, "Synthetic × Huh-7 × THLE｜独立观测 + 均值置信区间 + GP 后验｜当前证据支持剂量效应，不替代个体材料模型", color=MUTED, fontsize=11)
    fig.text(0.99, 0.01, "当前仅 3 个剂量水平；曲线用于趋势沟通，不宣称连续剂量机制", ha="right", color=ORANGE, fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.13, wspace=0.12)
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
    fig.suptitle("双目标决策地图：在杀伤与安全之间寻找 Pareto 候选", x=0.08, ha="left", fontsize=22, fontweight="bold")
    fig.text(0.08, 0.91, "每个点为一条完整配对观测｜右上越优｜候选 A–C 仅按当前双目标证据标记", color=MUTED, fontsize=11)
    fig.text(0.99, 0.01, "这是观测证据前沿，不是虚拟候选预测；下一阶段由多目标 BO 在可制造空间内扩展", ha="right", color=ORANGE, fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.11, right=0.96)
    save(fig, "07_multiobjective_landscape.png")


def main() -> None:
    if not DATABASE.exists():
        raise FileNotFoundError(f"Run `mg2si ingest` first: {DATABASE}")
    configure()
    project_loop()
    with sqlite3.connect(DATABASE) as connection:
        data_readiness(connection)
        baseline_validation(connection)
        gp_dose_response(connection)
        gp_cross_validation(connection)
        multiobjective_landscape(connection)
    roadmap()
    print({"status": "ok", "figures": sorted(path.name for path in OUTPUT.glob("*.png"))})


if __name__ == "__main__":
    main()
