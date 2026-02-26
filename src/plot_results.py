"""结果可视化模块（中文注释版）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _plt():
    """延迟导入 matplotlib，避免无绘图需求时增加依赖开销。"""

    import matplotlib.pyplot as plt

    return plt


def plot_experiment_results(experiment: Dict[str, Any]) -> Tuple[object, np.ndarray]:
    """绘制对比实验结果图（轨迹/误差/NIS/幅值）。"""

    plt = _plt()
    baseline = experiment["baseline"]
    defense = experiment["defense"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 图1：轨迹对比（基线真实轨迹、机动真实轨迹、攻击后测量）
    ax = axes[0, 0]
    ax.plot(baseline["true_position"][:, 0], baseline["true_position"][:, 1], label="True (baseline)")
    ax.plot(defense["true_position"][:, 0], defense["true_position"][:, 1], label="True (maneuver)")
    ax.plot(
        baseline["measurements_attacked"][:, 0],
        baseline["measurements_attacked"][:, 1],
        alpha=0.5,
        label="Attacked meas",
    )
    ax.set_title("Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.legend()

    # 图2：滤波后位置误差对比
    ax = axes[0, 1]
    ax.plot(baseline["t"], baseline["filtered_error"], label="Baseline filtered error")
    ax.plot(defense["t"], defense["filtered_error"], label="Maneuver filtered error")
    ax.set_title("Position Error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error norm")
    ax.legend()

    # 图3：NIS 序列及检测阈值
    ax = axes[1, 0]
    ax.plot(baseline["t"], baseline["ekf"]["nis"], label="Baseline NIS")
    ax.plot(defense["t"], defense["ekf"]["nis"], label="Maneuver NIS")
    ax.axhline(baseline["detection"]["threshold"], color="r", linestyle="--", linewidth=1.0, label="NIS threshold")
    # 攻击起始时刻（如果传入了 config）
    if "config" in experiment:
        attack_start = float(experiment["config"].attack.start_time)
        ax.axvline(attack_start, color="k", linestyle=":", linewidth=1.0, label="Attack start")
    # 首次报警点（便于观察检测延迟）
    baseline_alarm_idx = baseline["detection"]["first_alarm_index"]
    if baseline_alarm_idx is not None:
        ax.scatter(
            baseline["t"][baseline_alarm_idx],
            baseline["ekf"]["nis"][baseline_alarm_idx],
            color="tab:red",
            s=28,
            zorder=5,
            label="Baseline alarm",
        )
    defense_alarm_idx = defense["detection"]["first_alarm_index"]
    if defense_alarm_idx is not None:
        ax.scatter(
            defense["t"][defense_alarm_idx],
            defense["ekf"]["nis"][defense_alarm_idx],
            color="tab:green",
            s=28,
            zorder=5,
            label="Maneuver alarm",
        )
    ax.set_title("NIS")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("NIS")
    ax.legend()

    # 图4：攻击幅值与机动幅值随时间变化
    ax = axes[1, 1]
    ax.plot(baseline["t"], baseline["attack_bias_norm"], label="Attack bias norm")
    ax.plot(defense["t"], np.linalg.norm(defense["maneuver_offset"], axis=1), label="Maneuver offset norm")
    ax.set_title("Attack / Maneuver Magnitude")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("magnitude")
    ax.legend()

    fig.tight_layout()
    return fig, axes


def save_figure(fig: object, path: str) -> None:
    """保存绘图结果到文件。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, bbox_inches="tight")
