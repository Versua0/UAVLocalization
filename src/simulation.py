"""仿真主流程模块（中文注释版）。

负责串联：轨迹生成 -> 主动机动 -> 攻击注入 -> EKF估计 -> NIS检测 -> 指标统计。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np

if __package__ in (None, ""):
    # 支持使用 `python src/simulation.py` 直接运行。
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from config.parameters import SimulationConfig, build_default_config
from data.input_data import generate_sensor_measurements, generate_trajectory_data
from src.active_maneuver import apply_active_maneuver
from src.attack_simulation import TrajectoryAwareAttackSimulator
from src.detection import NISAttackDetector
from src.ekf_filter import ExtendedKalmanFilter2D
from src.utils import compute_rmse, ensure_rng, row_norm


def _select_attack_reference(
    reference_mode: str,
    nominal_position: np.ndarray,
    true_position: np.ndarray,
    measurements_clean: np.ndarray,
) -> np.ndarray:
    """选择攻击者所使用的参考轨迹。

    - nominal: 攻击者只知道名义轨迹（更容易被随机机动干扰）
    - true: 攻击者“过强”地知道真实轨迹（用于上界对比）
    - measurement: 攻击者依据测量估计轨迹
    """

    mode = reference_mode.lower()
    if mode == "nominal":
        return nominal_position
    if mode == "true":
        return true_position
    if mode == "measurement":
        return measurements_clean
    raise ValueError(f"Unsupported attack reference mode: {reference_mode}")


def run_single_scenario(
    config: Optional[SimulationConfig] = None,
    use_active_maneuver: bool = False,
) -> Dict[str, Any]:
    """运行单个场景（基线或主动机动防御场景）。"""

    cfg = config or build_default_config()
    # 1) 生成名义轨迹
    nominal = generate_trajectory_data(cfg.trajectory)
    t = nominal["t"]
    nominal_position = nominal["position"]
    nominal_velocity = nominal["velocity"]

    # 2) 根据是否启用主动防御，决定真实飞行轨迹
    maneuver_offset = np.zeros_like(nominal_position)
    true_position = nominal_position.copy()
    true_velocity = nominal_velocity.copy()
    if use_active_maneuver and cfg.maneuver.enabled:
        maneuver_out = apply_active_maneuver(nominal_position, t, cfg.maneuver)
        true_position = maneuver_out["position"]
        true_velocity = maneuver_out["velocity"]
        maneuver_offset = maneuver_out["offset"]

    # 为两个场景使用不同随机种子，避免完全相同噪声序列造成“偶然对齐”。
    measurement_seed = cfg.random_seed + (101 if use_active_maneuver else 0)
    rng = ensure_rng(measurement_seed)
    measurements_clean = generate_sensor_measurements(true_position, cfg.sensor, rng=rng)

    # 3) 攻击者根据参考轨迹注入时变偏移
    attack_reference = _select_attack_reference(
        cfg.attack.reference_mode,
        nominal_position=nominal_position,
        true_position=true_position,
        measurements_clean=measurements_clean,
    )
    attacker = TrajectoryAwareAttackSimulator(cfg.attack)
    measurements_attacked, attack_bias, attack_mask = attacker.inject_attack(
        measurements=measurements_clean,
        t=t,
        reference_position=attack_reference,
    )

    # 4) EKF 对受攻击测量进行状态估计
    ekf = ExtendedKalmanFilter2D(cfg.ekf, dt=cfg.trajectory.dt)
    ekf_output = ekf.run(measurements_attacked)

    # 5) 基于 NIS 做异常检测
    detector = NISAttackDetector(cfg.detection)
    detection = detector.detect(ekf_output["nis"])

    # 6) 统计误差和检测指标
    raw_error = row_norm(measurements_attacked - true_position)
    filtered_error = row_norm(ekf_output["positions"] - true_position)
    attack_bias_norm = row_norm(attack_bias)

    first_alarm_index = detection["first_alarm_index"]
    first_alarm_time = float(t[first_alarm_index]) if first_alarm_index is not None else None
    detection_delay = None
    if first_alarm_time is not None and cfg.attack.enabled:
        detection_delay = float(first_alarm_time - cfg.attack.start_time)

    metrics = {
        "raw_rmse": compute_rmse(raw_error),
        "filtered_rmse": compute_rmse(filtered_error),
        "max_filtered_error": float(np.max(filtered_error)),
        "mean_nis": float(np.mean(ekf_output["nis"])),
        "peak_nis": float(np.max(ekf_output["nis"])),
        "attack_active_ratio": float(np.mean(attack_mask)) if len(attack_mask) else 0.0,
        "detection_rate_during_attack": float(np.mean(detection["threshold_exceeded"][attack_mask])) if np.any(attack_mask) else 0.0,
        "first_alarm_time": first_alarm_time,
        "detection_delay": detection_delay,
    }

    return {
        "t": t,
        "nominal_position": nominal_position,
        "nominal_velocity": nominal_velocity,
        "true_position": true_position,
        "true_velocity": true_velocity,
        "maneuver_offset": maneuver_offset,
        "measurements_clean": measurements_clean,
        "measurements_attacked": measurements_attacked,
        "attack_bias": attack_bias,
        "attack_bias_norm": attack_bias_norm,
        "attack_mask": attack_mask,
        "ekf": ekf_output,
        "detection": detection,
        "raw_error": raw_error,
        "filtered_error": filtered_error,
        "metrics": metrics,
        "scenario_name": "maneuver" if use_active_maneuver else "baseline",
    }


def summarize_comparison(results: Dict[str, Any]) -> Dict[str, float]:
    """汇总基线场景与防御场景的关键指标，便于快速对比。"""

    baseline_metrics = results["baseline"]["metrics"]
    defense_metrics = results["defense"]["metrics"]
    return {
        "baseline_filtered_rmse": float(baseline_metrics["filtered_rmse"]),
        "defense_filtered_rmse": float(defense_metrics["filtered_rmse"]),
        "rmse_improvement": float(baseline_metrics["filtered_rmse"] - defense_metrics["filtered_rmse"]),
        "baseline_peak_nis": float(baseline_metrics["peak_nis"]),
        "defense_peak_nis": float(defense_metrics["peak_nis"]),
    }


def run_comparison_experiment(config: Optional[SimulationConfig] = None) -> Dict[str, Any]:
    """运行“无机动基线”与“主动机动防御”两组对比实验。"""

    cfg = config or build_default_config()
    baseline = run_single_scenario(cfg, use_active_maneuver=False)
    defense = run_single_scenario(cfg, use_active_maneuver=True)
    summary = summarize_comparison({"baseline": baseline, "defense": defense})
    return {"config": cfg, "baseline": baseline, "defense": defense, "summary": summary}


def print_experiment_summary(experiment: Dict[str, Any]) -> None:
    """打印简要实验结果。"""

    print("Experiment summary")
    for key, value in experiment["summary"].items():
        print(f"- {key}: {value:.4f}")


def main(
    plot: bool = True,
    show_plot: bool = False,
    save_plot: bool = True,
    figure_path: Optional[str] = None,
) -> Dict[str, Any]:
    """脚本入口：运行对比实验，并按需绘图。"""

    experiment = run_comparison_experiment()
    print_experiment_summary(experiment)
    if plot:
        from src.plot_results import plot_experiment_results, save_figure

        fig, _ = plot_experiment_results(experiment)

        if save_plot:
            default_path = Path(__file__).resolve().parents[1] / "results" / "comparison_experiment.png"
            output_path = Path(figure_path) if figure_path else default_path
            save_figure(fig, str(output_path))
            print(f"Saved figure: {output_path}")

        if show_plot:
            import matplotlib.pyplot as plt

            plt.show()
    return experiment


if __name__ == "__main__":
    # 直接运行时默认保存可视化图片；如需弹窗显示可改为 show_plot=True。
    main(plot=True, show_plot=False, save_plot=True)
