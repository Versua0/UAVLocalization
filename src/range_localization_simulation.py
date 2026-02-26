"""End-to-end simulation for UAV-assisted range localization under range inflation attack."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from config.range_parameters import (
    RangeLocalizationExperimentConfig,
    build_default_range_localization_config,
)
from data.input_data import generate_trajectory_data
from src.localization_solver import localization_error, solve_localization_l1, solve_localization_l2
from src.range_attack_simulation import TrajectoryAwareRangeInflationAttack
from src.range_measurement import compute_true_ranges, generate_noisy_ranges
from src.utils import ensure_rng


def _solve_all(
    uav_positions: np.ndarray,
    ranges_clean: np.ndarray,
    ranges_attacked: np.ndarray,
    cfg: RangeLocalizationExperimentConfig,
) -> Dict[str, Any]:
    l2_clean = solve_localization_l2(uav_positions, ranges_clean, cfg.solver)
    l1_clean = solve_localization_l1(uav_positions, ranges_clean, cfg.solver, initial_position=l2_clean["position"])
    l2_attacked = solve_localization_l2(uav_positions, ranges_attacked, cfg.solver)
    l1_attacked = solve_localization_l1(uav_positions, ranges_attacked, cfg.solver, initial_position=l2_attacked["position"])
    return {
        "clean_l2": l2_clean,
        "clean_l1": l1_clean,
        "attacked_l2": l2_attacked,
        "attacked_l1": l1_attacked,
    }


def run_range_localization_experiment(
    config: Optional[RangeLocalizationExperimentConfig] = None,
) -> Dict[str, Any]:
    """Run the range-localization attack simulation."""

    cfg = config or build_default_range_localization_config()
    traj = generate_trajectory_data(cfg.trajectory)
    t = traj["t"]
    uav_positions = traj["position"]

    true_xy = np.asarray(cfg.target.true_xy, dtype=float)
    fake_xy = np.asarray(cfg.attack.fake_position_xy, dtype=float)
    rng = ensure_rng(cfg.random_seed)

    ranges_true = compute_true_ranges(uav_positions, true_xy)
    ranges_clean = generate_noisy_ranges(
        uav_positions=uav_positions,
        target_xy=true_xy,
        noise_std=cfg.measurement.range_noise_std,
        rng=rng,
    )

    attacker = TrajectoryAwareRangeInflationAttack(cfg.attack)
    attack_out = attacker.inject_attack(
        ranges=ranges_clean,
        t=t,
        uav_positions=uav_positions,
        true_target_xy=true_xy,
    )

    estimates = _solve_all(
        uav_positions=uav_positions,
        ranges_clean=ranges_clean,
        ranges_attacked=attack_out["ranges_attacked"],
        cfg=cfg,
    )

    metrics = {
        "clean_l2_error": localization_error(estimates["clean_l2"]["position"], true_xy),
        "clean_l1_error": localization_error(estimates["clean_l1"]["position"], true_xy),
        "attacked_l2_error_to_true": localization_error(estimates["attacked_l2"]["position"], true_xy),
        "attacked_l1_error_to_true": localization_error(estimates["attacked_l1"]["position"], true_xy),
        "attacked_l2_error_to_fake": localization_error(estimates["attacked_l2"]["position"], fake_xy),
        "attacked_l1_error_to_fake": localization_error(estimates["attacked_l1"]["position"], fake_xy),
        "attack_bias_mean": float(np.mean(attack_out["bias"])),
        "attack_bias_max": float(np.max(attack_out["bias"])) if len(attack_out["bias"]) else 0.0,
        "attack_active_ratio": float(np.mean(attack_out["active_mask"])) if len(t) else 0.0,
        "attack_delay_max_ns": float(np.max(attack_out["extra_delay_s"]) * 1e9) if len(t) else 0.0,
    }
    metrics["attack_success_l2"] = metrics["attacked_l2_error_to_fake"] < metrics["attacked_l2_error_to_true"]
    metrics["attack_success_l1"] = metrics["attacked_l1_error_to_fake"] < metrics["attacked_l1_error_to_true"]

    return {
        "config": cfg,
        "trajectory": traj,
        "uav_positions": uav_positions,
        "target_true_xy": true_xy,
        "target_fake_xy": fake_xy,
        "ranges_true": ranges_true,
        "ranges_clean": ranges_clean,
        "attack": attack_out,
        "estimates": estimates,
        "metrics": metrics,
    }


def print_range_experiment_summary(result: Dict[str, Any]) -> None:
    """Print a compact summary of the attack and localization outcome."""

    m = result["metrics"]
    print("Range-localization attack experiment summary")
    print(f"- clean_l2_error: {m['clean_l2_error']:.4f} m")
    print(f"- clean_l1_error: {m['clean_l1_error']:.4f} m")
    print(f"- attacked_l2_error_to_true: {m['attacked_l2_error_to_true']:.4f} m")
    print(f"- attacked_l2_error_to_fake: {m['attacked_l2_error_to_fake']:.4f} m")
    print(f"- attacked_l1_error_to_true: {m['attacked_l1_error_to_true']:.4f} m")
    print(f"- attacked_l1_error_to_fake: {m['attacked_l1_error_to_fake']:.4f} m")
    print(f"- attack_bias_mean/max: {m['attack_bias_mean']:.4f}/{m['attack_bias_max']:.4f} m")
    print(f"- attack_delay_max: {m['attack_delay_max_ns']:.3f} ns")
    print(f"- success_l2 / success_l1: {m['attack_success_l2']} / {m['attack_success_l1']}")


def main(
    plot: bool = True,
    show_plot: bool = False,
    save_plot: bool = True,
    figure_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Script entry point with optional plotting."""

    result = run_range_localization_experiment()
    print_range_experiment_summary(result)

    if plot:
        from src.plot_range_localization_results import plot_range_localization_results, save_figure

        fig, _ = plot_range_localization_results(result)

        if save_plot:
            default_path = Path(__file__).resolve().parents[1] / "results" / "range_localization_attack.png"
            output_path = Path(figure_path) if figure_path else default_path
            save_figure(fig, str(output_path))
            print(f"Saved figure: {output_path}")

        if show_plot:
            import matplotlib.pyplot as plt

            plt.show()

    return result


if __name__ == "__main__":
    main(plot=True, show_plot=False, save_plot=True)
