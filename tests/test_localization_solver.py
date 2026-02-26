"""Tests for range-based localization solvers and attack distortion effect."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.range_parameters import LocalizationSolverConfig, RangeAttackConfig
from src.localization_solver import solve_localization_l1, solve_localization_l2
from src.range_attack_simulation import TrajectoryAwareRangeInflationAttack
from src.range_measurement import compute_true_ranges


def test_l2_solver_recovers_true_position_without_noise() -> None:
    uav_positions = np.array(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [-10.0, 0.0],
            [0.0, -10.0],
            [7.0, 7.0],
        ],
        dtype=float,
    )
    true_xy = np.array([2.5, -3.5], dtype=float)
    ranges = compute_true_ranges(uav_positions, true_xy)

    cfg = LocalizationSolverConfig(max_iterations=100, tolerance=1e-10)
    out = solve_localization_l2(uav_positions, ranges, cfg=cfg)

    assert out["converged"]
    assert np.linalg.norm(out["position"] - true_xy) < 1e-6


def test_targeted_range_inflation_moves_estimate_toward_fake_position() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    radius = 15.0
    uav_positions = np.column_stack((radius * np.cos(t), radius * np.sin(t)))
    true_xy = np.array([0.0, 0.0], dtype=float)
    fake_xy = np.array([0.0, 35.0], dtype=float)
    ranges = compute_true_ranges(uav_positions, true_xy)

    attack_cfg = RangeAttackConfig(
        enabled=True,
        start_time=0.0,
        fake_position_xy=tuple(fake_xy.tolist()),
        max_bias_m=None,
        max_slew_rate_mps=None,
        smooth_window=1,
    )
    attacked = TrajectoryAwareRangeInflationAttack(attack_cfg).inject_attack(ranges, t, uav_positions, true_xy)

    solver_cfg = LocalizationSolverConfig(max_iterations=100, tolerance=1e-9)
    est_clean = solve_localization_l2(uav_positions, ranges, cfg=solver_cfg)
    est_attacked_l2 = solve_localization_l2(uav_positions, attacked["ranges_attacked"], cfg=solver_cfg)
    est_attacked_l1 = solve_localization_l1(
        uav_positions,
        attacked["ranges_attacked"],
        cfg=solver_cfg,
        initial_position=est_attacked_l2["position"],
    )

    clean_err = np.linalg.norm(est_clean["position"] - true_xy)
    attacked_l2_to_true = np.linalg.norm(est_attacked_l2["position"] - true_xy)
    attacked_l2_to_fake = np.linalg.norm(est_attacked_l2["position"] - fake_xy)
    attacked_l1_to_true = np.linalg.norm(est_attacked_l1["position"] - true_xy)
    attacked_l1_to_fake = np.linalg.norm(est_attacked_l1["position"] - fake_xy)

    assert clean_err < 1e-5
    assert attacked_l2_to_true > 1.0
    assert attacked_l2_to_fake < attacked_l2_to_true
    assert attacked_l1_to_fake < attacked_l1_to_true
