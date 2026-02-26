"""Tests for trajectory-aware time-varying range inflation attacks."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.range_parameters import RangeAttackConfig
from src.range_attack_simulation import TrajectoryAwareRangeInflationAttack
from src.range_measurement import compute_true_ranges


def test_range_attack_is_zero_before_start_and_positive_after() -> None:
    t = np.arange(0.0, 6.0, 1.0)
    uav_positions = np.column_stack((np.linspace(-10.0, 10.0, len(t)), np.zeros_like(t)))
    true_xy = np.array([0.0, 0.0])
    ranges = compute_true_ranges(uav_positions, true_xy)

    cfg = RangeAttackConfig(
        enabled=True,
        start_time=2.0,
        fake_position_xy=(0.0, 30.0),
        max_bias_m=None,
        max_slew_rate_mps=None,
        smooth_window=1,
        protocol="twr",
    )
    attacker = TrajectoryAwareRangeInflationAttack(cfg)
    out = attacker.inject_attack(ranges, t, uav_positions, true_xy)

    bias = out["bias"]
    mask = out["active_mask"]
    assert np.allclose(bias[t < cfg.start_time], 0.0)
    assert np.all(mask == (t >= cfg.start_time))
    assert np.all(bias[mask] >= -1e-12)
    assert np.std(bias[mask]) > 0.0
    assert np.allclose(out["ranges_attacked"], ranges + bias)
    assert np.all(out["extra_delay_s"] >= -1e-18)


def test_range_attack_disabled_returns_original_ranges() -> None:
    t = np.linspace(0.0, 5.0, 11)
    uav_positions = np.column_stack((5.0 * np.cos(t), 5.0 * np.sin(t)))
    true_xy = np.array([1.0, -2.0])
    ranges = compute_true_ranges(uav_positions, true_xy)

    cfg = RangeAttackConfig(enabled=False)
    out = TrajectoryAwareRangeInflationAttack(cfg).inject_attack(ranges, t, uav_positions, true_xy)

    assert np.allclose(out["ranges_attacked"], ranges)
    assert np.allclose(out["bias"], 0.0)
    assert not np.any(out["active_mask"])
