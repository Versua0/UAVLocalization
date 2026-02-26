"""Trajectory-aware time-varying range inflation attack."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from config.range_parameters import RangeAttackConfig
from src.range_measurement import compute_true_ranges
from src.utils import rolling_mean


class TrajectoryAwareRangeInflationAttack:
    """Inject a causal, non-negative, time-varying range inflation bias."""

    def __init__(self, config: RangeAttackConfig):
        self.config = config

    def _desired_bias_from_fake_position(
        self,
        uav_positions: np.ndarray,
        true_target_xy: np.ndarray,
    ) -> np.ndarray:
        fake_xy = np.asarray(self.config.fake_position_xy, dtype=float)
        true_xy = np.asarray(true_target_xy, dtype=float)
        if fake_xy.shape != (2,):
            raise ValueError("fake_position_xy must have shape (2,)")
        desired = compute_true_ranges(uav_positions, fake_xy) - compute_true_ranges(uav_positions, true_xy)
        # Range inflation attack can only add delay / increase distance.
        return np.maximum(desired, 0.0)

    def _apply_time_constraints(self, desired_bias: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(t, dtype=float)
        desired_bias = np.asarray(desired_bias, dtype=float)
        if t.ndim != 1:
            raise ValueError("t must be 1D")
        if desired_bias.shape != t.shape:
            raise ValueError("desired_bias and t must have the same shape")

        bias = desired_bias.copy()
        active_mask = t >= float(self.config.start_time)
        bias[~active_mask] = 0.0

        if self.config.max_bias_m is not None:
            bias = np.minimum(bias, float(self.config.max_bias_m))

        if self.config.smooth_window > 1 and len(bias) > 0:
            smoothed = rolling_mean(bias, int(self.config.smooth_window))
            # Preserve pre-attack zeros exactly.
            smoothed[~active_mask] = 0.0
            bias = np.maximum(smoothed, 0.0)

        if self.config.max_slew_rate_mps is not None and len(bias) > 1:
            max_rate = float(self.config.max_slew_rate_mps)
            # Causal forward slew-rate limiting to model relay timing agility.
            for i in range(1, len(bias)):
                dt = max(float(t[i] - t[i - 1]), 1e-12)
                max_step = max_rate * dt
                upper = bias[i - 1] + max_step
                lower = max(bias[i - 1] - max_step, 0.0)
                bias[i] = min(max(bias[i], lower), upper)
                if not active_mask[i]:
                    bias[i] = 0.0

        return bias, active_mask

    def _extra_delay_seconds(self, bias_m: np.ndarray) -> np.ndarray:
        protocol = self.config.protocol.lower()
        c = float(self.config.propagation_speed_mps)
        if c <= 0:
            raise ValueError("propagation_speed_mps must be positive")
        if protocol == "single":
            return bias_m / c
        if protocol == "twr":
            # Round-trip timing inflation corresponds to half-distance per one-way delay.
            return 2.0 * bias_m / c
        raise ValueError("protocol must be 'single' or 'twr'")

    def inject_attack(
        self,
        ranges: np.ndarray,
        t: np.ndarray,
        uav_positions: np.ndarray,
        true_target_xy: np.ndarray,
    ) -> Dict[str, Any]:
        """Apply a fake-position-driven range inflation attack to scalar ranges."""

        ranges = np.asarray(ranges, dtype=float)
        t = np.asarray(t, dtype=float)
        uav_positions = np.asarray(uav_positions, dtype=float)
        true_target_xy = np.asarray(true_target_xy, dtype=float)

        if ranges.ndim != 1:
            raise ValueError("ranges must be 1D")
        if t.ndim != 1:
            raise ValueError("t must be 1D")
        if uav_positions.ndim != 2 or uav_positions.shape[1] != 2:
            raise ValueError("uav_positions must have shape (N, 2)")
        if len(ranges) != len(t) or len(ranges) != len(uav_positions):
            raise ValueError("ranges, t, and uav_positions must share the same length")

        if not self.config.enabled:
            zeros = np.zeros_like(ranges)
            return {
                "ranges_attacked": ranges.copy(),
                "bias": zeros,
                "desired_bias": zeros,
                "active_mask": np.zeros_like(ranges, dtype=bool),
                "extra_delay_s": zeros,
            }

        desired_bias = self._desired_bias_from_fake_position(uav_positions, true_target_xy)
        bias, active_mask = self._apply_time_constraints(desired_bias, t)
        ranges_attacked = ranges + bias

        return {
            "ranges_attacked": ranges_attacked,
            "bias": bias,
            "desired_bias": desired_bias,
            "active_mask": active_mask,
            "extra_delay_s": self._extra_delay_seconds(bias),
        }
