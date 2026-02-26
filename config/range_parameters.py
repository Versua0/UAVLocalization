"""Configuration for UAV-assisted range localization attack experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from config.parameters import TrajectoryConfig


@dataclass
class TargetNodeConfig:
    """True position of the node to localize."""

    true_xy: Tuple[float, float] = (4.0, -6.0)


@dataclass
class RangeMeasurementConfig:
    """Noise model for scalar range measurements."""

    range_noise_std: float = 0.25


@dataclass
class RangeAttackConfig:
    """Time-varying range inflation attack driven by a fake target position."""

    enabled: bool = True
    start_time: float = 10.0
    fake_position_xy: Tuple[float, float] = (22.0, 16.0)
    max_bias_m: float | None = 60.0
    max_slew_rate_mps: float | None = 8.0
    smooth_window: int = 3
    protocol: str = "twr"  # "single" or "twr"
    propagation_speed_mps: float = 299_792_458.0


@dataclass
class LocalizationSolverConfig:
    """Nonlinear least-squares / IRLS solver parameters."""

    max_iterations: int = 50
    tolerance: float = 1e-6
    damping: float = 1e-6
    irls_max_outer_iterations: int = 8
    irls_epsilon: float = 1e-3


@dataclass
class RangeLocalizationExperimentConfig:
    """Full configuration for range-localization attack simulation."""

    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    target: TargetNodeConfig = field(default_factory=TargetNodeConfig)
    measurement: RangeMeasurementConfig = field(default_factory=RangeMeasurementConfig)
    attack: RangeAttackConfig = field(default_factory=RangeAttackConfig)
    solver: LocalizationSolverConfig = field(default_factory=LocalizationSolverConfig)
    random_seed: int = 42


def build_default_range_localization_config() -> RangeLocalizationExperimentConfig:
    """Return a runnable default config."""

    cfg = RangeLocalizationExperimentConfig()
    # Favor a non-degenerate UAV path for multilateration.
    cfg.trajectory.pattern = "circle"
    cfg.trajectory.radius = 25.0
    cfg.trajectory.angular_speed = 0.18
    cfg.trajectory.total_time = 35.0
    cfg.trajectory.dt = 0.5
    return cfg
