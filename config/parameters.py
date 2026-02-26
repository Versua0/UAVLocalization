"""实验参数配置模块（中文注释版）。

集中定义轨迹、传感器、攻击、EKF、检测等参数，便于统一管理与复现实验。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class TrajectoryConfig:
    """无人机名义轨迹参数。"""

    pattern: str = "circle"  # supported: circle, line
    total_time: float = 40.0
    dt: float = 0.1
    start_xy: Tuple[float, float] = (0.0, 0.0)
    radius: float = 30.0
    angular_speed: float = 0.12  # rad/s
    line_speed: float = 3.0
    line_heading_rad: float = 0.0


@dataclass
class SensorConfig:
    """传感器测量噪声参数。"""

    position_noise_std: float = 1.2


@dataclass
class ManeuverConfig:
    """主动防御随机机动参数。"""

    enabled: bool = True
    jitter_amplitude: float = 0.8
    jitter_frequency_hz: float = 1.5
    random_std: float = 0.15
    seed: int = 7


@dataclass
class AttackConfig:
    """轨迹感知攻击模型参数。"""

    enabled: bool = True
    start_time: float = 12.0
    amplitude: float = 6.0
    frequency_hz: float = 0.25
    drift_rate: float = 0.03
    phase_rad: float = 0.0
    axis_ratio: Tuple[float, float] = (1.0, 0.6)
    align_with_velocity: bool = True
    reference_mode: str = "nominal"  # nominal, true, measurement


@dataclass
class EKFConfig:
    """EKF 滤波参数。"""

    process_noise_pos: float = 0.1
    process_noise_vel: float = 0.8  # treated as acceleration std for CV model
    measurement_noise_std: float = 1.2
    init_pos_std: float = 3.0
    init_vel_std: float = 2.0


@dataclass
class DetectionConfig:
    """基于 NIS 的攻击检测参数。"""

    nis_threshold: float = 9.21  # approx chi-square(2) 99%
    consecutive_count: int = 2


@dataclass
class SimulationConfig:
    """仿真实验总配置（组合所有子配置）。"""

    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    maneuver: ManeuverConfig = field(default_factory=ManeuverConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    random_seed: int = 42


def build_default_config() -> SimulationConfig:
    """构建一份默认实验配置。"""

    return SimulationConfig()


def config_to_dict(config: SimulationConfig) -> Dict[str, Any]:
    """将 dataclass 配置展开为字典，便于打印/保存。"""

    return asdict(config)


# 提供一个模块级默认配置，方便快速实验。
DEFAULT_CONFIG = build_default_config()
