"""数据输入与仿真数据生成模块（中文注释版）。

本文件既支持生成仿真轨迹/测量，也预留了从 CSV 读取真实传感器数据的入口。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional, Union

import numpy as np

if __package__ in (None, ""):
    # 允许直接运行本文件时，仍然可以按项目根目录导入 config/src 包。
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from config.parameters import SensorConfig, SimulationConfig, TrajectoryConfig
from src.utils import ensure_rng


def generate_time_axis(total_time: float, dt: float) -> np.ndarray:
    """生成仿真时间轴 [0, total_time]，步长为 dt。"""

    if dt <= 0:
        raise ValueError("dt must be positive")
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    return np.arange(0.0, total_time + 1e-12, dt, dtype=float)


def _circle_trajectory(t: np.ndarray, cfg: TrajectoryConfig) -> tuple[np.ndarray, np.ndarray]:
    """生成圆周轨迹及对应速度。"""

    x0, y0 = cfg.start_xy
    r = cfg.radius
    w = cfg.angular_speed
    x = x0 + r * np.cos(w * t)
    y = y0 + r * np.sin(w * t)
    vx = -r * w * np.sin(w * t)
    vy = r * w * np.cos(w * t)
    return np.column_stack((x, y)), np.column_stack((vx, vy))


def _line_trajectory(t: np.ndarray, cfg: TrajectoryConfig) -> tuple[np.ndarray, np.ndarray]:
    """生成直线匀速轨迹及对应速度。"""

    x0, y0 = cfg.start_xy
    v = cfg.line_speed
    heading = cfg.line_heading_rad
    vx = np.full_like(t, v * np.cos(heading), dtype=float)
    vy = np.full_like(t, v * np.sin(heading), dtype=float)
    x = x0 + vx * t
    y = y0 + vy * t
    return np.column_stack((x, y)), np.column_stack((vx, vy))


def generate_trajectory_data(cfg: TrajectoryConfig) -> Dict[str, np.ndarray]:
    """根据配置生成名义轨迹（位置+速度+时间）。"""

    t = generate_time_axis(cfg.total_time, cfg.dt)
    pattern = cfg.pattern.lower()
    if pattern == "circle":
        position, velocity = _circle_trajectory(t, cfg)
    elif pattern == "line":
        position, velocity = _line_trajectory(t, cfg)
    else:
        raise ValueError(f"Unsupported trajectory pattern: {cfg.pattern}")
    return {"t": t, "position": position, "velocity": velocity}


def _resolve_noise_std(sensor: Union[SensorConfig, float]) -> float:
    """兼容两种传参方式：SensorConfig 或直接给噪声标准差。"""

    if isinstance(sensor, SensorConfig):
        return float(sensor.position_noise_std)
    return float(sensor)


def generate_sensor_measurements(
    true_position: np.ndarray,
    sensor: Union[SensorConfig, float],
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """在真实位置上叠加高斯噪声，生成二维位置测量。"""

    true_position = np.asarray(true_position, dtype=float)
    if true_position.ndim != 2 or true_position.shape[1] != 2:
        raise ValueError("true_position must have shape (N, 2)")
    noise_std = _resolve_noise_std(sensor)
    generator = ensure_rng(rng)
    noise = generator.normal(loc=0.0, scale=noise_std, size=true_position.shape)
    return true_position + noise


def load_sensor_data(csv_path: Union[str, Path], delimiter: str = ",") -> np.ndarray:
    """从 CSV 文件读取传感器位置数据（仅取前两列 x/y）。"""

    path = Path(csv_path)
    data = np.loadtxt(path, delimiter=delimiter)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 2:
        raise ValueError("Sensor data must contain at least two columns (x, y)")
    return data[:, :2]


def build_nominal_dataset(
    config: Optional[SimulationConfig] = None,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Dict[str, np.ndarray]:
    """快速构造一份“名义轨迹 + 含噪测量”的基础数据集。"""

    cfg = config or SimulationConfig()
    trajectory = generate_trajectory_data(cfg.trajectory)
    # 这里的 measurements 是未受攻击的基础测量。
    measurements = generate_sensor_measurements(trajectory["position"], cfg.sensor, rng=rng)
    return {
        "t": trajectory["t"],
        "nominal_position": trajectory["position"],
        "nominal_velocity": trajectory["velocity"],
        "measurements": measurements,
    }
