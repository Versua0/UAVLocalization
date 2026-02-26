"""主动随机机动模块

在名义轨迹上叠加高频抖动与随机扰动，用于提升攻击者预测难度。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional, Union

import numpy as np

if __package__ in (None, ""):
    # 支持直接运行脚本时的项目根目录导入。
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from config.parameters import ManeuverConfig
from src.utils import ensure_rng, finite_difference, rolling_mean


def generate_random_maneuver_offsets(
    t: np.ndarray,
    cfg: ManeuverConfig,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """生成二维随机机动偏移量序列。

    偏移由“周期抖动 + 平滑随机噪声”组成。
    """

    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be a 1D time vector")
    if len(t) == 0:
        return np.zeros((0, 2), dtype=float)

    if not cfg.enabled:
        return np.zeros((len(t), 2), dtype=float)

    generator = ensure_rng(cfg.seed if rng is None else rng)
    phases = generator.uniform(0.0, 2.0 * np.pi, size=2)
    omega = 2.0 * np.pi * cfg.jitter_frequency_hz

    # 周期项：模拟高频抖动（攻击者较难准确预测）
    periodic = cfg.jitter_amplitude * np.column_stack(
        (
            np.sin(omega * t + phases[0]),
            np.cos(omega * t + phases[1]),
        )
    )
    stochastic = generator.normal(loc=0.0, scale=cfg.random_std, size=(len(t), 2))
    if len(t) > 1:
        dt = max(float(t[1] - t[0]), 1e-6)
        smooth_window = max(2, int(round(0.4 / dt)))
    else:
        smooth_window = 2
    stochastic = rolling_mean(stochastic, window=smooth_window)

    # 合成机动偏移；首样本置零，避免仿真起点突变。
    offsets = periodic + stochastic
    offsets[0] = 0.0
    return offsets


def apply_active_maneuver(
    nominal_position: np.ndarray,
    t: np.ndarray,
    cfg: ManeuverConfig,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Dict[str, np.ndarray]:
    """将主动机动偏移叠加到名义轨迹，并计算机动后的速度。"""

    nominal_position = np.asarray(nominal_position, dtype=float)
    t = np.asarray(t, dtype=float)
    if nominal_position.ndim != 2 or nominal_position.shape[1] != 2:
        raise ValueError("nominal_position must have shape (N, 2)")
    if len(nominal_position) != len(t):
        raise ValueError("nominal_position and t must have same length")

    offsets = generate_random_maneuver_offsets(t, cfg, rng=rng)
    maneuvered_position = nominal_position + offsets
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    # 通过数值微分近似速度，便于后续攻击器按轨迹方向对齐。
    maneuvered_velocity = finite_difference(maneuvered_position, dt)

    return {
        "position": maneuvered_position,
        "velocity": maneuvered_velocity,
        "offset": offsets,
    }
