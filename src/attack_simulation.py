"""轨迹感知信号注入攻击模拟模块（中文注释版）。

攻击者根据参考轨迹方向生成时变偏移，并叠加到测量值上。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np

if __package__ in (None, ""):
    # 支持直接执行本文件时导入项目内模块。
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from config.parameters import AttackConfig
from src.utils import finite_difference, perpendicular_vectors, unit_vectors


class TrajectoryAwareAttackSimulator:
    """基于轨迹方向的二维时变偏移攻击器。"""

    def __init__(self, config: AttackConfig):
        self.config = config

    def generate_bias(
        self,
        t: np.ndarray,
        reference_position: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成攻击偏移序列与激活掩码。

        返回:
        - bias: 每个时刻注入到测量中的二维偏移
        - active_mask: 攻击是否处于激活状态
        """

        t = np.asarray(t, dtype=float)
        if t.ndim != 1:
            raise ValueError("t must be a 1D time vector")

        n = len(t)
        bias = np.zeros((n, 2), dtype=float)
        active_mask = np.zeros(n, dtype=bool)
        if n == 0 or not self.config.enabled:
            return bias, active_mask

        # 相对攻击起始时刻的时间；攻击开始前偏移强制为 0。
        elapsed = t - float(self.config.start_time)
        active_mask = elapsed >= 0.0
        elapsed_clip = np.clip(elapsed, 0.0, None)

        if reference_position is not None:
            ref = np.asarray(reference_position, dtype=float)
            if ref.shape != (n, 2):
                raise ValueError("reference_position must have shape (N, 2)")
            dt = float(np.median(np.diff(t))) if n > 1 else 1.0
            # 由参考位置数值微分得到参考速度，用于构造“沿轨迹方向”的攻击。
            velocity = finite_difference(ref, dt)
            if self.config.align_with_velocity:
                direction = unit_vectors(velocity)
            else:
                direction = np.tile([1.0, 0.0], (n, 1))
        else:
            direction = np.tile([1.0, 0.0], (n, 1))

        # 在“切向方向 + 法向方向”上分别施加时变偏移。
        perp = perpendicular_vectors(direction)
        amp = self.config.amplitude * (1.0 + self.config.drift_rate * elapsed_clip)
        omega = 2.0 * np.pi * self.config.frequency_hz
        phase = self.config.phase_rad

        par_component = amp * self.config.axis_ratio[0] * np.sin(omega * elapsed_clip + phase)
        perp_component = amp * self.config.axis_ratio[1] * np.cos(omega * elapsed_clip + phase)
        bias = par_component[:, None] * direction + perp_component[:, None] * perp
        bias[~active_mask] = 0.0
        return bias, active_mask

    def inject_attack(
        self,
        measurements: np.ndarray,
        t: np.ndarray,
        reference_position: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将攻击偏移注入原始测量，返回攻击后测量、偏移和激活掩码。"""

        measurements = np.asarray(measurements, dtype=float)
        if measurements.ndim != 2 or measurements.shape[1] != 2:
            raise ValueError("measurements must have shape (N, 2)")
        bias, active_mask = self.generate_bias(t=t, reference_position=reference_position)
        if len(bias) != len(measurements):
            raise ValueError("measurements and t must have the same length")
        attacked = measurements + bias
        return attacked, bias, active_mask
