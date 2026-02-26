"""基于新息统计量（NIS）的攻击检测模块（中文注释版）。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np

if __package__ in (None, ""):
    # 支持直接执行本文件时导入项目内配置模块。
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from config.parameters import DetectionConfig


def compute_nis(innovation: np.ndarray, innovation_covariance: np.ndarray) -> float:
    """计算单个时刻的 NIS = v^T S^-1 v。"""

    innovation = np.asarray(innovation, dtype=float)
    innovation_covariance = np.asarray(innovation_covariance, dtype=float)
    return float(innovation.T @ np.linalg.solve(innovation_covariance, innovation))


def compute_nis_series(innovations: np.ndarray, innovation_covariances: np.ndarray) -> np.ndarray:
    """批量计算 NIS 序列。"""

    innovations = np.asarray(innovations, dtype=float)
    innovation_covariances = np.asarray(innovation_covariances, dtype=float)
    if innovations.ndim != 2 or innovations.shape[1] != 2:
        raise ValueError("innovations must have shape (N, 2)")
    if innovation_covariances.shape != (len(innovations), 2, 2):
        raise ValueError("innovation_covariances must have shape (N, 2, 2)")
    nis = np.zeros(len(innovations), dtype=float)
    for i in range(len(innovations)):
        nis[i] = compute_nis(innovations[i], innovation_covariances[i])
    return nis


class NISAttackDetector:
    """基于阈值与连续超限计数的简单攻击检测器。"""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def detect(self, nis_values: np.ndarray) -> Dict[str, Optional[object]]:
        """对 NIS 序列做检测，返回报警掩码和首个报警位置。"""

        nis_values = np.asarray(nis_values, dtype=float)
        threshold_exceeded = nis_values > float(self.config.nis_threshold)
        alarms = np.zeros_like(threshold_exceeded, dtype=bool)

        count = 0
        for i, exceeded in enumerate(threshold_exceeded):
            if exceeded:
                count += 1
            else:
                count = 0
            # 连续多次超阈值后才报警，可降低偶发噪声误报。
            if count >= int(self.config.consecutive_count):
                alarms[i] = True

        alarm_indices = np.flatnonzero(alarms)
        first_alarm_index = int(alarm_indices[0]) if len(alarm_indices) else None

        return {
            "nis": nis_values,
            "threshold": float(self.config.nis_threshold),
            "threshold_exceeded": threshold_exceeded,
            "alarms": alarms,
            "alarm_indices": alarm_indices,
            "first_alarm_index": first_alarm_index,
            "attack_detected": first_alarm_index is not None,
        }
