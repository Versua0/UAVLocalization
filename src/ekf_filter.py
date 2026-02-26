"""扩展卡尔曼滤波（EKF）模块（中文注释版）。

这里使用的是二维位置观测 + 常速度状态模型。虽然模型线性，也保留 EKF 接口形式，
方便后续扩展为非线性观测/非线性运动模型。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np

if __package__ in (None, ""):
    # 支持直接运行本文件时导入项目模块。
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from config.parameters import EKFConfig


class ExtendedKalmanFilter2D:
    """二维位置跟踪 EKF（状态: [x, y, vx, vy]，观测: [x, y]）。"""

    def __init__(self, config: EKFConfig, dt: float):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.config = config
        self.dt = float(dt)
        self.dim_x = 4
        self.dim_z = 2
        self.x = np.zeros(self.dim_x, dtype=float)
        self.P = np.eye(self.dim_x, dtype=float)
        self.R = (self.config.measurement_noise_std ** 2) * np.eye(self.dim_z, dtype=float)

    def _state_transition(self) -> np.ndarray:
        """常速度模型状态转移矩阵 F。"""

        dt = self.dt
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _process_noise(self) -> np.ndarray:
        """构造过程噪声协方差 Q。

        使用离散化加速度噪声模型，并附加一个小的位置噪声项。
        """

        dt = self.dt
        sigma_a = float(self.config.process_noise_vel)
        g = np.array(
            [
                [0.5 * dt**2, 0.0],
                [0.0, 0.5 * dt**2],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=float,
        )
        qa = (sigma_a**2) * np.eye(2, dtype=float)
        q = g @ qa @ g.T
        q += np.diag(
            [
                self.config.process_noise_pos**2,
                self.config.process_noise_pos**2,
                1e-9,
                1e-9,
            ]
        )
        return q

    @staticmethod
    def _measurement_function(x: np.ndarray) -> np.ndarray:
        """观测函数 h(x)：只观测位置分量。"""

        return x[:2].copy()

    @staticmethod
    def _measurement_jacobian(_: np.ndarray) -> np.ndarray:
        """观测函数对状态的雅可比矩阵 H。"""

        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def reset(self, x0: Optional[np.ndarray] = None, p0: Optional[np.ndarray] = None) -> None:
        """重置滤波器状态与协方差。"""

        if x0 is None:
            self.x = np.zeros(self.dim_x, dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (self.dim_x,):
                raise ValueError("x0 must have shape (4,)")
            self.x = x0.copy()

        if p0 is None:
            self.P = np.diag(
                [
                    self.config.init_pos_std**2,
                    self.config.init_pos_std**2,
                    self.config.init_vel_std**2,
                    self.config.init_vel_std**2,
                ]
            )
        else:
            p0 = np.asarray(p0, dtype=float)
            if p0.shape != (self.dim_x, self.dim_x):
                raise ValueError("p0 must have shape (4, 4)")
            self.P = p0.copy()

    def predict(self) -> None:
        """预测步骤：x(k|k-1), P(k|k-1)。"""

        f = self._state_transition()
        q = self._process_noise()
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + q

    def update(self, z: np.ndarray) -> Dict[str, np.ndarray | float]:
        """更新步骤：融合当前观测，并返回创新/NIS 等诊断量。"""

        z = np.asarray(z, dtype=float)
        if z.shape != (self.dim_z,):
            raise ValueError("z must have shape (2,)")

        h = self._measurement_function(self.x)
        H = self._measurement_jacobian(self.x)
        # 创新（残差）：实际观测 - 预测观测
        innovation = z - h
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        I = np.eye(self.dim_x, dtype=float)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T

        # NIS（Normalized Innovation Squared）常用于异常检测。
        nis = float(innovation.T @ np.linalg.solve(S, innovation))
        return {
            "innovation": innovation,
            "innovation_covariance": S,
            "nis": nis,
            "predicted_measurement": h,
            "state": self.x.copy(),
            "covariance": self.P.copy(),
        }

    def run(
        self,
        measurements: np.ndarray,
        x0: Optional[np.ndarray] = None,
        p0: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """批量运行滤波器，输出完整时序结果。"""

        measurements = np.asarray(measurements, dtype=float)
        if measurements.ndim != 2 or measurements.shape[1] != 2:
            raise ValueError("measurements must have shape (N, 2)")
        n = len(measurements)
        if n == 0:
            raise ValueError("measurements cannot be empty")

        if x0 is None:
            x0 = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0], dtype=float)

        self.reset(x0=x0, p0=p0)

        states = np.zeros((n, self.dim_x), dtype=float)
        covariances = np.zeros((n, self.dim_x, self.dim_x), dtype=float)
        innovations = np.zeros((n, self.dim_z), dtype=float)
        innovation_covariances = np.zeros((n, self.dim_z, self.dim_z), dtype=float)
        nis = np.zeros(n, dtype=float)
        predicted_measurements = np.zeros((n, self.dim_z), dtype=float)

        for k, z in enumerate(measurements):
            # 第一帧直接用测量初始化后更新，后续帧走预测+更新。
            if k > 0:
                self.predict()
            result = self.update(z)
            states[k] = result["state"]
            covariances[k] = result["covariance"]
            innovations[k] = result["innovation"]
            innovation_covariances[k] = result["innovation_covariance"]
            nis[k] = result["nis"]
            predicted_measurements[k] = result["predicted_measurement"]

        return {
            "states": states,
            "positions": states[:, :2],
            "covariances": covariances,
            "innovations": innovations,
            "innovation_covariances": innovation_covariances,
            "nis": nis,
            "predicted_measurements": predicted_measurements,
        }
