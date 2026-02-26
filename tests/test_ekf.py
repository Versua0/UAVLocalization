"""EKF 模块单元测试（中文注释版）。"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 允许直接运行测试文件时找到项目根目录下的包。
    sys.path.insert(0, str(ROOT))

from config.parameters import EKFConfig
from src.ekf_filter import ExtendedKalmanFilter2D


def _build_linear_measurements(n: int = 120, dt: float = 0.1, noise_std: float = 0.8, seed: int = 0):
    """构造线性匀速运动的带噪测量，用于测试 EKF。"""

    t = np.arange(n, dtype=float) * dt
    true_position = np.column_stack((1.0 + 2.0 * t, -3.0 + 0.5 * t))
    rng = np.random.default_rng(seed)
    measurements = true_position + rng.normal(0.0, noise_std, size=true_position.shape)
    return t, true_position, measurements


def test_ekf_run_output_shapes_and_nis_nonnegative() -> None:
    """验证 EKF 输出张量维度正确，NIS 非负。"""

    dt = 0.1
    _, _, measurements = _build_linear_measurements(dt=dt)
    cfg = EKFConfig(measurement_noise_std=0.8, process_noise_vel=0.4)
    ekf = ExtendedKalmanFilter2D(cfg, dt=dt)

    out = ekf.run(measurements)

    n = len(measurements)
    assert out["states"].shape == (n, 4)
    assert out["positions"].shape == (n, 2)
    assert out["innovations"].shape == (n, 2)
    assert out["innovation_covariances"].shape == (n, 2, 2)
    assert out["nis"].shape == (n,)
    assert np.all(out["nis"] >= -1e-9)


def test_ekf_improves_rmse_over_raw_measurements() -> None:
    """验证在平稳线性场景下，EKF 对测量具有一定降噪效果。"""

    dt = 0.1
    _, true_position, measurements = _build_linear_measurements(dt=dt, noise_std=1.0, seed=123)
    cfg = EKFConfig(measurement_noise_std=1.0, process_noise_vel=0.3, process_noise_pos=0.01)
    ekf = ExtendedKalmanFilter2D(cfg, dt=dt)

    out = ekf.run(measurements)
    raw_rmse = float(np.sqrt(np.mean(np.sum((measurements - true_position) ** 2, axis=1))))
    ekf_rmse = float(np.sqrt(np.mean(np.sum((out["positions"] - true_position) ** 2, axis=1))))

    assert ekf_rmse < raw_rmse
