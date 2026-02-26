"""通用数学/数组工具函数（中文注释版）。"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


def ensure_rng(seed_or_rng: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    """统一随机数接口：传 seed 或已存在的 Generator 都可。"""

    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


def finite_difference(samples: np.ndarray, dt: float) -> np.ndarray:
    """对时间序列做数值微分（按行求导），常用于位置->速度近似。"""

    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]
    if len(samples) < 2:
        return np.zeros_like(samples)
    edge_order = 2 if len(samples) > 2 else 1
    return np.gradient(samples, dt, axis=0, edge_order=edge_order)


def row_norm(values: np.ndarray) -> np.ndarray:
    """计算二维/多维向量序列每一行的欧氏范数。"""

    values = np.asarray(values, dtype=float)
    return np.linalg.norm(values, axis=1)


def compute_rmse(values: np.ndarray) -> float:
    """计算序列的均方根（常用于误差统计）。"""

    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def unit_vectors(values: np.ndarray, fallback: tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    """将向量序列归一化为单位向量；零向量用 fallback 替代。"""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("values must have shape (N, 2)")
    norms = np.linalg.norm(values, axis=1)
    out = np.zeros_like(values)
    mask = norms > 1e-12
    out[mask] = values[mask] / norms[mask, None]
    if np.any(~mask):
        out[~mask] = np.asarray(fallback, dtype=float)
    return out


def perpendicular_vectors(values: np.ndarray) -> np.ndarray:
    """对二维向量 [x, y] 生成垂直向量 [-y, x]。"""

    values = np.asarray(values, dtype=float)
    return np.column_stack((-values[:, 1], values[:, 0]))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """简易滑动平均，用于平滑随机扰动。"""

    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    window = min(window, len(values))
    kernel = np.ones(window, dtype=float) / window
    if values.ndim == 1:
        return np.convolve(values, kernel, mode="same")
    out = np.empty_like(values)
    for i in range(values.shape[1]):
        out[:, i] = np.convolve(values[:, i], kernel, mode="same")
    return out
