"""Range measurement generation for UAV-assisted localization."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from src.utils import ensure_rng


def compute_true_ranges(uav_positions: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """Compute ideal ranges from UAV positions to a fixed target."""

    uav_positions = np.asarray(uav_positions, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    if uav_positions.ndim != 2 or uav_positions.shape[1] != 2:
        raise ValueError("uav_positions must have shape (N, 2)")
    if target_xy.shape != (2,):
        raise ValueError("target_xy must have shape (2,)")
    return np.linalg.norm(uav_positions - target_xy[None, :], axis=1)


def generate_noisy_ranges(
    uav_positions: np.ndarray,
    target_xy: np.ndarray,
    noise_std: float,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Generate noisy scalar range measurements."""

    true_ranges = compute_true_ranges(uav_positions, target_xy)
    generator = ensure_rng(rng)
    noise = generator.normal(loc=0.0, scale=float(noise_std), size=true_ranges.shape)
    return true_ranges + noise
