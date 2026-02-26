"""Solvers for 2D localization from UAV position and range measurements."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from config.range_parameters import LocalizationSolverConfig


def _validate_inputs(uav_positions: np.ndarray, ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uav_positions = np.asarray(uav_positions, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    if uav_positions.ndim != 2 or uav_positions.shape[1] != 2:
        raise ValueError("uav_positions must have shape (N, 2)")
    if ranges.ndim != 1:
        raise ValueError("ranges must be 1D")
    if len(uav_positions) != len(ranges):
        raise ValueError("uav_positions and ranges must have the same length")
    if len(ranges) < 3:
        raise ValueError("At least 3 range measurements are required for 2D localization")
    return uav_positions, ranges


def _linearized_initial_guess(uav_positions: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Linear LS initialization using squared-range difference equations."""

    u0 = uav_positions[0]
    r0 = float(ranges[0])
    A = 2.0 * (uav_positions[1:] - u0[None, :])
    b = (
        np.sum(uav_positions[1:] ** 2, axis=1)
        - np.sum(u0**2)
        - (ranges[1:] ** 2 - r0**2)
    )
    if len(A) < 2:
        return np.mean(uav_positions, axis=0)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        x = np.mean(uav_positions, axis=0)
    if not np.all(np.isfinite(x)):
        x = np.mean(uav_positions, axis=0)
    return np.asarray(x, dtype=float)


def _residual_and_jacobian(uav_positions: np.ndarray, ranges: np.ndarray, position_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diffs = position_xy[None, :] - uav_positions
    dists = np.linalg.norm(diffs, axis=1)
    dists_safe = np.maximum(dists, 1e-9)
    residuals = dists - ranges
    jacobian = diffs / dists_safe[:, None]
    return residuals, jacobian


def _gauss_newton(
    uav_positions: np.ndarray,
    ranges: np.ndarray,
    cfg: LocalizationSolverConfig,
    initial_position: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    p = (
        np.asarray(initial_position, dtype=float).copy()
        if initial_position is not None
        else _linearized_initial_guess(uav_positions, ranges)
    )
    if p.shape != (2,):
        raise ValueError("initial_position must have shape (2,)")

    if weights is None:
        weights = np.ones_like(ranges, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != ranges.shape:
            raise ValueError("weights must have the same shape as ranges")
        weights = np.maximum(weights, 1e-12)

    converged = False
    iterations = 0
    for it in range(int(cfg.max_iterations)):
        residuals, jacobian = _residual_and_jacobian(uav_positions, ranges, p)
        sqrt_w = np.sqrt(weights)
        j_w = jacobian * sqrt_w[:, None]
        r_w = residuals * sqrt_w
        h = j_w.T @ j_w + float(cfg.damping) * np.eye(2)
        g = j_w.T @ r_w
        try:
            step = -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(h, g, rcond=None)[0]

        p = p + step
        iterations = it + 1
        if np.linalg.norm(step) <= float(cfg.tolerance):
            converged = True
            break

    residuals, _ = _residual_and_jacobian(uav_positions, ranges, p)
    return {
        "position": p,
        "residuals": residuals,
        "objective_l2": float(np.sum(residuals**2)),
        "objective_l1": float(np.sum(np.abs(residuals))),
        "converged": converged,
        "iterations": iterations,
    }


def solve_localization_l2(
    uav_positions: np.ndarray,
    ranges: np.ndarray,
    cfg: Optional[LocalizationSolverConfig] = None,
    initial_position: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Solve 2D localization via nonlinear least squares (L2)."""

    uav_positions, ranges = _validate_inputs(uav_positions, ranges)
    config = cfg or LocalizationSolverConfig()
    out = _gauss_newton(uav_positions, ranges, config, initial_position=initial_position, weights=None)
    out["method"] = "l2"
    return out


def solve_localization_l1(
    uav_positions: np.ndarray,
    ranges: np.ndarray,
    cfg: Optional[LocalizationSolverConfig] = None,
    initial_position: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Solve 2D localization via IRLS approximation to L1 minimization."""

    uav_positions, ranges = _validate_inputs(uav_positions, ranges)
    config = cfg or LocalizationSolverConfig()
    p = (
        np.asarray(initial_position, dtype=float).copy()
        if initial_position is not None
        else _linearized_initial_guess(uav_positions, ranges)
    )

    converged = False
    total_inner_iterations = 0
    outer_iterations = 0
    last = None
    weights = np.ones_like(ranges, dtype=float)
    for outer in range(int(config.irls_max_outer_iterations)):
        step_out = _gauss_newton(
            uav_positions,
            ranges,
            config,
            initial_position=p,
            weights=weights,
        )
        p_new = step_out["position"]
        total_inner_iterations += int(step_out["iterations"])
        outer_iterations = outer + 1
        residuals = step_out["residuals"]
        weights = 1.0 / np.maximum(np.abs(residuals), float(config.irls_epsilon))
        last = step_out
        if np.linalg.norm(p_new - p) <= float(config.tolerance):
            p = p_new
            converged = True
            break
        p = p_new

    if last is None:
        last = _gauss_newton(uav_positions, ranges, config, initial_position=p, weights=weights)

    return {
        "method": "l1_irls",
        "position": np.asarray(p, dtype=float),
        "residuals": last["residuals"],
        "objective_l2": float(np.sum(last["residuals"] ** 2)),
        "objective_l1": float(np.sum(np.abs(last["residuals"]))),
        "converged": bool(converged),
        "outer_iterations": outer_iterations,
        "iterations": total_inner_iterations,
    }


def localization_error(position_xy: np.ndarray, true_xy: np.ndarray) -> float:
    """Euclidean localization error."""

    return float(np.linalg.norm(np.asarray(position_xy, dtype=float) - np.asarray(true_xy, dtype=float)))
