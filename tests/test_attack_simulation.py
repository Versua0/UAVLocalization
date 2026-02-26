"""攻击模拟模块单元测试（中文注释版）。"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 允许直接运行测试文件时找到项目根目录下的包。
    sys.path.insert(0, str(ROOT))

from config.parameters import AttackConfig
from src.attack_simulation import TrajectoryAwareAttackSimulator


def test_attack_bias_activates_after_start_time() -> None:
    """验证：攻击只会在 start_time 之后激活，并正确注入偏移。"""

    t = np.arange(0.0, 10.0, 1.0)
    reference = np.column_stack((t, np.zeros_like(t)))
    measurements = np.zeros((len(t), 2), dtype=float)

    cfg = AttackConfig(
        enabled=True,
        start_time=3.0,
        amplitude=2.0,
        frequency_hz=0.5,
        drift_rate=0.0,
        axis_ratio=(1.0, 0.5),
    )
    attacker = TrajectoryAwareAttackSimulator(cfg)
    attacked, bias, mask = attacker.inject_attack(measurements, t, reference_position=reference)

    assert bias.shape == measurements.shape
    assert attacked.shape == measurements.shape
    assert mask.shape == (len(t),)
    assert np.allclose(bias[t < cfg.start_time], 0.0)
    assert np.any(np.linalg.norm(bias[t >= cfg.start_time], axis=1) > 0.0)
    assert np.allclose(attacked, measurements + bias)


def test_attack_disabled_returns_original_measurements() -> None:
    """验证：关闭攻击器时，输出应与输入测量完全一致。"""

    t = np.linspace(0.0, 5.0, 11)
    reference = np.column_stack((t, t))
    measurements = np.column_stack((0.2 * t, -0.1 * t))
    cfg = AttackConfig(enabled=False)

    attacker = TrajectoryAwareAttackSimulator(cfg)
    attacked, bias, mask = attacker.inject_attack(measurements, t, reference_position=reference)

    assert np.allclose(attacked, measurements)
    assert np.allclose(bias, 0.0)
    assert not np.any(mask)
