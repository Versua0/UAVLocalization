"""Visualization for range-localization attack experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _plt():
    """Lazy import matplotlib."""

    import matplotlib.pyplot as plt

    return plt


def plot_range_localization_results(result: Dict[str, Any]) -> Tuple[object, np.ndarray]:
    """Plot geometry, range series, attack bias, and residuals."""

    plt = _plt()

    t = result["trajectory"]["t"]
    uav_positions = result["uav_positions"]
    true_xy = result["target_true_xy"]
    fake_xy = result["target_fake_xy"]
    ranges_true = result["ranges_true"]
    ranges_clean = result["ranges_clean"]
    attack = result["attack"]
    estimates = result["estimates"]
    cfg = result["config"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Geometry panel: UAV trajectory and localization outputs.
    ax = axes[0, 0]
    ax.plot(uav_positions[:, 0], uav_positions[:, 1], color="tab:blue", lw=1.8, label="UAV trajectory")
    ax.scatter(uav_positions[0, 0], uav_positions[0, 1], color="tab:blue", s=40, marker="o", label="UAV start")
    ax.scatter(uav_positions[-1, 0], uav_positions[-1, 1], color="tab:blue", s=40, marker="s", label="UAV end")
    ax.scatter(true_xy[0], true_xy[1], color="tab:green", s=90, marker="*", label="True target")
    ax.scatter(fake_xy[0], fake_xy[1], color="tab:red", s=90, marker="X", label="Fake target")

    ax.scatter(
        estimates["clean_l2"]["position"][0],
        estimates["clean_l2"]["position"][1],
        color="tab:green",
        s=55,
        marker="o",
        facecolors="none",
        label="Clean L2",
    )
    ax.scatter(
        estimates["attacked_l2"]["position"][0],
        estimates["attacked_l2"]["position"][1],
        color="tab:red",
        s=55,
        marker="o",
        facecolors="none",
        label="Attacked L2",
    )
    ax.scatter(
        estimates["attacked_l1"]["position"][0],
        estimates["attacked_l1"]["position"][1],
        color="tab:orange",
        s=55,
        marker="D",
        facecolors="none",
        label="Attacked L1",
    )

    ax.set_title("Geometry and Estimated Positions")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # Range sequence panel.
    ax = axes[0, 1]
    ax.plot(t, ranges_true, color="black", lw=1.5, label="True range")
    ax.plot(t, ranges_clean, color="tab:blue", alpha=0.7, label="Noisy range")
    ax.plot(t, attack["ranges_attacked"], color="tab:red", alpha=0.8, label="Attacked range")
    ax.axvline(float(cfg.attack.start_time), color="k", ls=":", lw=1.0, label="Attack start")
    ax.set_title("Range Measurements")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("range [m]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # Bias and equivalent delay panel.
    ax = axes[1, 0]
    ax.plot(t, attack["desired_bias"], color="tab:gray", ls="--", lw=1.4, label="Desired bias")
    ax.plot(t, attack["bias"], color="tab:red", lw=1.8, label="Applied bias")
    ax.axvline(float(cfg.attack.start_time), color="k", ls=":", lw=1.0)
    ax.set_title("Time-Varying Range Inflation Bias")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("bias [m]")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(t, attack["extra_delay_s"] * 1e9, color="tab:purple", alpha=0.5, lw=1.2, label="Equivalent delay")
    ax2.set_ylabel("extra delay [ns]")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    # Residuals panel for solver comparison.
    ax = axes[1, 1]
    idx = np.arange(len(t))
    ax.plot(idx, np.abs(estimates["clean_l2"]["residuals"]), color="tab:green", lw=1.3, label="|res| clean L2")
    ax.plot(idx, np.abs(estimates["attacked_l2"]["residuals"]), color="tab:red", lw=1.3, label="|res| attacked L2")
    ax.plot(idx, np.abs(estimates["attacked_l1"]["residuals"]), color="tab:orange", lw=1.3, label="|res| attacked L1")
    ax.set_title("Residual Magnitudes by Sample")
    ax.set_xlabel("sample index")
    ax.set_ylabel("|residual| [m]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # Compact metric annotations.
    m = result["metrics"]
    fig.suptitle(
        (
            "UAV-Assisted Range Localization Under Time-Varying Range Inflation Attack\n"
            f"clean L2 err={m['clean_l2_error']:.2f} m, attacked L2 true/fake={m['attacked_l2_error_to_true']:.2f}/{m['attacked_l2_error_to_fake']:.2f} m, "
            f"success L2/L1={m['attack_success_l2']}/{m['attack_success_l1']}"
        ),
        fontsize=11,
        y=0.98,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, axes


def save_figure(fig: object, path: str) -> None:
    """Save plot to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, bbox_inches="tight")
