"""Plotting — visual diagnostics of a run (``io`` cross-cutting leaf).

Plotting lives in ``io`` by design (``dependency_rules.md`` §3 keeps metric
computation free of visualisation side effects). ``matplotlib`` is a dev/
analysis dependency, not a "problem-solving" library, so it is allowed here;
it is imported **lazily** inside each function so that importing the package
never requires matplotlib to be installed.

Each function returns the matplotlib ``Figure`` so the caller can show, tweak,
or save it (``save_figure`` is provided for convenience).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.types import SimulationResult

# State-vector component labels (notation.md §3 ordering).
_STATE_LABELS = (r"$\theta_p$ (rad)", r"$\dot\theta_p$ (rad/s)",
                 r"$\theta_w$ (rad)", r"$\dot\theta_w$ (rad/s)")


def plot_trajectory(result: SimulationResult, *, title: str | None = None):
    """Plot the true vs estimated state trajectories and the control history.

    One row per state component (true solid, Kalman estimate dashed) plus a
    final row for the applied control. Returns the ``Figure``.
    """
    import matplotlib.pyplot as plt

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    n_x = result.n_x
    fig, axes = plt.subplots(n_x + 1, 1, figsize=(9, 2.0 * (n_x + 1)), sharex=True)
    time = result.time
    for i in range(n_x):
        axes[i].plot(time, result.true_states[:, i], label="true", linewidth=1.5)
        axes[i].plot(time, result.estimated_states[:, i], "--",
                     label="estimate", linewidth=1.0)
        label = _STATE_LABELS[i] if i < len(_STATE_LABELS) else f"x[{i}]"
        axes[i].set_ylabel(label)
        axes[i].axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
        axes[i].legend(loc="upper right", fontsize=8)
    for j in range(result.n_u):
        axes[-1].plot(time, result.controls[:, j], color="tab:red",
                      label=f"u[{j}] (V)")
    axes[-1].set_ylabel("control (V)")
    axes[-1].set_xlabel("time (s)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(title or ("DIVERGED rollout" if result.diverged
                           else "Closed-loop rollout"))
    fig.tight_layout()
    return fig


def plot_convergence(y_history, *, title: str = "Bayesian-optimisation progress"):
    """Plot observed cost per evaluation and the running best-so-far.

    ``y_history`` is the sequence of observed costs in evaluation order (the
    ``Dataset.y`` of the optimiser). Returns the ``Figure``.
    """
    import matplotlib.pyplot as plt

    y = np.asarray(y_history, dtype=np.float64).ravel()
    if y.size == 0:
        raise ValueError("y_history is empty")
    evaluations = np.arange(1, y.size + 1)
    running_best = np.minimum.accumulate(y)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(evaluations, y, "o", alpha=0.5, label="observed cost")
    ax.plot(evaluations, running_best, "-", color="tab:green",
            label="best so far")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("cost  $y$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(figure, path) -> Path:
    """Save a figure to ``path`` (parent directories created) and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    return path
