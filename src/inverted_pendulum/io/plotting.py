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


def plot_bode(omega, magnitude_db, phase_deg, *, title: str = "Bode diagram",
              label: str | None = None):
    """Plot a Bode diagram: magnitude (dB) and phase (deg) vs log-frequency.

    Takes the plain arrays produced by
    ``dynamics.frequency_response.bode`` (the computation layer owns the
    transfer-function maths; ``io`` only draws). Returns the two-row ``Figure``.
    """
    import matplotlib.pyplot as plt

    omega = np.asarray(omega, dtype=np.float64)
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_mag.semilogx(omega, magnitude_db, label=label)
    ax_mag.set_ylabel("magnitude (dB)")
    ax_mag.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    ax_mag.grid(True, which="both", linewidth=0.3)
    if label is not None:
        ax_mag.legend(loc="best", fontsize=8)
    ax_phase.semilogx(omega, phase_deg)
    ax_phase.set_ylabel("phase (deg)")
    ax_phase.set_xlabel(r"angular frequency $\omega$ (rad/s)")
    ax_phase.grid(True, which="both", linewidth=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_time_response(time, signals, *, labels=None, ylabel: str = "output",
                       title: str = "Time response", reference: float | None = None):
    """Plot one or more time signals on a shared time axis.

    ``signals`` is a 1-D array or a list/2-D array of equal-length signals
    (one per column / entry); ``labels`` names them. ``reference`` draws a
    horizontal target line (e.g. 0 for a regulator). Returns the ``Figure``.
    """
    import matplotlib.pyplot as plt

    time = np.asarray(time, dtype=np.float64)
    signals = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    if signals.shape[0] != time.shape[0]:
        signals = signals.T  # accept (n_signals, T) too
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j in range(signals.shape[1]):
        label = labels[j] if labels is not None and j < len(labels) else None
        ax.plot(time, signals[:, j], label=label, linewidth=1.4)
    if reference is not None:
        ax.axhline(reference, color="0.6", linewidth=0.9, linestyle=":",
                   label="reference")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.3)
    if labels is not None or reference is not None:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_step_response(time, output, *, title: str = "Step response",
                       label: str | None = None):
    """Plot a single step-response curve (thin wrapper over plot_time_response)."""
    return plot_time_response(time, output, labels=None if label is None else [label],
                              ylabel="output", title=title)


def save_figure(figure, path) -> Path:
    """Save a figure to ``path`` (parent directories created) and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    return path
