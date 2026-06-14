"""Unit tests for io.plotting (matplotlib diagnostics).

matplotlib is forced to the non-interactive Agg backend so the tests need no
display. They check that figures are produced and saved, not pixel content.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from inverted_pendulum.io.plotting import (  # noqa: E402
    plot_convergence,
    plot_trajectory,
    save_figure,
)


def test_plot_trajectory_returns_a_figure(make_result):
    result = make_result([0.05, 0.02, 0.0, -0.01], dt=0.01)
    fig = plot_trajectory(result)
    # one axis per state (4) plus the control row
    assert len(fig.axes) == result.n_x + 1


def test_plot_trajectory_titles_diverged(make_result):
    result = make_result([0.05, 0.0], diverged=True)
    fig = plot_trajectory(result)
    assert "DIVERGED" in fig._suptitle.get_text()


def test_plot_trajectory_rejects_non_result():
    with pytest.raises(TypeError):
        plot_trajectory("not a result")


def test_plot_convergence_returns_a_figure():
    fig = plot_convergence([10.0, 3.0, 5.0, 1.0, 2.0])
    assert len(fig.axes) == 1


def test_plot_convergence_empty_raises():
    with pytest.raises(ValueError):
        plot_convergence([])


def test_save_figure_writes_png(tmp_path):
    fig = plot_convergence([5.0, 4.0, 4.5])
    path = save_figure(fig, tmp_path / "sub" / "conv.png")
    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_bode_returns_two_axes():
    from inverted_pendulum.io.plotting import plot_bode

    omega = np.logspace(-2, 2, 50)
    magnitude_db = -20.0 * np.log10(omega)
    phase_deg = -90.0 * np.ones_like(omega)
    fig = plot_bode(omega, magnitude_db, phase_deg, title="test", label="ch")
    assert len(fig.axes) == 2  # magnitude and phase rows


def test_plot_time_response_multiple_signals():
    from inverted_pendulum.io.plotting import plot_time_response

    time = np.linspace(0, 1, 50)
    signals = np.stack([np.exp(-time), np.cos(time)], axis=1)
    fig = plot_time_response(time, signals, labels=["a", "b"], reference=0.0)
    assert len(fig.axes) == 1


def test_plot_step_response_single():
    from inverted_pendulum.io.plotting import plot_step_response

    time = np.linspace(0, 2, 50)
    fig = plot_step_response(time, 1 - np.exp(-time), label="step")
    assert len(fig.axes) == 1


def test_plot_loop_bode_with_margins():
    from inverted_pendulum.dynamics.frequency_response import stability_margins
    from inverted_pendulum.io.plotting import plot_loop_bode

    w = np.logspace(-2, 2, 2000)
    loop = 2.0 / ((1j * w + 1.0) ** 3)  # finite GM and PM
    margins = stability_margins(w, loop)
    fig = plot_loop_bode(w, loop, margins=margins)
    assert len(fig.axes) == 2  # magnitude and phase rows
    assert "GM" in fig._suptitle.get_text() and "PM" in fig._suptitle.get_text()
