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
