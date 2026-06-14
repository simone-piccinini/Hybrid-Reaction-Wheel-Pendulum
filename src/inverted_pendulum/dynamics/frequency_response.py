"""Frequency response of a linear state-space model — the Bode pipeline.

Takes the **state-space** form of a model and moves to the **frequency** form by
the Laplace transform, then evaluates it on the imaginary axis to get the data a
Bode diagram plots. The full derivation is in
``docs/guides/frequency_analysis.md``; the essentials:

A continuous LTI system ``ẋ = A x + B u``, ``y = C x + D u`` has, under the
Laplace transform with zero initial conditions, the transfer matrix

    G(s) = C (s I − A)⁻¹ B + D                         (Laplace → transfer function)

and its **frequency response** is ``G(jω)`` — substitute ``s = jω``. The Bode
diagram plots, per input→output channel,

    magnitude(ω) = 20 · log₁₀ |G(jω)|     (decibels)
    phase(ω)     = ∠ G(jω)                 (degrees)

against ``ω`` on a logarithmic axis.

Computing ``G(jω)`` needs ``(jωI − A)⁻¹ B``, i.e. the solution of a **complex**
linear system. To stay inside the project's from-scratch real linear algebra
(``AGENTS.md`` §3 — no ``numpy.linalg``/SciPy in ``src/``), the complex system
is solved through the standard real ``2n × 2n`` embedding (see
:func:`_solve_shifted_system`), using ``numerics.linalg.lu_solve_matrix``.

For a discrete model the analogous substitution is ``z = e^{jωΔt}`` and
``G(z) = C (z I − A)⁻¹ B + D``; both cases are handled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import StateSpaceModel
from ..numerics.linalg import lu_solve_matrix


def _solve_shifted_system(A: np.ndarray, B: np.ndarray, s: complex) -> np.ndarray:
    """Solve the complex system ``(s I − A) X = B`` with the real-only LU.

    Writing ``s = σ + jω``, ``X = Xr + j Xi`` and ``B`` (real here) as the
    real/imag split, the complex equation ``(sI − A)(Xr + jXi) = B`` separates
    into the real ``2n × 2n`` block system

        ⎡ σI − A   −ωI   ⎤ ⎡Xr⎤   ⎡B⎤
        ⎢                ⎥ ⎢  ⎥ = ⎢ ⎥
        ⎣  ωI     σI − A ⎤ ⎣Xi⎦   ⎣0⎦

    which the project's real partial-pivot LU solves directly. Returns the
    complex ``X = Xr + j Xi``. Raises ``SingularMatrixError`` (from the LU) if
    ``sI − A`` is singular — e.g. ``s`` hitting an eigenvalue of ``A``, which
    for ``s = jω`` happens at a pole on the imaginary axis (an integrator gives
    one at ``ω = 0``).
    """
    n = A.shape[0]
    sigma, omega = float(np.real(s)), float(np.imag(s))
    shift = sigma * np.eye(n) - A
    top = np.hstack([shift, -omega * np.eye(n)])
    bottom = np.hstack([omega * np.eye(n), shift])
    block = np.vstack([top, bottom])                      # (2n, 2n) real
    rhs = np.vstack([B, np.zeros_like(B)])                # (2n, n_u) real
    solution = lu_solve_matrix(block, rhs)
    return solution[:n] + 1j * solution[n:]


def transfer_function(model: StateSpaceModel, omega) -> np.ndarray:
    """Frequency response ``G(jω)`` (continuous) or ``G(e^{jωΔt})`` (discrete).

    Implements ``G = C (sI − A)⁻¹ B + D`` channel-by-channel over the requested
    frequencies, with ``s = jω`` for a continuous model and ``s = e^{jωΔt}``
    (the unit-circle point) for a discrete one.

    Parameters
    ----------
    model : StateSpaceModel
        The linear model whose response is wanted.
    omega : float or array_like
        Angular frequencies ω (rad/s), each ≥ 0.

    Returns
    -------
    ndarray of complex128, shape ``(n_omega, n_y, n_u)``
        ``G`` at each frequency. ``n_omega`` follows the input length.

    Raises
    ------
    SingularMatrixError
        If ``sI − A`` is singular at some requested ω (a pole on the contour).
    """
    if not isinstance(model, StateSpaceModel):
        raise TypeError("model must be a StateSpaceModel")
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if np.any(omega < 0.0):
        raise ValueError("angular frequencies must be non-negative")
    A, B, C, D = model.A, model.B, model.C, model.D
    response = np.empty((omega.size, model.n_y, model.n_u), dtype=np.complex128)
    for k, w in enumerate(omega):
        s = np.exp(1j * w * model.dt) if model.is_discrete else 1j * w
        X = _solve_shifted_system(A, B, s)
        response[k] = C @ X + D
    return response


@dataclass(frozen=True)
class BodeData:
    """A single input→output Bode curve (the data a Bode plot draws).

    Fields
    ------
    omega : ndarray
        Angular frequencies ω (rad/s).
    magnitude_db : ndarray
        ``20 log₁₀ |G(jω)|`` (decibels).
    phase_deg : ndarray
        ``∠G(jω)`` in degrees, unwrapped for a continuous curve.
    output_index, input_index : int
        Which channel of ``G`` this curve is.
    """

    omega: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    output_index: int
    input_index: int


def bode(
    model: StateSpaceModel,
    omega,
    *,
    output_index: int = 0,
    input_index: int = 0,
) -> BodeData:
    """Bode data for one input→output channel of ``model``.

    Computes the frequency response and reduces the chosen channel to
    magnitude (dB) and unwrapped phase (degrees) — the two traces of a Bode
    diagram. Plotting lives in ``io.plotting.plot_bode`` (the ``io`` layer owns
    visualisation; this layer only produces the numbers).
    """
    response = transfer_function(model, omega)
    n_y, n_u = response.shape[1], response.shape[2]
    if not (0 <= output_index < n_y):
        raise ValueError(f"output_index must be in [0, {n_y}), got {output_index}")
    if not (0 <= input_index < n_u):
        raise ValueError(f"input_index must be in [0, {n_u}), got {input_index}")
    channel = response[:, output_index, input_index]
    magnitude_db = 20.0 * np.log10(np.abs(channel))
    phase_deg = np.degrees(np.unwrap(np.angle(channel)))
    return BodeData(
        omega=np.atleast_1d(np.asarray(omega, dtype=np.float64)),
        magnitude_db=magnitude_db,
        phase_deg=phase_deg,
        output_index=output_index,
        input_index=input_index,
    )


def log_frequencies(omega_min: float, omega_max: float, n_points: int = 400) -> np.ndarray:
    """A log-spaced frequency grid ``[ω_min, ω_max]`` (rad/s) for Bode sweeps.

    Logarithmic spacing matches the logarithmic frequency axis of a Bode plot.
    """
    if omega_min <= 0.0 or omega_max <= omega_min:
        raise ValueError("require 0 < omega_min < omega_max")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    return np.logspace(np.log10(omega_min), np.log10(omega_max), n_points)
