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

import warnings
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


def loop_transfer_function(
    plant: StateSpaceModel, gain_K, gain_L, omega
) -> np.ndarray:
    """Open-loop gain ``L = −K_c(z)·G(z)`` of the LQG loop, broken at the input.

    For stability-margin analysis an expert looks not at the closed-loop
    response but at the **loop transfer function** of the broken loop — the
    return ratio seen at the plant input. With the LQG controller (Kalman
    observer + LQR feedback) realised, in current-estimator predictor form, as

        A_c = (A_d − B_d K)(I − L C),   B_c = (A_d − B_d K) L,
        C_c = −K (I − L C),            D_c = −K L,

    its transfer function ``K_c(z)`` (measurement → control) composes with the
    plant ``G(z) = C(zI − A_d)⁻¹B_d`` into the scalar return ratio at the input

        L(z) = −K_c(z) · G(z)

    (the leading minus puts it in the standard negative-feedback / critical
    point −1 convention, since the controller already carries the ``−K``). This
    is a discrete-time loop (``z = e^{jωΔt}``); the controller realised here is
    exactly the one the simulator runs (predict → update → ``u = −K x̂``).

    Parameters
    ----------
    plant : StateSpaceModel
        The **discrete** plant ``(A_d, B_d, C)`` the controller was designed for
        (single input). Must be controllable and observable for the LQG loop to
        be well-defined — drop any decoupled/undetectable mode first (e.g. the
        wheel angle), see ``scripts/stability_margins.py``.
    gain_K : (n_u, n_x) array_like
        The LQR feedback gain ``K``.
    gain_L : (n_x, n_y) array_like
        The steady-state Kalman gain ``L``.
    omega : float or array_like
        Angular frequencies ω (rad/s); evaluated on the unit circle.

    Returns
    -------
    (n_omega,) ndarray of complex128
        The scalar loop gain ``L(jω)``.
    """
    if not isinstance(plant, StateSpaceModel):
        raise TypeError("plant must be a StateSpaceModel")
    if not plant.is_discrete:
        raise ValueError("loop_transfer_function expects a discrete plant model")
    if plant.n_u != 1:
        raise ValueError("the loop is broken at a single (scalar) input")
    K = np.asarray(gain_K, dtype=np.float64)
    L = np.asarray(gain_L, dtype=np.float64)
    n_x, n_u, n_y = plant.n_x, plant.n_u, plant.n_y
    if K.shape != (n_u, n_x):
        raise ValueError(f"gain_K must be (n_u, n_x)=({n_u}, {n_x}), got {K.shape}")
    if L.shape != (n_x, n_y):
        raise ValueError(f"gain_L must be (n_x, n_y)=({n_x}, {n_y}), got {L.shape}")

    A_d, B_d, C = plant.A, plant.B, plant.C
    closed = A_d - B_d @ K
    filtered = np.eye(n_x) - L @ C
    controller = StateSpaceModel(
        closed @ filtered,        # A_c
        closed @ L,               # B_c   (input: measurement, n_y)
        -K @ filtered,            # C_c   (output: control, n_u)
        -K @ L,                   # D_c
        is_discrete=True, dt=plant.dt,
    )
    K_c = transfer_function(controller, omega)   # (n_omega, n_u, n_y)
    G = transfer_function(plant, omega)          # (n_omega, n_y, n_u)
    loop = -np.einsum("kij,kjl->kil", K_c, G)    # (n_omega, n_u, n_u) = (.,1,1)
    return loop[:, 0, 0]


@dataclass(frozen=True)
class StabilityMargins:
    """Classical stability margins of a loop gain (``scripts/stability_margins.py``).

    Fields
    ------
    gain_margin_db : float
        How much the loop gain may grow before instability — the attenuation
        ``−20log₁₀|L|`` at the phase crossover (``inf`` if the phase never
        reaches −180°). An expert wants ≳ 6 dB.
    phase_margin_deg : float
        How much extra phase lag (e.g. computation/actuation delay) the loop can
        absorb before instability — ``180° + ∠L`` at the gain crossover (``inf``
        if the gain never crosses 0 dB). An expert wants ≳ 30–45°.
    gain_crossover : float
        Frequency where ``|L| = 1`` (0 dB), rad/s (``nan`` if none).
    phase_crossover : float
        Frequency where ``∠L = −180°``, rad/s (``nan`` if none).
    """

    gain_margin_db: float
    phase_margin_deg: float
    gain_crossover: float
    phase_crossover: float


def _log_interp_frequency(omega, values, level, i) -> float:
    """Frequency where ``values`` reaches ``level`` between indices ``i, i+1`` (log-ω)."""
    frac = (level - values[i]) / (values[i + 1] - values[i])
    return float(omega[i] * (omega[i + 1] / omega[i]) ** frac)


def stability_margins(omega, loop) -> StabilityMargins:
    """Gain and phase margins of a scalar loop gain ``L(jω)`` (``omega`` ascending).

    The gain margin is read at the first phase crossover (∠L = −180° − 360k),
    the phase margin at the first gain crossover (|L| = 1), both by interpolation
    on the supplied grid. A finer ``omega`` grid gives sharper crossovers.
    """
    omega = np.asarray(omega, dtype=np.float64)
    loop = np.asarray(loop, dtype=np.complex128)
    magnitude = np.abs(loop)
    phase = np.degrees(np.unwrap(np.angle(loop)))

    # phase margin at the first gain crossover (|L| = 1)
    phase_margin, gain_crossover = float("inf"), float("nan")
    gain_cross = np.where(np.diff(np.sign(magnitude - 1.0)))[0]
    if gain_cross.size:
        i = int(gain_cross[0])
        gain_crossover = _log_interp_frequency(omega, magnitude, 1.0, i)
        frac = (1.0 - magnitude[i]) / (magnitude[i + 1] - magnitude[i])
        phase_at_gc = phase[i] + frac * (phase[i + 1] - phase[i])
        phase_margin = ((180.0 + phase_at_gc) + 180.0) % 360.0 - 180.0

    # gain margin at the first phase crossover (∠L = −180° − 360k)
    gain_margin_db, phase_crossover = float("inf"), float("nan")
    candidates = []
    for k in range(6):
        level = -180.0 - 360.0 * k
        for i in np.where(np.diff(np.sign(phase - level)))[0]:
            candidates.append((omega[int(i)], int(i), level))
    if candidates:
        _, i, level = min(candidates, key=lambda t: t[0])
        phase_crossover = _log_interp_frequency(omega, phase, level, i)
        frac = (level - phase[i]) / (phase[i + 1] - phase[i])
        magnitude_at_pc = magnitude[i] + frac * (magnitude[i + 1] - magnitude[i])
        gain_margin_db = float(-20.0 * np.log10(abs(magnitude_at_pc)))
    return StabilityMargins(gain_margin_db, phase_margin, gain_crossover, phase_crossover)


def log_frequencies(omega_min: float, omega_max: float, n_points: int = 400) -> np.ndarray:
    """A log-spaced frequency grid ``[ω_min, ω_max]`` (rad/s) for Bode sweeps.

    Logarithmic spacing matches the logarithmic frequency axis of a Bode plot.
    """
    if omega_min <= 0.0 or omega_max <= omega_min:
        raise ValueError("require 0 < omega_min < omega_max")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    return np.logspace(np.log10(omega_min), np.log10(omega_max), n_points)


def loop_margins(
    plant: StateSpaceModel,
    gain_K,
    gain_L,
    *,
    omega_min: float = 1e-2,
    n_points: int = 2000,
) -> StabilityMargins:
    """Gain/phase margins of the LQG loop ``L = −K_c·G`` for a discrete plant.

    A one-call convenience over :func:`log_frequencies`,
    :func:`loop_transfer_function`, and :func:`stability_margins` — the exact
    sequence ``scripts/stability_margins.py`` runs, factored out so both the
    script and the simulator use one implementation (the deduplication flagged in
    ``docs/papers/robustness_lqg_measured.md`` §5). The grid runs to just below the
    Nyquist frequency ``0.9·π/Δt``.

    The ``plant`` must be the **discrete, controllable + observable** model on
    which the steady-state LQG is defined — i.e. with any decoupled/undetectable
    mode already dropped (the wheel angle, ``KEEP = [0,1,3]``; see the module
    docstring of :func:`loop_transfer_function`). ``gain_K`` is the reduced LQR
    gain and ``gain_L`` the reduced steady-state Kalman gain.

    Returns
    -------
    StabilityMargins
        Gain/phase margins and their crossover frequencies. A loop that never
        crosses 0 dB / −180° yields an infinite phase / gain margin (the
        ``StabilityMargins`` convention), which downstream reads as "robust".
    """
    if not isinstance(plant, StateSpaceModel):
        raise TypeError("plant must be a StateSpaceModel")
    if not plant.is_discrete:
        raise ValueError("loop_margins expects a discrete plant model")
    omega = log_frequencies(omega_min, 0.9 * np.pi / plant.dt, n_points)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # log10(0)/divide at crossovers are handled
        loop = loop_transfer_function(plant, gain_K, gain_L, omega)
    return stability_margins(omega, loop)
