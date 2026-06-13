"""Time-domain response of a linear state-space model.

The companion to ``frequency_response.py``: instead of evaluating the model on
the imaginary axis, here we solve it **in time**. Three classical analyses,
all on the project's own numerics (``expm``, ``eigvals`` — no SciPy in ``src/``):

- **Free (initial-condition) response** — the autonomous solution
  ``x(t) = e^{A t} x(0)``. For a *regulator* (closed loop ``A − BK``) this is the
  regulation transient: how the controlled state returns to the equilibrium
  after a disturbance. It is the time-domain test that matters for this system,
  whose open loop is unstable.
- **Step response** — the response ``y(t)`` to a unit step input from rest,
  ``x_{k+1} = A_d x_k + B_d u``, the standard transfer-function step response.
- **Modal analysis** — the eigenvalues of ``A`` recast as natural frequencies
  and damping ratios, the time-domain reading of the same poles the Bode
  diagram shows.

The derivations and the transient-metric definitions are in
``docs/guides/time_domain_response.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import StateSpaceModel
from ..numerics.linalg import eigvals
from ..numerics.matrix_exp import expm


def closed_loop(model: StateSpaceModel, gain) -> StateSpaceModel:
    """The closed-loop model under state feedback ``u = −K x``.

    Substituting ``u = −K x`` into ``ẋ = A x + B u`` gives the autonomous
    ``ẋ = (A − BK) x``; the output map ``(C, D)`` is carried over. ``gain`` is a
    raw ``n_u × n_x`` array (this stays dependency-clean — ``dynamics`` does not
    import ``control``). Continuous or discrete is preserved.
    """
    if not isinstance(model, StateSpaceModel):
        raise TypeError("model must be a StateSpaceModel")
    K = np.asarray(gain, dtype=np.float64)
    if K.shape != (model.n_u, model.n_x):
        raise ValueError(
            f"gain must be (n_u, n_x)=({model.n_u}, {model.n_x}), got {K.shape}"
        )
    A_cl = model.A - model.B @ K
    dt = model.dt if model.is_discrete else None
    return StateSpaceModel(A_cl, model.B, model.C, model.D,
                           is_discrete=model.is_discrete, dt=dt)


def free_response(model: StateSpaceModel, x0, t_end: float, dt: float | None = None):
    """Initial-condition response ``x(t) = e^{A t} x₀``, sampled on a uniform grid.

    The autonomous solution of ``ẋ = A x`` (continuous) propagated exactly with
    the matrix exponential: ``Φ = e^{A·dt}`` once, then ``x_{k+1} = Φ x_k`` — no
    integrator truncation. For a discrete model the one-step map is ``A`` itself
    and ``dt`` defaults to the model's.

    Parameters
    ----------
    model : StateSpaceModel
    x0 : (n_x,) array_like
        Initial state.
    t_end : float
        Final time (s), > 0.
    dt : float, optional
        Sample step; required for a continuous model, defaults to ``model.dt``
        for a discrete one.

    Returns
    -------
    (time, states, outputs) :
        ``time`` (T,), ``states`` (T, n_x), ``outputs`` (T, n_y) with
        ``y = C x`` (the autonomous output, input held at zero).
    """
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (model.n_x,):
        raise ValueError(f"x0 must have shape ({model.n_x},), got {x0.shape}")
    dt = _resolve_dt(model, dt)
    if t_end <= 0.0:
        raise ValueError("t_end must be > 0")
    n_steps = int(round(t_end / dt)) + 1
    transition = model.A if model.is_discrete else expm(model.A * dt)

    states = np.empty((n_steps, model.n_x), dtype=np.float64)
    states[0] = x0
    for k in range(1, n_steps):
        states[k] = transition @ states[k - 1]
    time = dt * np.arange(n_steps)
    outputs = states @ model.C.T
    return time, states, outputs


def step_response(model: StateSpaceModel, t_end: float, dt: float | None = None,
                  input_index: int = 0):
    """Unit-step response of ``model`` from rest (zero-order hold).

    Discretises a continuous model (ZOH, via ``StateSpaceModel.discretize``),
    then iterates ``x_{k+1} = A_d x_k + B_d u`` with ``u`` the unit step on
    ``input_index``, ``x₀ = 0``. Returns the standard step response.

    Returns
    -------
    (time, outputs) :
        ``time`` (T,), ``outputs`` (T, n_y) — the response ``y = C x + D u`` to
        the step.
    """
    if not 0 <= input_index < model.n_u:
        raise ValueError(f"input_index must be in [0, {model.n_u}), got {input_index}")
    dt = _resolve_dt(model, dt)
    if t_end <= 0.0:
        raise ValueError("t_end must be > 0")
    discrete = model if model.is_discrete else model.discretize(dt)
    A_d, B_d, C, D = discrete.A, discrete.B, discrete.C, discrete.D

    u = np.zeros(model.n_u)
    u[input_index] = 1.0
    Bd_u = B_d @ u
    Du = D @ u

    n_steps = int(round(t_end / dt)) + 1
    outputs = np.empty((n_steps, model.n_y), dtype=np.float64)
    x = np.zeros(model.n_x)
    for k in range(n_steps):
        outputs[k] = C @ x + Du
        x = A_d @ x + Bd_u
    time = dt * np.arange(n_steps)
    return time, outputs


@dataclass(frozen=True)
class TransientMetrics:
    """Standard step-response characteristics of a scalar time signal.

    Defined for a response moving from ``initial`` to a steady-state ``final``
    (Ogata, *Modern Control Engineering* §5-4). For a regulation transient pass
    ``final = 0``; for a unit step from rest, ``initial = 0``.
    """

    rise_time: float          # 10 % → 90 % of the step, in seconds
    peak_time: float          # time of the largest excursion past `final`
    percent_overshoot: float  # 100 · (peak excursion past `final`) / |step|
    settling_time: float      # last exit from the ±settle_tol band around `final`
    steady_state: float       # the `final` value used


def transient_metrics(time, signal, *, final: float | None = None,
                      initial: float | None = None,
                      settle_tol: float = 0.02) -> TransientMetrics:
    """Compute step-response characteristics of ``signal(time)``.

    ``final`` defaults to the last sample, ``initial`` to the first. The
    settling band is ``±settle_tol · |final − initial|`` around ``final``
    (the 2 % criterion by default). A flat signal (zero step) yields zeros.
    """
    time = np.asarray(time, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    if time.shape != signal.shape or time.ndim != 1:
        raise ValueError("time and signal must be 1-D arrays of equal length")
    final = float(signal[-1]) if final is None else float(final)
    initial = float(signal[0]) if initial is None else float(initial)
    step = final - initial
    if abs(step) <= 0.0:
        return TransientMetrics(0.0, 0.0, 0.0, 0.0, final)

    sign = np.sign(step)
    # rise time: 10 % -> 90 % of the way from initial to final
    low = initial + 0.1 * step
    high = initial + 0.9 * step
    t_low = _first_crossing(time, signal, low, sign)
    t_high = _first_crossing(time, signal, high, sign)
    rise_time = max(0.0, t_high - t_low)

    # overshoot: the largest excursion of the signal *past* `final`
    excursion = sign * (signal - final)          # > 0 when overshooting
    peak_index = int(np.argmax(excursion))
    peak_excursion = max(0.0, float(excursion[peak_index]))
    percent_overshoot = 100.0 * peak_excursion / abs(step)
    peak_time = float(time[peak_index]) if peak_excursion > 0.0 else float(time[-1])

    # settling time: last time the signal is outside the ±tol band around final
    band = settle_tol * abs(step)
    outside = np.abs(signal - final) > band
    if not bool(outside.any()):
        settling_time = float(time[0])
    else:
        last = int(np.flatnonzero(outside)[-1])
        settling_time = float(time[last + 1]) if last + 1 < time.size else float(time[-1])

    return TransientMetrics(rise_time, peak_time, percent_overshoot,
                            settling_time, final)


@dataclass(frozen=True)
class Mode:
    """A single vibrational mode of a continuous model.

    ``eigenvalue`` is the pole ``λ``; ``natural_frequency`` is ``ω_n = |λ|``
    (rad/s); ``damping_ratio`` is ``ζ = −Re(λ)/|λ|`` (so ``ζ < 0`` flags an
    unstable mode, ``ζ ≥ 1`` an overdamped real one).
    """

    eigenvalue: complex
    natural_frequency: float
    damping_ratio: float


def modal_analysis(model: StateSpaceModel) -> list[Mode]:
    """Natural frequencies and damping ratios of the model's poles.

    The time-domain reading of the eigenvalues of ``A`` (continuous). For a
    discrete model the poles are first mapped to continuous-equivalents via
    ``λ_c = ln(λ_d) / Δt``. A pole at the origin (an integrator) has
    ``ω_n = 0`` and an undefined damping, reported as ``ζ = 0``.
    """
    poles = eigvals(model.A)
    if model.is_discrete:
        poles = np.log(poles.astype(np.complex128)) / model.dt
    modes = []
    for lam in poles:
        omega_n = float(abs(lam))
        zeta = 0.0 if omega_n == 0.0 else float(-np.real(lam) / omega_n)
        modes.append(Mode(eigenvalue=complex(lam), natural_frequency=omega_n,
                          damping_ratio=zeta))
    return modes


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resolve_dt(model: StateSpaceModel, dt: float | None) -> float:
    if model.is_discrete:
        return model.dt if dt is None else float(dt)
    if dt is None:
        raise ValueError("dt is required for a continuous model")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    return float(dt)


def _first_crossing(time, signal, level, sign) -> float:
    """First time ``sign·signal`` reaches ``sign·level`` (else the final time)."""
    reached = np.flatnonzero(sign * signal >= sign * level)
    return float(time[reached[0]]) if reached.size else float(time[-1])
