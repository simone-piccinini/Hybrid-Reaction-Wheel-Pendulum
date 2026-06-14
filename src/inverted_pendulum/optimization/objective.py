"""ObjectiveFunction — the scalar rollout cost the Bayesian optimiser minimises.

The oracle's scoring half (``optimization.md`` §1: the rollout cost is the
"opaque oracle" output): a pure map ``SimulationResult → finite float`` built
on the metrics layer. The cost combines four terms (``notation.md`` §6):

    y = w_e · ITAE
      + w_u · ∫ u(t)² dt
      + w_p · max(0, (M_p − M_p^des) / M_p^max)²
      + w_t · max(0, (T_s − T_s^des) / T_s^max)²
      [ + w_sat · max(0, max|u| − U_max)²    if U_max is set ]

with, term by term:

- **ITAE** ``∫ t·|e(t)| dt`` — the global error index, judging the *whole*
  time history rather than two isolated points of the step response; the time
  weight rewards fast settling and a vanishing steady-state error.
- **Control energy** ``∫ u² dt`` — discourages over-actuation (the system-level
  analogue of the LQR's ``R`` weight, scored on the realised input).
- **Hinge penalties** on overshoot and settling time — **asymmetric**: zero
  while the response meets the spec (it is free to be *better* than target) and
  growing quadratically once it violates it. Normalising by the *maximum
  acceptable* value ``M_p^max`` / ``T_s^max`` keeps the penalties scale-free and
  avoids the division-by-zero of a desired-value denominator. The penalty
  weights ``w_p, w_t`` are meant to be large, so a spec violation makes the cost
  spike and the optimiser discards those hyper-parameters.
- **Saturation penalty** (optional) — a hinge on the peak actuation past a
  hardware limit ``U_max``.

A diverged rollout is mapped to the large **finite** ``PENALTY`` — never
``inf``/``nan``, which would poison the GP surrogate (``data_contracts.md`` §3;
``numerical_standards.md`` §7; ``optimization.md`` §6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.constants import PENALTY
from ..core.types import SimulationResult
from ..metrics.performance_metrics import control_effort, itae
from ..metrics.stability_metrics import overshoot, settling_time


def _hinge(value: float, desired: float, scale: float) -> float:
    """One-sided normalised penalty ``max(0, (value − desired) / scale)²``."""
    return max(0.0, (value - desired) / scale) ** 2


@dataclass(frozen=True)
class ObjectiveFunction:
    """The rollout cost ``y`` (``class_diagram.md``; ``notation.md`` §6).

    Parameters
    ----------
    Mp_desired, Ts_desired : float
        The target overshoot ``M_p^des`` (%) and settling time ``T_s^des`` (s).
        A response meeting these incurs **no** spec penalty.
    Mp_max, Ts_max : float
        The maximum *acceptable* overshoot and settling time, used to normalise
        the hinge penalties (> 0 — they are denominators).
    w_error : float
        Weight ``w_e`` on the ITAE term.
    w_control : float
        Weight ``w_u`` on the control-energy term ``∫ u² dt``.
    w_overshoot, w_settling : float
        Penalty weights ``w_p, w_t`` on the hinge terms (meant to be large).
    U_max : float | None
        Optional actuation limit (V). If set, a hinge penalty on the peak
        ``max|u|`` past ``U_max`` is added with weight ``w_saturation``.
    w_saturation : float
        Penalty weight on the saturation hinge (used only when ``U_max`` is set).
    error_state_index : int
        Which true-state column is the regulated error ``e(t)`` for the ITAE
        (default 0 = ``theta_p``, the pendulum angle).
    penalty : float
        Finite cost assigned to a diverged rollout (default ``PENALTY``).
    """

    Mp_desired: float
    Ts_desired: float
    Mp_max: float = 20.0
    Ts_max: float = 3.0
    w_error: float = 1.0
    w_control: float = 1.0
    w_overshoot: float = 100.0
    w_settling: float = 100.0
    U_max: float | None = None
    w_saturation: float = 100.0
    error_state_index: int = 0
    penalty: float = PENALTY

    def __post_init__(self) -> None:
        for name in ("Mp_desired", "Ts_desired", "Mp_max", "Ts_max", "w_error",
                     "w_control", "w_overshoot", "w_settling", "w_saturation",
                     "penalty"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.U_max is not None:
            object.__setattr__(self, "U_max", float(self.U_max))
        if self.Mp_max <= 0.0 or self.Ts_max <= 0.0:
            raise ValueError("Mp_max and Ts_max must be > 0 (they are denominators)")
        if min(self.w_error, self.w_control, self.w_overshoot, self.w_settling,
               self.w_saturation) < 0.0:
            raise ValueError("all weights must be >= 0")
        if self.U_max is not None and self.U_max <= 0.0:
            raise ValueError("U_max must be > 0 when set")
        if not math.isfinite(self.penalty) or self.penalty <= 0.0:
            raise ValueError("penalty must be a positive finite float (§7)")

    # ------------------------------------------------------------------ #
    # individual metrics (class_diagram.md)
    # ------------------------------------------------------------------ #
    def compute_overshoot(self, result: SimulationResult) -> float:
        """Percentage overshoot ``M_p`` of the rollout (``metrics``)."""
        return overshoot(result, state_index=self.error_state_index)

    def compute_settling_time(self, result: SimulationResult) -> float:
        """Settling time ``T_s`` of the rollout (``metrics``)."""
        return settling_time(result, state_index=self.error_state_index)

    def compute_control_effort(self, result: SimulationResult) -> float:
        """Control energy ``∫‖u‖² dt`` (``metrics``)."""
        return control_effort(result)

    def compute_itae(self, result: SimulationResult) -> float:
        """ITAE ``∫ t·|e(t)| dt`` of the regulated error (``metrics``)."""
        return itae(result, state_index=self.error_state_index)

    # ------------------------------------------------------------------ #
    # the scalar cost
    # ------------------------------------------------------------------ #
    def evaluate(self, result: SimulationResult) -> float:
        """The scalar cost ``y`` of one rollout (``notation.md`` §6).

        A diverged run returns ``penalty`` directly (``data_contracts.md`` §3).
        Otherwise the four (or five) terms above are summed; every term is
        finite by construction, so ``y`` is always finite.
        """
        if not isinstance(result, SimulationResult):
            raise TypeError("result must be a SimulationResult")
        if result.diverged:
            return self.penalty

        cost = (
            self.w_error * self.compute_itae(result)
            + self.w_control * self.compute_control_effort(result)
            + self.w_overshoot * _hinge(self.compute_overshoot(result),
                                        self.Mp_desired, self.Mp_max)
            + self.w_settling * _hinge(self.compute_settling_time(result),
                                       self.Ts_desired, self.Ts_max)
        )
        if self.U_max is not None and result.controls.size:
            peak_input = float(np.max(np.abs(result.controls)))
            cost += self.w_saturation * max(0.0, peak_input - self.U_max) ** 2
        return float(cost)
