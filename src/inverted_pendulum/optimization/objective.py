"""ObjectiveFunction — the scalar rollout cost the Bayesian optimiser minimises.

The oracle's scoring half (``optimization.md`` §1: the rollout cost is the
"opaque oracle" output): a pure map ``SimulationResult → finite float`` built
on the metrics layer. The cost is exactly ``notation.md`` §6:

    y = w₁ ((M_p − M_p^des) / M_p^des)² + w₂ ((T_s − T_s^des) / T_s^des)²,

a weighted, normalised quadratic distance of the realised overshoot and
settling time from their desired values. A diverged rollout is mapped to the
large **finite** ``PENALTY`` — never ``inf``/``nan``, which would poison the
GP surrogate (``data_contracts.md`` §3; ``numerical_standards.md`` §7;
``optimization.md`` §6 "Finite cost surface").

Control effort is exposed as an additional metric per ``class_diagram.md``
(``computeControlEffort``) for analysis and extensions, but the documented
default cost uses only ``M_p`` and ``T_s`` (``notation.md`` §6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.constants import PENALTY
from ..core.types import SimulationResult
from ..metrics.performance_metrics import control_effort
from ..metrics.stability_metrics import overshoot, settling_time


@dataclass(frozen=True)
class ObjectiveFunction:
    """The rollout cost ``y`` (``class_diagram.md``; ``notation.md`` §6).

    Parameters
    ----------
    Mp_desired : float
        Desired percentage overshoot ``M_p^des`` (> 0 — it is a denominator).
    Ts_desired : float
        Desired settling time ``T_s^des`` in seconds (> 0 — a denominator).
    w1, w2 : float
        Non-negative weights on the overshoot and settling-time terms.
    penalty : float
        Finite cost assigned to a diverged rollout (default ``PENALTY`` from
        ``core.constants``; ``numerical_standards.md`` §7).
    """

    Mp_desired: float
    Ts_desired: float
    w1: float = 1.0
    w2: float = 1.0
    penalty: float = PENALTY

    def __post_init__(self) -> None:
        for name in ("Mp_desired", "Ts_desired", "w1", "w2", "penalty"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.Mp_desired <= 0.0:
            raise ValueError("Mp_desired must be > 0 (it is a denominator)")
        if self.Ts_desired <= 0.0:
            raise ValueError("Ts_desired must be > 0 (it is a denominator)")
        if self.w1 < 0.0 or self.w2 < 0.0:
            raise ValueError("w1 and w2 must be >= 0")
        if not math.isfinite(self.penalty) or self.penalty <= 0.0:
            raise ValueError("penalty must be a positive finite float (§7)")

    def compute_overshoot(self, result: SimulationResult) -> float:
        """Percentage overshoot ``M_p`` of the rollout (``metrics``; notation §6)."""
        return overshoot(result)

    def compute_settling_time(self, result: SimulationResult) -> float:
        """Settling time ``T_s`` of the rollout (``metrics``; notation §6)."""
        return settling_time(result)

    def compute_control_effort(self, result: SimulationResult) -> float:
        """Integrated squared control ``∫‖u‖² dt`` (``class_diagram.md``)."""
        return control_effort(result)

    def evaluate(self, result: SimulationResult) -> float:
        """The scalar cost ``y`` of one rollout (``notation.md`` §6).

        A diverged run returns ``penalty`` directly (``data_contracts.md`` §3).
        The returned value is always finite: both metrics are finite by
        construction (overshoot is a ratio of finite trajectories; settling
        time is capped at the horizon).
        """
        if not isinstance(result, SimulationResult):
            raise TypeError("result must be a SimulationResult")
        if result.diverged:
            return self.penalty
        overshoot_term = (
            (self.compute_overshoot(result) - self.Mp_desired) / self.Mp_desired
        ) ** 2
        settling_term = (
            (self.compute_settling_time(result) - self.Ts_desired) / self.Ts_desired
        ) ** 2
        return self.w1 * overshoot_term + self.w2 * settling_term
