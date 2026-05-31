"""Domain-level constants.

Re-exports the numerics tolerances so that upper layers depend on ``core`` for
constants instead of reaching down into ``numerics`` directly, and adds the
physical/optimisation domain constants. ``core`` may import ``numerics``
(``docs/architecture/dependency_rules.md`` §2).
"""

from ..numerics.constants import (
    ATOL,
    COND_WARN,
    JITTER_MAX,
    ML2_GRAD_TOL,
    ML2_MAX_ITER,
    ML2_RESTARTS,
    ML2_STEP_TOL,
    PSD_JITTER,
    RTOL,
    SYM_TOL,
    VALIDATION_RTOL,
)

GRAVITY: float = 9.81
"""Gravitational acceleration g (m/s²); ``notation.md`` §2."""

PENALTY: float = 1e3
"""Finite cost for a diverged run, protecting the GP (``numerical_standards.md`` §7)."""

__all__ = [
    "ATOL",
    "RTOL",
    "SYM_TOL",
    "PSD_JITTER",
    "VALIDATION_RTOL",
    "JITTER_MAX",
    "COND_WARN",
    "ML2_MAX_ITER",
    "ML2_GRAD_TOL",
    "ML2_STEP_TOL",
    "ML2_RESTARTS",
    "GRAVITY",
    "PENALTY",
]
