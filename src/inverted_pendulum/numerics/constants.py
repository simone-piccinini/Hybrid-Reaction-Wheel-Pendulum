"""Numerical tolerances and conditioning constants.

Single source of truth, transcribed verbatim from
``docs/conventions/numerical_standards.md`` §§2 and 4.

These live in the *numerics* layer rather than ``core/`` because the numerics
primitives consume them directly and this layer may import ``numpy`` only
(``docs/architecture/dependency_rules.md`` §2); a copy in ``core`` would invert
the dependency. ``core`` is expected to re-export them.
"""

# numerical_standards.md §2 — global tolerances
ATOL: float = 1e-9
"""Absolute tolerance for general floating-point comparisons."""

RTOL: float = 1e-6
"""Relative tolerance for general floating-point comparisons."""

SYM_TOL: float = 1e-8
"""Maximum allowed asymmetry ``‖M − Mᵀ‖∞`` before a matrix must be symmetrised."""

PSD_JITTER: float = 1e-9
"""Baseline diagonal jitter added before a Cholesky factorisation."""

VALIDATION_RTOL: float = 1e-6
"""Tolerance when checking against reference libraries in ``tests/validation/``."""

# numerical_standards.md §4 — SPD conditioning guards
JITTER_MAX: float = 1e-3
"""Cap for geometric jitter escalation; beyond this the model is misspecified."""

COND_WARN: float = 1e12
"""Condition-number threshold above which a warning should be logged."""
