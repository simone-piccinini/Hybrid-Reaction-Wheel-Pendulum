"""Domain value objects — the typed contracts passed between layers.

Implemented as frozen dataclasses with read-only float64 arrays, supporting the
determinism contract (``data_contracts.md``; ``AGENTS.md`` §7). ``core`` may
import ``numerics`` (``dependency_rules.md`` §2) — :meth:`StateSpaceModel.discretize`
uses the matrix-exponential primitive.

This module currently provides :class:`StateSpaceModel`. The remaining contracts
(``LQGConfig``, ``SearchSpace``, ``SimulationResult``, ``Dataset``,
``GPPosterior``) and the abstract interfaces will be added in subsequent files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics.matrix_exp import expm
from .constants import ATOL, RTOL, SYM_TOL


@dataclass(frozen=True)
class StateSpaceModel:
    """Linear time-invariant state-space model (``data_contracts.md`` §6).

    Continuous: ``ẋ = A x + B u``, ``y = C x + D u``. Discrete (one step):
    ``x⁺ = A x + B u``, ``y = C x + D u`` when ``is_discrete``. All arrays are
    coerced to read-only float64 (immutability supports determinism). Shapes use
    the ``notation.md`` symbols: ``A`` is ``n_x × n_x``, ``B`` is ``n_x × n_u``,
    ``C`` is ``n_y × n_x``, ``D`` is ``n_y × n_u``.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    is_discrete: bool = False
    dt: float | None = None

    def __post_init__(self) -> None:
        A = np.array(self.A, dtype=np.float64)  # copy: we make these read-only
        B = np.array(self.B, dtype=np.float64)
        C = np.array(self.C, dtype=np.float64)
        D = np.array(self.D, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square (n_x, n_x), got {A.shape}")
        n_x = A.shape[0]
        if B.ndim != 2 or B.shape[0] != n_x:
            raise ValueError(f"B must be (n_x, n_u) with n_x={n_x}, got {B.shape}")
        n_u = B.shape[1]
        if C.ndim != 2 or C.shape[1] != n_x:
            raise ValueError(f"C must be (n_y, n_x) with n_x={n_x}, got {C.shape}")
        n_y = C.shape[0]
        if D.shape != (n_y, n_u):
            raise ValueError(f"D must be (n_y, n_u)=({n_y}, {n_u}), got {D.shape}")
        if self.is_discrete:
            if self.dt is None or self.dt <= 0:
                raise ValueError("a discrete model requires dt > 0")
        elif self.dt is not None:
            raise ValueError("a continuous model must not set dt")
        for arr in (A, B, C, D):
            arr.flags.writeable = False
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "D", D)

    @property
    def n_x(self) -> int:
        """State dimension."""
        return self.A.shape[0]

    @property
    def n_u(self) -> int:
        """Input dimension."""
        return self.B.shape[1]

    @property
    def n_y(self) -> int:
        """Output dimension."""
        return self.C.shape[0]

    def discretize(self, dt: float) -> "StateSpaceModel":
        """Return a **new** zero-order-hold discrete model at step ``dt``.

        Uses the augmented matrix exponential (``model.md``, "Discretise";
        ``data_contracts.md`` §6):

            expm([[A, B], [0, 0]] · dt) = [[A_d, B_d], [0, I]],

        so ``A_d`` and ``B_d`` are the top-left and top-right blocks. ``C`` and
        ``D`` carry over unchanged. Does not mutate ``self``.

        Raises
        ------
        ValueError
            If the model is already discrete, or ``dt <= 0``.
        """
        if self.is_discrete:
            raise ValueError("model is already discrete")
        if dt <= 0:
            raise ValueError("dt must be > 0")
        n_x, n_u = self.n_x, self.n_u
        aug = np.zeros((n_x + n_u, n_x + n_u), dtype=np.float64)
        aug[:n_x, :n_x] = self.A
        aug[:n_x, n_x:] = self.B
        transition = expm(aug * dt)
        A_d = transition[:n_x, :n_x]
        B_d = transition[:n_x, n_x:]
        return StateSpaceModel(A_d, B_d, self.C, self.D, is_discrete=True, dt=dt)


# Packing order for LQGConfig.to_vector / from_vector — fixed once, changing it
# is a breaking change (data_contracts.md §1). θ stacks the diagonals of the four
# weight matrices, in log-space:
#   θ = [ log diag(Q_lqr) | log diag(R_lqr) | log diag(W_process) | log diag(V_measure) ]
@dataclass(frozen=True, eq=False)
class LQGConfig:
    """The four LQG weight matrices and their packing to/from θ (``data_contracts.md`` §1).

    Each weight is modelled as a **diagonal** matrix with strictly-positive
    diagonal — the documented default packing: every state, input, and output
    carries one positive weight, and those weights are the free parameters the
    Bayesian optimiser searches in log-space. Off-diagonal terms are not
    modelled (a possible future extension). Strict positivity is required
    because the packing is logarithmic (it slightly strengthens the PSD invariant
    of ``data_contracts.md`` §1, and matches log-space search).

    Fields (``notation.md`` §1, §6):
      ``Q_lqr`` ``n_x × n_x`` LQR state cost; ``R_lqr`` ``n_u × n_u`` LQR control
      cost; ``W_process`` ``n_x × n_x`` Kalman process-noise covariance;
      ``V_measure`` ``n_y × n_y`` Kalman measurement-noise covariance.
    """

    Q_lqr: np.ndarray
    R_lqr: np.ndarray
    W_process: np.ndarray
    V_measure: np.ndarray

    def __post_init__(self) -> None:
        for name in ("Q_lqr", "R_lqr", "W_process", "V_measure"):
            M = np.array(getattr(self, name), dtype=np.float64)  # copy -> read-only
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError(f"{name} must be square 2-D, got {M.shape}")
            off_diagonal = M - np.diag(np.diag(M))
            if float(np.max(np.abs(off_diagonal))) > SYM_TOL:
                raise ValueError(
                    f"{name} must be diagonal (the default packing models diagonals only)"
                )
            if np.any(np.diag(M) <= 0.0):
                raise ValueError(
                    f"{name} must have a strictly-positive diagonal (log-space packing)"
                )
            M.flags.writeable = False
            object.__setattr__(self, name, M)

    @property
    def n_x(self) -> int:
        """State dimension."""
        return self.Q_lqr.shape[0]

    @property
    def n_u(self) -> int:
        """Input dimension."""
        return self.R_lqr.shape[0]

    @property
    def n_y(self) -> int:
        """Output dimension."""
        return self.V_measure.shape[0]

    @property
    def dim(self) -> int:
        """Dimension ``d`` of the packed decision vector θ."""
        return 2 * self.n_x + self.n_u + self.n_y

    def to_vector(self) -> np.ndarray:
        """Pack the matrix diagonals into θ in log-space (``data_contracts.md`` §1).

        Order: ``log diag(Q_lqr)``, ``log diag(R_lqr)``, ``log diag(W_process)``,
        ``log diag(V_measure)``. Inverse of :meth:`from_vector`.
        """
        diagonals = np.concatenate(
            [
                np.diag(self.Q_lqr),
                np.diag(self.R_lqr),
                np.diag(self.W_process),
                np.diag(self.V_measure),
            ]
        )
        return np.log(diagonals)

    @classmethod
    def from_vector(cls, theta, n_x: int, n_u: int, n_y: int) -> "LQGConfig":
        """Rebuild the diagonal weight matrices from θ — inverse of :meth:`to_vector`.

        The dimensions ``(n_x, n_u, n_y)`` are required to split ``theta`` (its
        length alone is ambiguous). Exponentiates out of log-space and forms
        diagonal matrices, guaranteeing the round-trip
        ``from_vector(c.to_vector(), c.n_x, c.n_u, c.n_y) == c``
        (``data_contracts.md`` §1).
        """
        theta = np.asarray(theta, dtype=np.float64)
        expected = 2 * n_x + n_u + n_y
        if theta.shape != (expected,):
            raise ValueError(f"theta must have shape ({expected},), got {theta.shape}")
        values = np.exp(theta)
        i = 0
        q = values[i : i + n_x]
        i += n_x
        r = values[i : i + n_u]
        i += n_u
        w = values[i : i + n_x]
        i += n_x
        v = values[i : i + n_y]
        return cls(np.diag(q), np.diag(r), np.diag(w), np.diag(v))

    def __eq__(self, other: object) -> bool:
        """Tolerance-based equality, supporting the log/exp round-trip contract."""
        if not isinstance(other, LQGConfig):
            return NotImplemented
        return all(
            a.shape == b.shape and np.allclose(a, b, rtol=RTOL, atol=ATOL)
            for a, b in (
                (self.Q_lqr, other.Q_lqr),
                (self.R_lqr, other.R_lqr),
                (self.W_process, other.W_process),
                (self.V_measure, other.V_measure),
            )
        )

    __hash__ = None  # mutable-by-value semantics; not hashable


@dataclass(frozen=True, eq=False)
class SearchSpace:
    """Bounded domain for the Bayesian-optimisation outer loop (``data_contracts.md`` §2).

    Operates in the **search coordinates** the GP consumes — the same coordinates
    :meth:`LQGConfig.to_vector` emits, i.e. log-space for the dimensions flagged in
    ``log_scale``. The bounds are therefore given in those coordinates (log-values
    for log-scale dims); ``sample`` and ``clip`` work directly on this box, so the
    contract "every θ entering the GP/acquisition lies within these bounds" is
    expressed in the kernel's own coordinates (``data_contracts.md`` §4). ``log_scale``
    records which dimensions are logarithmic so a point can be mapped back to
    natural units with :meth:`to_natural`. Positive-only quantities (variances,
    cost weights) use ``log_scale = True``.

    Fields (``notation.md`` §6): ``dimension`` is ``d``; ``lower_bounds`` and
    ``upper_bounds`` are length ``d``; ``log_scale`` is a length-``d`` bool mask
    aligned with the packing order of :meth:`LQGConfig.to_vector`.
    """

    dimension: int
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    log_scale: np.ndarray

    def __post_init__(self) -> None:
        d = int(self.dimension)
        lower = np.array(self.lower_bounds, dtype=np.float64)
        upper = np.array(self.upper_bounds, dtype=np.float64)
        log_scale = np.array(self.log_scale, dtype=bool)
        for name, arr in (("lower_bounds", lower), ("upper_bounds", upper), ("log_scale", log_scale)):
            if arr.ndim != 1 or arr.shape[0] != d:
                raise ValueError(f"{name} must be 1-D of length dimension={d}, got {arr.shape}")
        if np.any(lower >= upper):
            raise ValueError("each lower bound must be strictly less than its upper bound")
        for arr in (lower, upper, log_scale):
            arr.flags.writeable = False
        object.__setattr__(self, "dimension", d)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        object.__setattr__(self, "log_scale", log_scale)

    @property
    def width(self) -> np.ndarray:
        """Per-dimension box width ``upper − lower``."""
        return self.upper_bounds - self.lower_bounds

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points by Latin-hypercube sampling within the box.

        Stratified per dimension: each of the ``n`` strata ``[i/n, (i+1)/n)`` is
        sampled once and the strata are independently permuted across rows, then
        scaled to ``[lower, upper]`` (``data_contracts.md`` §2). Reproducible from
        the seeded ``rng`` (determinism contract §6).

        Returns
        -------
        (n, d) ndarray
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        d = self.dimension
        unit = np.empty((n, d), dtype=np.float64)
        for j in range(d):
            strata = rng.permutation(n)
            unit[:, j] = (strata + rng.random(n)) / n
        return self.lower_bounds + unit * self.width

    def clip(self, theta) -> np.ndarray:
        """Project a point back into the box (``data_contracts.md`` §2)."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.dimension,):
            raise ValueError(f"theta must have shape ({self.dimension},), got {theta.shape}")
        return np.clip(theta, self.lower_bounds, self.upper_bounds)

    def contains(self, theta, tol: float = ATOL) -> bool:
        """True if ``theta`` lies within the box (within ``tol``)."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.dimension,):
            return False
        return bool(
            np.all(theta >= self.lower_bounds - tol)
            and np.all(theta <= self.upper_bounds + tol)
        )

    def to_natural(self, theta) -> np.ndarray:
        """Map a search-coordinate point to natural units (``exp`` on log-scale dims).

        Inverse log-transform for the dimensions flagged in ``log_scale``; linear
        dimensions pass through unchanged. Useful for reporting the discovered
        weights in physical units.
        """
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.dimension,):
            raise ValueError(f"theta must have shape ({self.dimension},), got {theta.shape}")
        natural = theta.copy()
        natural[self.log_scale] = np.exp(theta[self.log_scale])
        return natural


class Dataset:
    """The Bayesian-optimisation observation history 𝒟ₙ (``data_contracts.md`` §4).

    Owned by the ``GaussianProcess``; read-only to the ``BayesianOptimizer``.
    Holds the evaluated points ``X`` (``n × d``) and their scalar costs ``y``
    (``n``). ``X`` rows are in SearchSpace / search coordinates (log-space where
    applicable), consistent with what the kernel consumes. Unlike the other
    contracts this object is **mutable**: :meth:`append` grows the history.
    Non-finite targets are rejected so they cannot poison the GP
    (``numerical_standards.md`` §7).
    """

    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._dim = int(dim)
        self._X = np.empty((0, self._dim), dtype=np.float64)
        self._y = np.empty((0,), dtype=np.float64)
        self._X.flags.writeable = False
        self._y.flags.writeable = False

    @property
    def dim(self) -> int:
        """Decision-space dimension ``d``."""
        return self._dim

    @property
    def X(self) -> np.ndarray:
        """The ``n × d`` matrix of evaluated points (read-only)."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """The ``n`` vector of observed costs (read-only)."""
        return self._y

    def append(self, theta, y) -> None:
        """Append one observation ``(θ, y)``; rejects non-finite values (§7).

        Raises
        ------
        ValueError
            If ``theta`` has the wrong shape, or ``theta``/``y`` is non-finite.
        """
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self._dim,):
            raise ValueError(f"theta must have shape ({self._dim},), got {theta.shape}")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta contains non-finite values")
        y = float(y)
        if not np.isfinite(y):
            raise ValueError("y must be finite (a diverged run maps to a large finite PENALTY)")
        new_X = np.vstack([self._X, theta.reshape(1, self._dim)])
        new_y = np.concatenate([self._y, np.array([y], dtype=np.float64)])
        new_X.flags.writeable = False
        new_y.flags.writeable = False
        self._X = new_X
        self._y = new_y

    def size(self) -> int:
        """Number of observations ``n`` (``data_contracts.md`` §4)."""
        return self._X.shape[0]

    def __len__(self) -> int:
        return self.size()


@dataclass(frozen=True, eq=False)
class GPPosterior:
    """Predictive distribution from ``GaussianProcess.predict`` (``data_contracts.md`` §5).

    ``mean`` and ``variance`` are length-``m`` (one entry per query point; ``m=1``
    for a single query). Variance is non-negative: tiny round-off negatives are
    clipped to ``0``, but a value below ``−ATOL`` is treated as a bug and raises
    (``numerical_standards.md`` §4).
    """

    mean: np.ndarray
    variance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.array(np.atleast_1d(self.mean), dtype=np.float64)
        variance = np.array(np.atleast_1d(self.variance), dtype=np.float64)
        if mean.ndim != 1 or variance.ndim != 1:
            raise ValueError("mean and variance must be 1-D")
        if mean.shape != variance.shape:
            raise ValueError(f"mean {mean.shape} and variance {variance.shape} must match")
        if np.any(variance < -ATOL):
            raise ValueError("predictive variance is negative beyond round-off (a bug)")
        np.maximum(variance, 0.0, out=variance)  # clip round-off negatives to 0
        mean.flags.writeable = False
        variance.flags.writeable = False
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)

    def __len__(self) -> int:
        return self.mean.shape[0]
