"""GaussianProcess — the surrogate model of the cost surface.

Implements ``optimization.md`` §2 with Rasmussen & Williams' **Algorithm 2.1**
verbatim, on the project's own Cholesky primitives (``numerics/linalg``,
``AGENTS.md`` §3):

    L = cholesky(K + σ_n² I)            (factor once per model state)
    α = Lᵀ \\ (L \\ y)
    mean(θ*)     = k*ᵀ α                              (R&W eq. 2.25)
    v = L \\ k*
    variance(θ*) = k(θ*,θ*) − vᵀ v                    (R&W eq. 2.26)
    log p(y|X)   = −½ yᵀα − Σᵢ log Lᵢᵢ − (n/2) log 2π  (R&W eq. 2.30)

Targets are **standardised** to zero mean and unit variance before fitting and
the transform is stored so predictions map back (``optimization.md`` §6,
``numerical_standards.md`` §7) — the zero-mean GP prior then refers to the
standardised costs, and the noise ``σ_n²`` is likewise in standardised units.

Conditioning follows ``numerical_standards.md`` §4: the factored matrix is
always ``K + (σ_n² + PSD_JITTER) I``; if the factorisation still fails, the
jitter escalates geometrically (×10) up to ``JITTER_MAX`` with a warning, and
beyond that the model is declared misspecified and raises.

Beyond ``predict`` (marginal moments, the ``GPPosterior`` contract), the GP
exposes the **joint** posterior over a finite set — ``data_contracts.md`` §8
requires it: Entropy Search builds ``p(θ*|D)`` by sampling functions, which
marginal variances cannot do.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..core.constants import JITTER_MAX, PSD_JITTER
from ..core.types import Dataset, GPPosterior
from ..numerics.linalg import (
    NotPositiveDefiniteError,
    cholesky,
    solve_lower,
    solve_upper,
    symmetrize,
)
from .kernels.base import Kernel


def cholesky_with_escalating_jitter(M, base_jitter: float = PSD_JITTER) -> np.ndarray:
    """Factor ``M + jitter·I`` with the §4 escalation policy.

    Starts at ``base_jitter``, multiplies by 10 on failure up to ``JITTER_MAX``
    (warning each escalation), then raises — "the model is misspecified"
    (``numerical_standards.md`` §4).
    """
    jitter = float(base_jitter)
    M = symmetrize(M)
    while True:
        try:
            return cholesky(M, jitter=jitter)
        except NotPositiveDefiniteError:
            jitter *= 10.0
            if jitter > JITTER_MAX:
                raise
            warnings.warn(
                f"Cholesky failed; escalating jitter to {jitter:g} "
                "(numerical_standards.md §4)",
                stacklevel=2,
            )


class GaussianProcess:
    """The GP surrogate ``J ~ GP(0, k)`` after standardisation (``class_diagram.md``).

    Mutable by design — :meth:`fit` and :meth:`optimize_hyperparameters`
    re-condition the model — but fully deterministic: the state is a pure
    function of the data and hyperparameters.

    Parameters
    ----------
    kernel : Kernel
        The covariance function (ARD; ``data_contracts.md`` §7).
    noise_variance : float
        Observation-noise variance ``σ_n²`` in **standardised** target units
        (> 0); learned by ML-II alongside the kernel hyperparameters.
    data : Dataset, optional
        The observation history this GP owns (``data_contracts.md`` §4);
        a fresh empty one is created if omitted.
    """

    def __init__(self, kernel: Kernel, noise_variance: float = 1e-2,
                 data: Dataset | None = None) -> None:
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be a Kernel")
        noise_variance = float(noise_variance)
        if noise_variance <= 0.0:
            raise ValueError("noise_variance must be > 0")
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.data = data if data is not None else Dataset(kernel.dim)
        if self.data.dim != kernel.dim:
            raise ValueError(
                f"data dimension {self.data.dim} != kernel dimension {kernel.dim}"
            )
        self._fitted = False

    # ------------------------------------------------------------------ #
    # fitting — R&W Algorithm 2.1, lines 1-2
    # ------------------------------------------------------------------ #
    def fit(self, X, y) -> None:
        """Condition the model on observations (R&W Alg. 2.1, factorisation).

        Standardises ``y`` (``optimization.md`` §6), forms
        ``K + (σ_n² + jitter) I``, factors it once, and solves for ``α``.
        The factorisation is reused by every prediction until the next fit.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.shape[0] != y.shape[0] or X.shape[0] < 1:
            raise ValueError(f"need matching, non-empty X {X.shape} and y {y.shape}")
        if X.shape[1] != self.kernel.dim:
            raise ValueError(
                f"X must have {self.kernel.dim} columns, got {X.shape[1]}"
            )
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise ValueError("non-finite training data (§7)")

        # target standardisation: store the transform so predictions map back
        self._y_mean = float(np.mean(y))
        spread = float(np.std(y))
        self._y_std = spread if spread > 0.0 else 1.0  # constant targets guard
        y_standardized = (y - self._y_mean) / self._y_std

        K = self.kernel.matrix(X, X)
        self._L_chol = cholesky_with_escalating_jitter(
            K + self.noise_variance * np.eye(X.shape[0])
        )
        self._alpha = solve_upper(
            self._L_chol.T, solve_lower(self._L_chol, y_standardized)
        )
        self._X_train = X.copy()
        self._y_standardized = y_standardized
        self._fitted = True

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("the GP must be fitted before use")

    @property
    def train_inputs(self) -> np.ndarray:
        """The conditioning inputs ``X`` (read-only copy)."""
        self._require_fitted()
        return self._X_train.copy()

    @property
    def observed_targets(self) -> np.ndarray:
        """The observed costs ``y`` in **raw** units (read-only copy).

        Acquisitions need the incumbent — the best cost seen so far — in the
        same units predictions are returned in (Expected Improvement,
        ``data_contracts.md`` §8).
        """
        self._require_fitted()
        return self._y_mean + self._y_std * self._y_standardized

    @property
    def n_train(self) -> int:
        """Number of conditioning observations."""
        self._require_fitted()
        return self._X_train.shape[0]

    @property
    def observation_noise_variance(self) -> float:
        """``σ_n²`` mapped back to raw target units (``σ_y² · σ_n²``).

        Entropy Search needs the noise of a *hypothetical raw observation*
        when it marginalises the outcome (``optimization.md`` §4.3).
        """
        self._require_fitted()
        return self.noise_variance * self._y_std**2

    # ------------------------------------------------------------------ #
    # prediction — R&W Alg. 2.1, lines 4-6
    # ------------------------------------------------------------------ #
    def predict(self, X_query) -> GPPosterior:
        """Marginal predictive moments at the query points (R&W eqs. 2.25-2.26).

        Returns the :class:`GPPosterior` contract (mean and variance per
        point, raw target units; variance round-off clipped at 0 by the
        contract itself).
        """
        self._require_fitted()
        X_query = np.atleast_2d(np.asarray(X_query, dtype=np.float64))
        k_star = self.kernel.matrix(self._X_train, X_query)  # (n, m)
        mean_standardized = k_star.T @ self._alpha
        v = solve_lower(self._L_chol, k_star)  # (n, m)
        k_star_star = np.array(
            [self.kernel.covariance(x, x) for x in X_query]
        )
        variance_standardized = k_star_star - np.sum(v * v, axis=0)
        return GPPosterior(
            mean=self._y_mean + self._y_std * mean_standardized,
            variance=self._y_std**2 * variance_standardized,
        )

    def joint_posterior(self, X_query) -> tuple[np.ndarray, np.ndarray]:
        """Joint posterior ``(mean, covariance)`` over a finite query set.

        The full-covariance form of eq. 2.26, ``Σ = K** − Vᵀ V`` with
        ``V = L \\ K*`` — required by Entropy Search to sample whole cost
        functions over its representer points (``data_contracts.md`` §8;
        ``optimization.md`` §4.3). Raw target units.
        """
        self._require_fitted()
        X_query = np.atleast_2d(np.asarray(X_query, dtype=np.float64))
        k_star = self.kernel.matrix(self._X_train, X_query)
        mean = self._y_mean + self._y_std * (k_star.T @ self._alpha)
        v = solve_lower(self._L_chol, k_star)
        prior_cov = self.kernel.matrix(X_query, X_query)
        covariance = self._y_std**2 * (prior_cov - v.T @ v)
        return mean, symmetrize(covariance)

    def sample_posterior(
        self, X_query, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``n_samples`` joint draws of the cost over ``X_query`` (raw units).

        ``f = mean + L_Σ z`` with ``Σ = L_Σ L_Σᵀ`` (jitter-escalated Cholesky)
        and ``z`` standard normal from the threaded generator — the
        function-sampling route Entropy Search builds ``p_min`` from
        (``optimization.md`` §4.3, "Monte Carlo over GP samples").

        Returns
        -------
        (n_samples, m) ndarray
        """
        mean, covariance = self.joint_posterior(X_query)
        L_cov = cholesky_with_escalating_jitter(covariance)
        z = rng.standard_normal((n_samples, mean.shape[0]))
        return mean + z @ L_cov.T

    # ------------------------------------------------------------------ #
    # marginal likelihood — R&W Alg. 2.1, line 7
    # ------------------------------------------------------------------ #
    def log_marginal_likelihood(self) -> float:
        """``log p(y|X, φ)`` of the **standardised** targets (R&W eq. 2.30).

        The quantity ML-II maximises (``optimization.md`` §3); standardised
        units, consistent with the prior and ``noise_variance``.
        """
        self._require_fitted()
        n = self.n_train
        data_fit = -0.5 * float(self._y_standardized @ self._alpha)
        complexity = -float(np.sum(np.log(np.diag(self._L_chol))))
        constant = -0.5 * n * np.log(2.0 * np.pi)
        return data_fit + complexity + constant

    def optimize_hyperparameters(self, rng: np.random.Generator, **kwargs) -> float:
        """Refit ``φ = (ℓ, σ_f, σ_n)`` by ML-II and re-condition (``optimization.md`` §3).

        Delegates to :func:`marginal_likelihood.fit_hyperparameters` (gradient
        ascent in log-space from multiple seeded restarts,
        ``numerical_standards.md`` §5), installs the winning kernel and noise,
        refits, and returns the achieved log marginal likelihood.
        """
        self._require_fitted()
        from .marginal_likelihood import fit_hyperparameters

        kernel, noise_variance, _ = fit_hyperparameters(
            self.kernel, self.noise_variance, self._X_train,
            self._y_standardized, rng, **kwargs,
        )
        self.kernel = kernel
        self.noise_variance = noise_variance
        # refit on the raw targets (standardisation transform is unchanged)
        raw_y = self._y_mean + self._y_std * self._y_standardized
        self.fit(self._X_train, raw_y)
        return self.log_marginal_likelihood()
