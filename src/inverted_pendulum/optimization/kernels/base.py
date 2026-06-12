"""Kernel — the abstract covariance-function interface (``data_contracts.md`` §7).

A kernel ``k(x, x')`` defines the GP prior over cost surfaces
(``optimization.md`` §2). Implementations are **ARD** (automatic relevance
determination): one lengthscale per search dimension, so the surrogate can
learn that some LQG weights matter far more than others. The hyperparameters
``φ`` exposed here are the *kernel's* share of ``notation.md`` §7's
``φ = (ℓ, σ_f, σ_n)`` — the observation noise ``σ_n`` belongs to the
GaussianProcess, not the kernel.

``matrix`` is implemented once here as an explicit double loop over
``covariance`` — the academically transparent form; the Gram matrices in this
project are small (tens of BO observations), so clarity wins over
vectorisation. ``gradient`` returns ``∂k/∂φ`` in **natural** units; the ML-II
optimiser applies its own log-space chain rule (``marginal_likelihood``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def _validate_hyperparams(lengthscales, signal_variance: float) -> tuple[np.ndarray, float]:
    ell = np.atleast_1d(np.asarray(lengthscales, dtype=np.float64))
    if ell.ndim != 1:
        raise ValueError(f"lengthscales must be a 1-D vector, got shape {ell.shape}")
    if np.any(ell <= 0.0):
        raise ValueError("lengthscales must be strictly positive")
    signal_variance = float(signal_variance)
    if signal_variance <= 0.0:
        raise ValueError("signal_variance must be strictly positive")
    ell.flags.writeable = False
    return ell, signal_variance


@dataclass(frozen=True)
class Kernel(ABC):
    """Abstract ARD covariance function (``data_contracts.md`` §7).

    Fields (``notation.md`` §7): ``lengthscales`` is the ARD vector ``ℓ``
    (one per dimension, > 0); ``signal_variance`` is ``σ_f²`` (> 0).
    """

    lengthscales: np.ndarray
    signal_variance: float

    def __post_init__(self) -> None:
        ell, sf2 = _validate_hyperparams(self.lengthscales, self.signal_variance)
        object.__setattr__(self, "lengthscales", ell)
        object.__setattr__(self, "signal_variance", sf2)

    @property
    def dim(self) -> int:
        """Input dimension ``d`` the ARD lengthscales cover."""
        return self.lengthscales.shape[0]

    @property
    def hyperparams(self) -> np.ndarray:
        """The kernel hyperparameters as a flat vector ``(ℓ₁ … ℓ_d, σ_f)``.

        ``data_contracts.md`` §7: "The kernel exposes its hyperparameters as a
        flat vector for the ML-II optimiser." Note the last entry is the
        *amplitude* ``σ_f = √(σ_f²)``, matching ``notation.md`` §7's ``φ``.
        """
        return np.concatenate(
            [self.lengthscales, [float(np.sqrt(self.signal_variance))]]
        )

    def with_hyperparams(self, values) -> "Kernel":
        """A new kernel of the same family with hyperparams ``(ℓ₁ … ℓ_d, σ_f)``.

        The functional update ML-II uses: kernels are frozen, so retuning
        builds a fresh instance.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.dim + 1,):
            raise ValueError(
                f"expected {self.dim + 1} hyperparameters (ℓ₁…ℓ_d, σ_f), "
                f"got shape {values.shape}"
            )
        return type(self)(
            lengthscales=values[:-1], signal_variance=float(values[-1] ** 2)
        )

    @abstractmethod
    def covariance(self, x1, x2) -> float:
        """The covariance ``k(x₁, x₂)`` between two points (scalar)."""

    @abstractmethod
    def gradient(self, x1, x2) -> np.ndarray:
        """``∂k(x₁,x₂)/∂φ`` w.r.t. the natural ``(ℓ₁ … ℓ_d, σ_f)`` (length d+1).

        Consumed by the ML-II gradient (R&W eq. 5.9); the log-space chain rule
        is applied by ``marginal_likelihood``, not here.
        """

    def matrix(self, X1, X2) -> np.ndarray:
        """The Gram matrix ``k(X₁, X₂)`` (``data_contracts.md`` §7).

        Explicit double loop over :meth:`covariance` — the textbook definition,
        kept deliberately simple (the matrices here are tens-by-tens).
        ``matrix(X, X)`` is symmetric PSD by Mercer's theorem.
        """
        A = np.atleast_2d(np.asarray(X1, dtype=np.float64))
        B = np.atleast_2d(np.asarray(X2, dtype=np.float64))
        if A.shape[1] != self.dim or B.shape[1] != self.dim:
            raise ValueError(
                f"inputs must have {self.dim} columns, got {A.shape} and {B.shape}"
            )
        gram = np.empty((A.shape[0], B.shape[0]), dtype=np.float64)
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                gram[i, j] = self.covariance(A[i], B[j])
        return gram

    def _scaled_sq_distance(self, x1, x2) -> tuple[np.ndarray, float]:
        """Helper: ARD-scaled differences and ``r² = Σ ((x₁ᵢ−x₂ᵢ)/ℓᵢ)²``."""
        diff = np.asarray(x1, dtype=np.float64) - np.asarray(x2, dtype=np.float64)
        if diff.shape != (self.dim,):
            raise ValueError(f"points must have shape ({self.dim},), got {diff.shape}")
        scaled = diff / self.lengthscales
        return diff, float(scaled @ scaled)
