"""Matérn-5/2 kernel with ARD lengthscales.

The standard "rougher than RBF" prior for Bayesian optimisation
(``optimization.md`` §2 names it alongside the squared exponential).
Rasmussen & Williams eq. (4.17) with ``ν = 5/2``, ARD-scaled distance:

    r = √( Σᵢ ((xᵢ − x'ᵢ)/ℓᵢ)² ),
    k(x, x') = σ_f² · (1 + √5 r + 5r²/3) · exp(−√5 r).

Sample paths are twice differentiable — often a better match for real cost
surfaces than the infinitely smooth RBF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import Kernel

_SQRT5 = math.sqrt(5.0)


@dataclass(frozen=True)
class Matern52ARD(Kernel):
    """``k = σ_f²(1 + √5r + 5r²/3)e^{−√5 r}`` with ARD ``r`` (R&W eq. 4.17)."""

    def covariance(self, x1, x2) -> float:
        """R&W eq. 4.17, ν = 5/2."""
        _, r2 = self._scaled_sq_distance(x1, x2)
        r = math.sqrt(r2)
        polynomial = 1.0 + _SQRT5 * r + 5.0 * r2 / 3.0
        return self.signal_variance * polynomial * math.exp(-_SQRT5 * r)

    def gradient(self, x1, x2) -> np.ndarray:
        """``∂k/∂(ℓ₁…ℓ_d, σ_f)`` in natural units.

        Differentiating eq. 4.17 in ``r`` gives the compact form

            ∂k/∂r = −σ_f² · (5r/3)(1 + √5 r) · e^{−√5 r},

        and the chain rule through ``r(ℓᵢ)`` (``∂r/∂ℓᵢ = −(xᵢ−x'ᵢ)²/(ℓᵢ³ r)``)
        yields ``∂k/∂ℓᵢ = σ_f² (5/3)(1+√5r) e^{−√5r} (xᵢ−x'ᵢ)²/ℓᵢ³`` — the
        ``r`` in the prefactor cancels, so the expression is smooth at ``r = 0``
        (where the gradient is zero). ``∂k/∂σ_f = 2k/σ_f`` as for any
        ``k ∝ σ_f²``.
        """
        diff, r2 = self._scaled_sq_distance(x1, x2)
        r = math.sqrt(r2)
        decay = math.exp(-_SQRT5 * r)
        grad = np.empty(self.dim + 1, dtype=np.float64)
        grad[:-1] = (
            self.signal_variance
            * (5.0 / 3.0)
            * (1.0 + _SQRT5 * r)
            * decay
            * diff**2
            / self.lengthscales**3
        )
        k = self.signal_variance * (1.0 + _SQRT5 * r + 5.0 * r2 / 3.0) * decay
        grad[-1] = 2.0 * k / math.sqrt(self.signal_variance)
        return grad
