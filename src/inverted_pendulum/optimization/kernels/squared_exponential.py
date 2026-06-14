"""Squared-exponential (RBF) kernel with ARD lengthscales.

The default smooth prior of ``optimization.md`` §2. Rasmussen & Williams
eq. (5.1) (the ARD form of eq. 4.9):

    k(x, x') = σ_f² · exp(−½ Σᵢ ((xᵢ − x'ᵢ) / ℓᵢ)²).

Sample paths are infinitely differentiable — the strongest smoothness
assumption; compare :class:`Matern52ARD` for a rougher prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import Kernel


@dataclass(frozen=True)
class SquaredExponentialARD(Kernel):
    """``k = σ_f² exp(−½ r²)`` with ARD-scaled ``r²`` (R&W eq. 5.1)."""

    def covariance(self, x1, x2) -> float:
        """R&W eq. 5.1."""
        _, r2 = self._scaled_sq_distance(x1, x2)
        return self.signal_variance * math.exp(-0.5 * r2)

    def gradient(self, x1, x2) -> np.ndarray:
        """``∂k/∂(ℓ₁…ℓ_d, σ_f)`` in natural units.

        From eq. 5.1: ``∂k/∂ℓᵢ = k · (xᵢ−x'ᵢ)²/ℓᵢ³`` (stretching a lengthscale
        raises distant covariances) and ``∂k/∂σ_f = 2k/σ_f`` (since
        ``k ∝ σ_f²``).
        """
        diff, r2 = self._scaled_sq_distance(x1, x2)
        k = self.signal_variance * math.exp(-0.5 * r2)
        grad = np.empty(self.dim + 1, dtype=np.float64)
        grad[:-1] = k * diff**2 / self.lengthscales**3
        grad[-1] = 2.0 * k / math.sqrt(self.signal_variance)
        return grad
