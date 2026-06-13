"""Entropy Search — the project's goal acquisition (``optimization.md`` §4).

Unlike EI/UCB, Entropy Search maintains an explicit belief over the **location
of the optimum**, ``p_min(θ) = Pr[θ = argmin J]`` (§4.1), and chooses the query
that is expected to reduce the Shannon entropy of that belief the most — the
mutual information between the minimiser and the would-be observation (§4.2):

    α_ES(θ) = H[p_min] − E_{y_θ}[ H[p_min | y_θ] ].

``p_min`` has no closed form, so it is estimated by the Monte-Carlo route of
§4.3 (the simple, "asymptotically exact" one; Expectation Propagation is the
differentiable alternative we do not need for candidate-set maximisation):

  • **Representer points** ``R`` discretise the domain; they are drawn from a
    non-uniform proposal (Expected Improvement renormalised as a density —
    §4.3), which concentrates resolution where ``p_min`` has mass.
  • **p_min on R** by Monte Carlo: draw ``n_optimum_samples`` joint posterior
    samples of the cost vector over ``R``, take each sample's ``argmin``;
    ``p_min(θ_r)`` is the fraction of samples minimised at ``θ_r``.
  • **Marginalising the outcome** ``y_θ ~ N(μ(θ), σ²(θ) + σ_n²)``: average the
    post-observation entropy over a few fantasy outcomes, each applying a
    **fantasy update** to the posterior. Per §4.3 the predictive covariance
    change is deterministic and only the mean shift carries the innovation —
    realised here as exact Gaussian conditioning of the joint posterior over
    ``R`` on the candidate's fantasised value (the candidate is included in
    ``R``, §4.3).

Determinism (§6): the candidate set, representer set, GP function samples and
fantasy outcomes all flow from the threaded ``rng``; the standard-normal draws
are taken **once per** :meth:`select` **and reused** across candidates, so the
acquisition is a smooth, reproducible function of the candidate set.

This is deliberately the clear, loop-based realisation of the algorithm
(academic transparency over speed): nested Monte-Carlo, O(N³) per candidate.
"""

from __future__ import annotations

import numpy as np

from ..gaussian_process import GaussianProcess, cholesky_with_escalating_jitter
from ...core.types import SearchSpace
from .base import AcquisitionFunction, expected_improvement_values


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy ``H[p] = −Σ pᵢ log pᵢ`` in nats (zeros contribute 0)."""
    p = np.asarray(p, dtype=np.float64)
    nonzero = p[p > 0.0]
    return float(-(nonzero * np.log(nonzero)).sum())


def _pmin_from_samples(samples: np.ndarray, n_points: int) -> np.ndarray:
    """Empirical ``p_min`` over points from joint cost samples (§4.3, MC route).

    ``samples`` is ``(n_samples, n_points)``; each row is one sampled cost
    vector over the points. ``p_min[r]`` is the fraction of rows whose minimum
    is at column ``r``.
    """
    minimisers = np.argmin(samples, axis=1)
    counts = np.bincount(minimisers, minlength=n_points)
    return counts / float(samples.shape[0])


class EntropySearch(AcquisitionFunction):
    """Information-theoretic acquisition over the optimum's location (§4).

    Parameters
    ----------
    n_optimum_samples : int
        GP function samples used to estimate ``p_min`` (``S``; ``class_diagram``).
    n_representers : int
        Representer points ``N`` discretising the domain.
    n_fantasies : int
        Fantasy outcomes averaged when marginalising ``y_θ``.
    n_candidates : int
        Candidate-set size for the §4.5 maximisation.
    proposal_pool : int
        Pool size the EI-weighted representer resampling draws from (§4.3).
    """

    def __init__(
        self,
        n_optimum_samples: int = 200,
        n_representers: int = 50,
        n_fantasies: int = 10,
        n_candidates: int = 200,
        proposal_pool: int = 500,
    ) -> None:
        super().__init__(n_candidates=n_candidates)
        for name, value in (
            ("n_optimum_samples", n_optimum_samples),
            ("n_representers", n_representers),
            ("n_fantasies", n_fantasies),
            ("proposal_pool", proposal_pool),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be >= 1")
        self.n_optimum_samples = int(n_optimum_samples)
        self.n_representers = int(n_representers)
        self.n_fantasies = int(n_fantasies)
        self.proposal_pool = int(proposal_pool)

    # ------------------------------------------------------------------ #
    # representer proposal — §4.3 "non-uniform proposal measure"
    # ------------------------------------------------------------------ #
    def sample_representers(
        self, gp: GaussianProcess, space: SearchSpace, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw ``n_representers`` points proportional to Expected Improvement.

        A pool is sampled uniformly from ``space``; each point is weighted by
        its (non-negative) EI and the representers are resampled with those
        weights (§4.3). If EI is flat-zero everywhere the proposal degenerates
        to uniform — a full-support fallback, as the doc allows.
        """
        pool = space.sample(self.proposal_pool, rng)
        incumbent = float(np.min(gp.observed_targets))
        weights = expected_improvement_values(gp, pool, incumbent)
        total = float(weights.sum())
        probabilities = None if total <= 0.0 else weights / total
        chosen = rng.choice(self.proposal_pool, size=self.n_representers,
                            replace=True, p=probabilities)
        return pool[chosen]

    # ------------------------------------------------------------------ #
    # p_min — §4.3 Monte Carlo over GP samples
    # ------------------------------------------------------------------ #
    def optimum_distribution(
        self, gp: GaussianProcess, space: SearchSpace, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(representers, p_min)`` — the belief over the optimum on a fresh ``R``.

        ``class_diagram.md``'s ``optimumDistribution``: draw representers, draw
        ``n_optimum_samples`` joint cost samples over them, return the empirical
        ``p_min``.
        """
        representers = self.sample_representers(gp, space, rng)
        samples = gp.sample_posterior(representers, self.n_optimum_samples, rng)
        return representers, _pmin_from_samples(samples, representers.shape[0])

    # ------------------------------------------------------------------ #
    # α_ES for one candidate — §4.2 / §4.3
    # ------------------------------------------------------------------ #
    def expected_entropy_reduction(
        self,
        gp: GaussianProcess,
        theta: np.ndarray,
        representers: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        """``α_ES(θ) = H[p_min] − E_{y_θ}[H[p_min | y_θ]]`` on ``R ∪ {θ}`` (§4.2).

        The candidate ``theta`` is appended to ``representers`` (§4.3) and sits
        at the last index ``c``. Both entropy terms are computed over the same
        set, so their difference is exactly the expected entropy reduction from
        observing ``θ``. The fantasy update is exact Gaussian conditioning: the
        posterior covariance over ``R ∪ {θ}`` after observing ``θ`` is
        deterministic, and each fantasy outcome only shifts the mean (§4.3).
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        points = np.vstack([representers, theta])  # candidate at index −1
        c = points.shape[0] - 1

        mean, covariance = gp.joint_posterior(points)
        L_prior = cholesky_with_escalating_jitter(covariance)

        # standard normals drawn once and reused (smoothness + determinism, §6)
        base_normals = rng.standard_normal((self.n_optimum_samples, points.shape[0]))
        fantasy_normals = rng.standard_normal(self.n_fantasies)

        # H[p_min] before any observation
        prior_samples = mean + base_normals @ L_prior.T
        entropy_before = shannon_entropy(
            _pmin_from_samples(prior_samples, points.shape[0])
        )

        # fantasy update: noisy conditioning on the candidate's value at c.
        observation_variance = covariance[c, c] + gp.observation_noise_variance
        gain = covariance[:, c] / observation_variance  # deterministic mean-shift weights
        posterior_covariance = covariance - np.outer(covariance[:, c], covariance[:, c]) / observation_variance
        L_post = cholesky_with_escalating_jitter(posterior_covariance)

        outcomes = mean[c] + np.sqrt(observation_variance) * fantasy_normals
        entropies_after = []
        for y in outcomes:
            conditioned_mean = mean + gain * (y - mean[c])
            samples = conditioned_mean + base_normals @ L_post.T
            entropies_after.append(
                shannon_entropy(_pmin_from_samples(samples, points.shape[0]))
            )
        return entropy_before - float(np.mean(entropies_after))

    # ------------------------------------------------------------------ #
    # scoring the candidate set
    # ------------------------------------------------------------------ #
    def _scores(
        self,
        gp: GaussianProcess,
        space: SearchSpace,
        candidates: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """``α_ES`` for each candidate, against one shared representer set.

        The representer set is drawn **once** from the EI proposal over the
        whole ``space`` (§4.3) and reused for every candidate, so the scores
        are comparable and the call is reproducible from ``rng``.
        """
        representers = self.sample_representers(gp, space, rng)
        return np.array(
            [
                self.expected_entropy_reduction(gp, theta, representers, rng)
                for theta in candidates
            ]
        )
