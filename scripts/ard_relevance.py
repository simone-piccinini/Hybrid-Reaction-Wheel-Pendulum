#!/usr/bin/env python3
"""Which of the 11 LQG weights does the cost actually care about? (ARD relevance)

The GP surrogate uses **automatic relevance determination**: one lengthscale per
search dimension, fit by ML-II (optimization.md §3). A *short* lengthscale means
the cost changes quickly as that weight moves — the surrogate has learned that
weight matters; a *long* one means the cost is nearly flat along it — that weight
barely moves the score within the searched range. Reading the fitted lengthscales
is therefore a sensitivity analysis of the tuning problem itself.

Method: evaluate a space-filling (Latin-hypercube) design over the search box,
fit the GP once with ML-II, and read `kernel.lengthscales`. A space-filling
design gives an *unbiased* sensitivity picture (unlike the BO's own sampling,
which concentrates near good regions). Inputs live in log-weight coordinates and
each dimension has its own box width, so relevance is reported as the
dimensionless ratio `box_width / lengthscale` — how many lengthscales of
variation the cost shows across that weight's searched range.

Caveats (this is a diagnostic, not a precise measurement):
- The cost is noisy (~20% CV, see `evaluation_noise.py`); each design point is
  averaged over `--rollouts` seeds to damp it, but the ranking is still
  indicative, not sharp.
- At a small budget the ML-II fit is **degenerate** — it drives the noise
  `sigma_n` to ~0 and inflates the irrelevant lengthscales to absurd values
  (interpolating the noise). This needs a decent design (~120 averaged points)
  to settle; the script warns when `sigma_n ~ 0` so a degenerate fit is not read
  as fact.

Usage
-----
    PYTHONPATH=src python scripts/ard_relevance.py [CONFIG] \
        [--samples N] [--rollouts R] [--seed S]
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from inverted_pendulum.core.types import LQGConfig
from inverted_pendulum.experiment.manager import ExperimentManager


def _labels(n_x: int, n_u: int, n_y: int) -> list[str]:
    """Human names for the packed dims (types.py packing: Q, R, W, V diagonals)."""
    states = (["theta_p", "theta_p_dot", "theta_w", "theta_w_dot"]
              if n_x == 4 else [f"x{i}" for i in range(n_x)])
    inputs = ["voltage"] if n_u == 1 else [f"u{i}" for i in range(n_u)]
    sensors = (["theta_p", "theta_w_dot"]
               if n_y == 2 else [f"y{i}" for i in range(n_y)])
    return ([f"Q·{s}" for s in states] + [f"R·{u}" for u in inputs]
            + [f"W·{s}" for s in states] + [f"V·{s}" for s in sensors])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/pendulum_measured.yaml")
    parser.add_argument("--samples", type=int, default=120,
                        help="number of Latin-hypercube design points (default 120)")
    parser.add_argument("--rollouts", type=int, default=5,
                        help="seeds averaged per design point, to damp cost noise (default 5)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manager = ExperimentManager.from_config_file(args.config)
    engine = manager.build_engine()
    objective = manager.build_objective()
    space = manager.build_search_space()
    gp = manager.build_gp(space.dimension)
    n_x, n_u, n_y = engine.linear_model.n_x, engine.linear_model.n_u, engine.linear_model.n_y

    rng = np.random.default_rng(args.seed)
    X = space.sample(args.samples, rng)
    y = np.empty(args.samples)
    diverged = 0
    for i, theta in enumerate(X):
        config_i = LQGConfig.from_vector(theta, n_x, n_u, n_y)
        results = [engine.run(config_i, seed=1000 * i + s) for s in range(args.rollouts)]
        y[i] = float(np.mean([objective.evaluate(r) for r in results]))
        diverged += int(all(r.diverged for r in results))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ML-II restarts may hit the iteration cap
        gp.fit(X, y)
        lml = gp.optimize_hyperparameters(rng)

    ell = gp.kernel.lengthscales
    width = space.width
    relevance = width / ell
    order = np.argsort(-relevance)  # most sensitive first
    labels = _labels(n_x, n_u, n_y)
    sigma_n = float(np.sqrt(gp.observation_noise_variance))

    print(f"ARD relevance on '{manager.config.name}' "
          f"(N = {args.samples} samples x {args.rollouts} rollouts, "
          f"{diverged} points diverged):")
    print(f"  fitted GP: log marginal likelihood {lml:.1f}, "
          f"observation noise sigma_n ~ {sigma_n:.0f} (raw cost units)")
    if sigma_n < 1.0:
        print("  ** WARNING: sigma_n collapsed to ~0 — the ML-II fit is DEGENERATE "
              "(interpolating the\n     noise); the ranking below is unreliable. "
              "Increase --samples (~120+) and re-run. **")
    print(f"\n  {'rank':>4}  {'parameter':16s} {'lengthscale':>12s} "
          f"{'box width':>10s} {'relevance':>10s}")
    for rank, j in enumerate(order, 1):
        ell_str = f"{ell[j]:12.2f}" if ell[j] < 1e4 else f"{'>1e4 (off)':>12s}"
        print(f"  {rank:>4}  {labels[j]:16s} {ell_str} "
              f"{width[j]:10.2f} {relevance[j]:10.2f}")

    top = ", ".join(labels[j] for j in order[:3])
    flat = [labels[j] for j in order if relevance[j] < 0.5]
    print(f"\n  => the cost is most sensitive to {top}.")
    if flat:
        print(f"     Nearly flat (the cost barely moves along these within the "
              f"searched range): {', '.join(flat)}.")


if __name__ == "__main__":
    main()
