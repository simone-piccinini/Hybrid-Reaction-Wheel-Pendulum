#!/usr/bin/env python3
"""Two-stage tuning: a broad exploration, then a warm-started refinement.

Experiment 1 (exploration)
    Tune the LQG weights of a sensible pendulum over the full search box, with
    several rollouts averaged per evaluation (n_rollouts_per_eval) so the noisy
    cost is estimated reliably.

Experiment 2 (refinement, reusing experiment 1)
    Reuse the useful data from experiment 1 in two ways:
      1. warm-start — every evaluation from experiment 1 is fed to the new GP
         before the search begins, so it starts already informed;
      2. focus — the new search box is a tight window around experiment 1's
         best observed point, so the budget is spent refining the good region.

The two best controllers are then re-evaluated, averaged over several seeds, for
a fair comparison. Per-run records are written under results/.

Usage
-----
    PYTHONPATH=src python scripts/two_stage_experiment.py [CONFIG] \
        [--refine-iters N] [--window W] [--compare-seeds M]
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

import numpy as np

from inverted_pendulum.experiment.manager import ExperimentManager
from inverted_pendulum.io.config_loader import SearchSpaceConfig, load_config
from inverted_pendulum.io.run_logging import save_run
from inverted_pendulum.metrics.performance_metrics import control_effort, itae
from inverted_pendulum.metrics.stability_metrics import overshoot, settling_time

N_X, N_U, N_Y = 4, 1, 2


def average_cost(engine, objective, config, seeds) -> float:
    """Mean objective of ``config`` over several seeded rollouts (fair score)."""
    return float(np.mean([objective.evaluate(engine.run(config, seed=s))
                          for s in seeds]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/pendulum_sensible.yaml")
    parser.add_argument("--refine-iters", type=int, default=10,
                        help="acquisition-driven steps in experiment 2")
    parser.add_argument("--window", type=float, default=1.2,
                        help="half-width (log-space) of the refinement box")
    parser.add_argument("--compare-seeds", type=int, default=12)
    args = parser.parse_args()

    base = load_config(args.config)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ------------------------------------------------------------------ #
    # Experiment 1 — exploration over the full box
    # ------------------------------------------------------------------ #
    print(f"[exp1] exploration: {base.optimization.n_initial}"
          f"+{base.optimization.n_iterations} evals, "
          f"{base.optimization.n_rollouts_per_eval} rollouts/eval")
    mgr1 = ExperimentManager(base)
    result1 = mgr1.run()
    mgr1.save_results(result1, Path("results") / "exp1_explore" / stamp)
    X1, y1 = result1.history_X, result1.history_y
    best_theta1 = X1[int(np.argmin(y1))]          # exp1's best OBSERVED point
    print(f"[exp1] best observed cost = {y1.min():.3f}  (reported {result1.metrics['objective']:.3f})")

    # ------------------------------------------------------------------ #
    # Experiment 2 — refinement, reusing experiment 1
    # ------------------------------------------------------------------ #
    base_space = mgr1.build_search_space()
    lower = np.maximum(best_theta1 - args.window, base_space.lower_bounds)
    upper = np.minimum(best_theta1 + args.window, base_space.upper_bounds)
    refined_cfg = dataclasses.replace(
        base,
        name=f"{base.name}_refine",
        search_space=SearchSpaceConfig(lower_bounds=lower, upper_bounds=upper),
    )
    mgr2 = ExperimentManager(refined_cfg)
    optimizer = mgr2.build_optimizer()
    rng = np.random.default_rng(base.seed)

    # (1) warm start: feed every experiment-1 evaluation to the new GP
    for theta, y in zip(X1, y1):
        optimizer.data.append(theta, float(y))
    optimizer.gp.fit(optimizer.data.X, optimizer.data.y)
    optimizer.gp.optimize_hyperparameters(rng)
    print(f"[exp2] warm-started with {optimizer.data.size()} prior evaluations; "
          f"refining in a ±{args.window} log-box for {args.refine_iters} steps")

    # (2) focused acquisition-driven steps in the tight box
    for _ in range(args.refine_iters):
        theta = optimizer.propose_next(rng)
        optimizer.data.append(theta, optimizer.evaluate_candidate(theta, rng))
        optimizer.gp.fit(optimizer.data.X, optimizer.data.y)
        optimizer.gp.optimize_hyperparameters(rng)

    best_config2 = optimizer.best_config(rng)
    best_result2 = optimizer.engine.run(best_config2, seed=base.seed)
    new_costs = optimizer.data.y[len(y1):]
    print(f"[exp2] best NEW observed cost = "
          f"{new_costs.min() if new_costs.size else float('nan'):.3f}")

    metrics2 = {
        "objective": optimizer.objective.evaluate(best_result2),
        "itae": itae(best_result2),
        "overshoot": overshoot(best_result2),
        "settling_time": settling_time(best_result2),
        "control_effort": control_effort(best_result2),
        "diverged": bool(best_result2.diverged),
        "best_observed_cost": float(np.min(optimizer.data.y)),
        "n_evaluations": int(optimizer.data.size()),
    }
    save_run(
        Path("results") / "exp2_refine" / stamp,
        config=dataclasses.asdict(refined_cfg),
        seed=base.seed,
        metrics=metrics2,
        result=best_result2,
        history=(optimizer.data.X, optimizer.data.y),
        summary={"best_theta": best_config2.to_vector(),
                 "warm_start_points": len(y1),
                 "refine_iters": args.refine_iters},
    )

    # ------------------------------------------------------------------ #
    # Fair comparison: re-score both winners over the same seeds
    # ------------------------------------------------------------------ #
    engine, objective = mgr1.build_engine(), mgr1.build_objective()
    seeds = range(args.compare_seeds)
    best_config1 = result1.best_config
    cost1 = average_cost(engine, objective, best_config1, seeds)
    cost2 = average_cost(engine, objective, best_config2, seeds)
    print(f"\n=== fair comparison (mean cost over {args.compare_seeds} seeds) ===")
    print(f"  experiment 1 winner : {cost1:.3f}")
    print(f"  experiment 2 winner : {cost2:.3f}")
    improvement = 100.0 * (cost1 - cost2) / cost1 if cost1 else 0.0
    print(f"  refinement change   : {improvement:+.1f}%  "
          f"({'better' if cost2 < cost1 else 'no improvement'})")
    print(f"\nrecords: results/exp1_explore/{stamp}  and  results/exp2_refine/{stamp}")


if __name__ == "__main__":
    main()
