#!/usr/bin/env python3
"""Measure the noise of a single cost evaluation — how repeatable is one score?

The Bayesian optimiser treats "build the LQG, run the closed loop, score the
trajectory" as a noisy black box: each rollout realises fresh process and
measurement noise and a random initial tilt, so the *same* controller scores
differently every time (optimization.md §1 — the homoscedastic-noise oracle).
This script quantifies that noise directly: it fixes one controller and
re-evaluates it over many seeds, reporting the mean, spread, and coefficient of
variation of the cost (and of the headline metrics).

The controller is the tuned optimum recorded in a run's ``metadata.json`` — so
the number answers "how noisy is the cost *near the solution*", the regime the
optimiser has to resolve to pick a winner.

Usage
-----
    PYTHONPATH=src python scripts/evaluation_noise.py [CONFIG] \
        [--run RESULTS_DIR] [--seeds N]

    CONFIG    the experiment YAML (default: configs/pendulum_measured.yaml)
    --run     a results directory holding metadata.json with the tuned weights
              (default: results/measured); its (Q,R,W,V) are re-evaluated
    --seeds   number of seeds to average over (default: 50)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inverted_pendulum.core.types import LQGConfig
from inverted_pendulum.experiment.manager import ExperimentManager


def _load_tuned_config(run_dir: Path) -> LQGConfig:
    """Rebuild the tuned LQGConfig from a run's ``metadata.json`` summary."""
    meta = json.loads((run_dir / "metadata.json").read_text())
    s = meta["summary"]
    return LQGConfig(
        Q_lqr=np.diag(s["best_config_Q_diag"]),
        R_lqr=np.diag(s["best_config_R_diag"]),
        W_process=np.diag(s["best_config_W_diag"]),
        V_measure=np.diag(s["best_config_V_diag"]),
    )


def _summary(name: str, values: np.ndarray, unit: str = "") -> str:
    mean, std = float(np.mean(values)), float(np.std(values))
    cv = 100.0 * std / abs(mean) if mean else float("nan")
    return (f"  {name:16s} mean {mean:8.3f}{unit}  std {std:7.3f}{unit}  "
            f"CV {cv:5.1f}%   [min {np.min(values):.3f}, max {np.max(values):.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/pendulum_measured.yaml")
    parser.add_argument("--run", default="results/measured",
                        help="results dir with metadata.json (the tuned weights)")
    parser.add_argument("--seeds", type=int, default=50)
    args = parser.parse_args()

    manager = ExperimentManager.from_config_file(args.config)
    engine = manager.build_engine()
    objective = manager.build_objective()
    config = _load_tuned_config(Path(args.run))

    costs = np.empty(args.seeds)
    settling = np.empty(args.seeds)
    overshoot = np.empty(args.seeds)
    diverged = 0
    for i in range(args.seeds):
        result = engine.run(config, seed=i)
        costs[i] = objective.evaluate(result)
        settling[i] = objective.compute_settling_time(result)
        overshoot[i] = objective.compute_overshoot(result)
        diverged += int(result.diverged)

    print(f"Evaluation noise at the tuned optimum ('{args.run}'), "
          f"N = {args.seeds} seeds, config '{manager.config.name}':")
    print(_summary("cost y", costs))
    print(_summary("settling T_s", settling, " s"))
    print(_summary("overshoot M_p", overshoot, " %"))
    print(f"  diverged         {diverged} / {args.seeds}")
    cv = 100.0 * float(np.std(costs)) / abs(float(np.mean(costs)))
    print(f"\n  => one cost evaluation has a {cv:.0f}% coefficient of variation: "
          f"the noise the surrogate's observation-noise term must absorb, and the "
          f"reason the reported answer is the posterior mean, not the best sample.")


if __name__ == "__main__":
    main()
