#!/usr/bin/env python3
"""Run a configured LQG-tuning experiment end to end.

Entry point (``scripts/`` may import any layer): load a YAML config, run the
Bayesian-optimisation tuning loop, and write the reproducibility bundle.

Usage
-----
    PYTHONPATH=src python scripts/run_experiment.py [CONFIG] [-o OUTDIR] [--no-plots]

    CONFIG    path to the experiment YAML (default: configs/default.yaml)
    -o/--out  directory for the run record (default: results/<config name>)
    --no-plots  skip writing the trajectory/convergence PNGs

Example
-------
    PYTHONPATH=src python scripts/run_experiment.py configs/default.yaml -o results/run01
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inverted_pendulum.experiment.manager import ExperimentManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/default.yaml",
                        help="path to the experiment YAML config")
    parser.add_argument("-o", "--out", default=None,
                        help="output directory for the run record")
    parser.add_argument("--no-plots", action="store_true",
                        help="do not render trajectory/convergence plots")
    args = parser.parse_args()

    manager = ExperimentManager.from_config_file(args.config)
    out_dir = Path(args.out) if args.out else Path("results") / manager.config.name

    print(f"Running experiment '{manager.config.name}' "
          f"(seed={manager.seed}, git={manager.git_hash[:10]})")
    result = manager.run()

    metrics = result.metrics
    print("\nReported optimum (posterior-mean minimiser):")
    print(f"  objective       = {metrics['objective']:.4f}")
    print(f"  overshoot M_p    = {metrics['overshoot']:.3f} %")
    print(f"  settling T_s     = {metrics['settling_time']:.3f} s")
    print(f"  control effort   = {metrics['control_effort']:.4f} V^2 s")
    print(f"  diverged         = {metrics['diverged']}")
    print(f"  best observed    = {metrics['best_observed_cost']:.4f} "
          f"over {metrics['n_evaluations']} evaluations")

    manager.save_results(result, out_dir, plots=not args.no_plots)
    print(f"\nRun record written to: {out_dir}")


if __name__ == "__main__":
    main()
