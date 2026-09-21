#!/usr/bin/env python3
"""Compare acquisition functions on the LQG-tuning problem.

Runs the same experiment with Expected Improvement, Upper/Lower Confidence
Bound, and Entropy Search across several seeds, and reports — for each
acquisition — the mean and spread of the best cost found, whether the reported
optimum stabilises the plant, and the wall-clock time. Entropy Search is the
project goal; EI and UCB are the baselines it is measured against.

Usage
-----
    PYTHONPATH=src python scripts/compare_acquisitions.py [CONFIG] \
        [--seeds N] [--n-initial N] [--n-iterations N] [--sim-time T] [--json PATH]

All artifacts (per-run records) are optional; this script's primary output is
the printed summary table and, with --json, a machine-readable summary.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from inverted_pendulum.experiment.manager import ExperimentManager
from inverted_pendulum.io.config_loader import load_config

# Acquisition presets kept small so a multi-seed sweep finishes quickly; the
# point is a fair *relative* comparison, not a converged tuning.
ACQUISITIONS = {
    "expected_improvement": {"n_candidates": 256},
    "ucb": {"beta": 2.0, "n_candidates": 256},
    "entropy_search": {
        "n_optimum_samples": 150,
        "n_representers": 25,
        "n_fantasies": 5,
        "n_candidates": 60,
    },
}


def run_one(base_config, kind: str, params: dict, seed: int) -> dict:
    """Run a single experiment and return its summary record."""
    config = dataclasses.replace(
        base_config,
        seed=seed,
        acquisition=dataclasses.replace(base_config.acquisition, kind=kind,
                                        params=params),
    )
    start = time.time()
    result = ExperimentManager(config).run()
    elapsed = time.time() - start
    return {
        "kind": kind,
        "seed": seed,
        "best_observed": result.metrics["best_observed_cost"],
        "reported_objective": result.metrics["objective"],
        "reported_diverged": result.metrics["diverged"],
        "n_evaluations": result.metrics["n_evaluations"],
        "history_y": result.history_y.tolist(),
        "seconds": elapsed,
    }


PRETTY = {
    "entropy_search": "Entropy Search",
    "expected_improvement": "Expected Improvement",
    "ucb": "UCB",
}
COLORS = {
    "entropy_search": "tab:blue",
    "expected_improvement": "tab:orange",
    "ucb": "tab:green",
}


def _detect_n_initial(records: list[dict]) -> int | None:
    """The shared initial-design length: the leading evals identical across
    acquisitions for one seed (the BO samples the same seeded design before the
    acquisition takes over). Returns None if they differ from the first eval."""
    seed = records[0]["seed"]
    histories = [np.asarray(r["history_y"]) for r in records if r["seed"] == seed]
    if len(histories) < 2:
        return None
    ref = histories[0]
    for k in range(len(ref)):
        if not all(np.isclose(h[k], ref[k]) for h in histories):
            return k or None
    return None


def _plot_convergence(records: list[dict], out_path: Path) -> None:
    """Overlay best-cost-so-far vs evaluation for each acquisition (mean ± range)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for kind in ACQUISITIONS:
        rows = [r for r in records if r["kind"] == kind]
        if not rows:
            continue
        # best-so-far per seed, then aggregate across seeds
        curves = np.array([np.minimum.accumulate(r["history_y"]) for r in rows])
        evals = np.arange(1, curves.shape[1] + 1)
        mean, lo, hi = curves.mean(0), curves.min(0), curves.max(0)
        ax.plot(evals, mean, color=COLORS[kind], lw=2.2, label=PRETTY[kind])
        ax.fill_between(evals, lo, hi, color=COLORS[kind], alpha=0.15)

    n_init = _detect_n_initial(records)
    if n_init:
        ax.axvline(n_init + 0.5, color="grey", ls="--", lw=1.0)
        ax.text(n_init + 0.7, ax.get_ylim()[1], "acquisition begins",
                rotation=90, va="top", ha="left", fontsize=8, color="grey")

    ax.set_yscale("log")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("best closed-loop cost so far (log)")
    ax.set_title(f"Acquisition convergence — mean over "
                 f"{len({r['seed'] for r in records})} seeds (band = min–max)")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nConvergence figure written to {out_path}")


def _print_summary(summary: dict) -> None:
    """Print the best-observed-cost table (shared by the sweep and replay paths)."""
    print("\n=== summary (best observed cost, lower is better) ===")
    print(f"{'acquisition':22s} {'mean':>8s} {'std':>8s} {'min':>8s} "
          f"{'stab.':>6s} {'sec':>6s}")
    for kind, s in summary.items():
        print(f"{kind:22s} {s['best_observed_mean']:8.3f} "
              f"{s['best_observed_std']:8.3f} {s['best_observed_min']:8.3f} "
              f"{s['reported_stabilised_fraction']:6.2f} {s['mean_seconds']:6.1f}")


def summarise(records: list[dict]) -> dict:
    """Aggregate per-acquisition statistics over seeds."""
    summary = {}
    for kind in ACQUISITIONS:
        rows = [r for r in records if r["kind"] == kind]
        best = np.array([r["best_observed"] for r in rows])
        stabilised = np.mean([not r["reported_diverged"] for r in rows])
        summary[kind] = {
            "best_observed_mean": float(best.mean()),
            "best_observed_std": float(best.std()),
            "best_observed_min": float(best.min()),
            "reported_stabilised_fraction": float(stabilised),
            "mean_seconds": float(np.mean([r["seconds"] for r in rows])),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-initial", type=int, default=6)
    parser.add_argument("--n-iterations", type=int, default=12)
    parser.add_argument("--sim-time", type=float, default=4.0)
    parser.add_argument("--json", default=None, help="write the summary JSON here")
    parser.add_argument("--plot", default=None,
                        help="write the convergence overlay figure (PNG) here")
    parser.add_argument("--from-json", default=None,
                        help="skip the sweep; load records from this JSON "
                             "(e.g. re-plot results/acquisition_comparison.json)")
    args = parser.parse_args()

    # replay mode: rebuild the table/figure from a saved run, no re-computation
    if args.from_json:
        records = json.loads(Path(args.from_json).read_text())["records"]
        summary = summarise(records)
        _print_summary(summary)
        if args.plot:
            _plot_convergence(records, Path(args.plot))
        return

    base = load_config(args.config)
    base = dataclasses.replace(
        base,
        simulation=dataclasses.replace(base.simulation, simulation_time=args.sim_time),
        optimization=dataclasses.replace(
            base.optimization, n_initial=args.n_initial,
            n_iterations=args.n_iterations, optimize_hyperparameters=True,
        ),
    )
    seeds = list(range(args.seeds))

    records = []
    print(f"Comparing acquisitions over {len(seeds)} seed(s), "
          f"{args.n_initial}+{args.n_iterations} evaluations, "
          f"sim_time={args.sim_time}s\n")
    for kind, params in ACQUISITIONS.items():
        for seed in seeds:
            record = run_one(base, kind, params, seed)
            records.append(record)
            print(f"  {kind:22s} seed={seed}  "
                  f"best={record['best_observed']:8.3f}  "
                  f"reported={record['reported_objective']:8.3f}  "
                  f"{'OK' if not record['reported_diverged'] else 'DIVERGED':8s}  "
                  f"{record['seconds']:5.1f}s")

    summary = summarise(records)
    _print_summary(summary)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump({"records": records, "summary": summary}, handle, indent=2)
        print(f"\nSummary written to {path}")

    if args.plot:
        _plot_convergence(records, Path(args.plot))


if __name__ == "__main__":
    main()
