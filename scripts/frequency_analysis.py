#!/usr/bin/env python3
"""Bode diagrams of the linearised plant from its state-space form.

Loads a config, builds the reaction-wheel pendulum, takes its continuous
state-space linearisation, and moves to the frequency form via the Laplace
transfer function G(s) = C(sI−A)⁻¹B + D evaluated at s = jω (see
docs/guides/frequency_analysis.md). One Bode diagram is drawn per
input→output channel.

Usage
-----
    PYTHONPATH=src python scripts/frequency_analysis.py [CONFIG] \
        [-o OUTDIR] [--w-min W] [--w-max W] [--points N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inverted_pendulum.dynamics.frequency_response import bode, log_frequencies
from inverted_pendulum.experiment.manager import ExperimentManager

# Channel names for the default 2-output, 1-input plant (notation.md §3).
OUTPUT_NAMES = (r"$\theta_p$ (pendulum angle)", r"$\dot\theta_w$ (wheel rate)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/default.yaml")
    parser.add_argument("-o", "--out", default="results/bode")
    parser.add_argument("--w-min", type=float, default=1e-2)
    parser.add_argument("--w-max", type=float, default=1e3)
    parser.add_argument("--points", type=int, default=600)
    args = parser.parse_args()

    manager = ExperimentManager.from_config_file(args.config)
    model = manager.build_engine().linear_model.continuous  # continuous (A,B,C,D)
    omega = log_frequencies(args.w_min, args.w_max, args.points)

    print(f"Plant '{manager.config.name}': continuous linearisation "
          f"(n_x={model.n_x}, n_u={model.n_u}, n_y={model.n_y})")

    from inverted_pendulum.io.plotting import plot_bode, save_figure  # lazy mpl

    out_dir = Path(args.out)
    for output_index in range(model.n_y):
        data = bode(model, omega, output_index=output_index, input_index=0)
        name = (OUTPUT_NAMES[output_index] if output_index < len(OUTPUT_NAMES)
                else f"output {output_index}")
        fig = plot_bode(
            data.omega, data.magnitude_db, data.phase_deg,
            title=f"Bode: voltage → {name}",
        )
        path = save_figure(fig, out_dir / f"bode_output{output_index}.png")
        peak = data.omega[int(np.argmax(data.magnitude_db))]
        print(f"  channel {output_index} ({name}): peak |G| near "
              f"ω ≈ {peak:.2f} rad/s   ->  {path}")

    print(f"\nBode diagrams written to: {out_dir}")


if __name__ == "__main__":
    main()
