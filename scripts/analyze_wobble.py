#!/usr/bin/env python3
"""Compare quiet vs wobble captures from ``firmware/y_wobble_v1``.

Answers the question that decides whether the pivot encoder can be rescued by
a calibration map: is its error a function of the swing angle ALONE, or does
out-of-plane tilt move the reading independently?

The firmware prints its own per-capture spread. This script does the part that
matters and is easy to get wrong by eye: at each swing angle it subtracts the
QUIET spread (sensor noise plus whatever the arm does on its own) from the
WOBBLE spread, so what is left is the y-tilt contribution by itself.

Usage
-----
    python scripts/analyze_wobble.py LOGFILE [LOGFILE ...]

Accepts either the raw serial capture (summary lines are parsed out) or a
directory of them. Verdict thresholds are documented in
``docs/hardware/bringup_pipeline.md``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# y-tilt contribution, degrees peak-to-peak
GOOD = 3.0      # below this: second order, a calibration map works
MARGINAL = 10.0  # above this: one reading cannot recover the true angle


def parse(paths: list[Path]) -> list[dict]:
    """Pull '# summary,...' lines out of one or more serial captures."""
    out: list[dict] = []
    for p in paths:
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("# summary,"):
                continue
            parts = line[len("# summary,"):].split(",")
            if not parts:
                continue
            rec: dict = {"kind": parts[0].strip(), "file": p.name}
            # remaining fields are key,value pairs
            for i in range(1, len(parts) - 1, 2):
                key = parts[i].strip()
                try:
                    rec[key] = float(parts[i + 1])
                except (ValueError, IndexError):
                    pass
            if "nominal_deg" in rec and "spread" in rec:
                out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()

    paths: list[Path] = []
    for s in a.logs:
        p = Path(s)
        paths.extend(sorted(p.glob("*"))) if p.is_dir() else paths.append(p)
    paths = [p for p in paths if p.is_file()]

    recs = parse(paths)
    if not recs:
        raise SystemExit("no '# summary,' lines found - keep them when you save "
                         "the serial capture")

    quiet = {r["nominal_deg"]: r for r in recs if r["kind"] == "yquiet"}
    wob = {r["nominal_deg"]: r for r in recs if r["kind"] == "ywobble"}
    angles = sorted(set(quiet) | set(wob))

    print(f"Parsed {len(recs)} capture(s) from {len(paths)} file(s)\n")
    print(f"  {'angle':>7} {'quiet':>9} {'wobble':>9} {'y-tilt':>9}  verdict")
    print(f"  {'(deg)':>7} {'spread':>9} {'spread':>9} {'alone':>9}")
    print("  " + "-" * 52)

    worst = 0.0
    missing = False
    for ang in angles:
        q = quiet.get(ang)
        w = wob.get(ang)
        qs = q["spread"] if q else float("nan")
        ws = w["spread"] if w else float("nan")
        if q is None or w is None:
            missing = True
            note = "need BOTH q and g at this angle"
            ytilt = float("nan")
        else:
            # spreads add roughly in quadrature: independent contributions
            ytilt = math.sqrt(max(0.0, ws ** 2 - qs ** 2))
            worst = max(worst, ytilt)
            note = ("ok" if ytilt < GOOD else
                    "marginal" if ytilt < MARGINAL else "TOO LARGE")
        f = lambda v: "    --   " if math.isnan(v) else f"{v:9.2f}"
        print(f"  {ang:>7.0f} {f(qs)} {f(ws)} {f(ytilt)}  {note}")

    print()
    if missing:
        print("  Some angles are missing a pair. Run both 'q <deg>' and")
        print("  'g <deg>' at each position before drawing a conclusion.\n")

    if worst == 0.0:
        return
    print(f"  WORST y-tilt contribution: {worst:.2f} deg\n")
    if worst < GOOD:
        print("  VERDICT: y-tilt is second order.")
        print("  The distortion is a repeatable function of the swing angle, so a")
        print("  calibration map inverts it. Proceed to build the map - dense")
        print("  around upright, where balancing actually lives.")
    elif worst < MARGINAL:
        print("  VERDICT: marginal.")
        print("  A map still helps, but this much residual must be carried")
        print("  honestly as measurement noise on the pivot channel, not")
        print("  calibrated away. Expect it to eat phase margin.")
        print(f"  Suggested measurement_std for theta_p: "
              f"{math.radians(worst / 2):.4g} rad (half the spread as ~1 sigma)")
    else:
        print("  VERDICT: two-variable problem.")
        print("  The reading depends on out-of-plane tilt independently of the")
        print("  swing angle, so one measurement cannot recover the true angle.")
        print("  A lookup table CANNOT fix this. The options are a mechanical")
        print("  constraint on the y play, or a second sensor observing it.")


if __name__ == "__main__":
    main()
