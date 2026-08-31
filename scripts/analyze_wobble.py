#!/usr/bin/env python3
"""Compare quiet vs wobble captures from ``firmware/y_wobble_v1``.

Answers the question that decides whether the pivot encoder can be rescued by a
calibration map: is its error a function of the swing angle ALONE, or does
out-of-plane tilt move the reading independently?

The firmware prints a per-capture spread. This script does the part that is easy
to get wrong by eye: at each swing angle it removes the QUIET spread (sensor
noise plus whatever the arm does on its own) from the WOBBLE spread, leaving the
y-tilt contribution by itself.

Input
-----
A raw PuTTY session log is fine. Every non-data line the firmware emits starts
with ``#``, and captures are bracketed by ``#BEGIN`` / ``#END``, so a whole
session with banners, idle lines and several captures parses cleanly. The
per-capture statistics are recomputed from the raw samples where they are
present, and fall back to the firmware's own ``# summary`` line otherwise.

Usage
-----
    python scripts/analyze_wobble.py results/hw/wobble.log
    python scripts/analyze_wobble.py results/hw/            # a directory
    python scripts/analyze_wobble.py results/hw/*.log --csv out.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# y-tilt contribution, degrees peak-to-peak
GOOD = 3.0       # below this: second order, a calibration map works
MARGINAL = 10.0  # above this: one reading cannot recover the true angle


def _stats(vals: list[float]) -> dict:
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"n": float(n), "mean": mean, "min": min(vals), "max": max(vals),
            "spread": max(vals) - min(vals), "std": math.sqrt(max(0.0, var))}


# The loop's measured gain margin is 6.85 dB (x2.20), so a sensor whose local
# gain falls outside this band breaks the controller no matter how it is tuned.
GAIN_LO, GAIN_HI = 0.45, 2.20


def report_linearity(marks: list[dict]) -> None:
    """Stage 0b: local gain between consecutive '# mark' points."""
    if len(marks) < 2:
        return
    marks = sorted(marks, key=lambda m: m["true_deg"])
    print("\nLINEARITY (Stage 0b) — from '# mark' points\n")
    print(f"  {'true':>7} {'measured':>10} {'error':>8} {'local gain':>11}  ")
    prev = None
    gains = []
    for m in marks:
        t, v = m["true_deg"], m["measured_deg"]
        g = ""
        if prev is not None and t != prev[0]:
            k = (v - prev[1]) / (t - prev[0])
            gains.append((prev[0], t, k))
            flag = "" if GAIN_LO <= k <= GAIN_HI else "  <-- OUTSIDE [0.45, 2.20]"
            g = f"{k:11.2f}{flag}"
        print(f"  {t:>7.1f} {v:>10.2f} {v - t:>+8.2f} {g}")
        prev = (t, v)
    if not gains:
        return
    ks = [k for _, _, k in gains]
    span = max(ks) / min(ks) if min(ks) > 0 else float("inf")
    print(f"\n  local gain {min(ks):.2f} .. {max(ks):.2f}   ({span:.1f}x variation; "
          f"a correct mount is 1.00 everywhere)")
    bad = [(a, b, k) for a, b, k in gains if not (GAIN_LO <= k <= GAIN_HI)]
    if bad:
        print("\n  VERDICT: at least one segment is outside the loop's gain-margin")
        print("  tolerance. In that region the controller sees the wrong amount of")
        print("  tilt and no choice of (Q,R,W,V) fixes it. Segments:")
        for a, b, k in bad:
            print(f"    {a:.0f}-{b:.0f} deg  gain {k:.2f}")
    else:
        print("\n  VERDICT: every segment is inside [0.45, 2.20]. Linearity is no")
        print("  longer the blocker; judge the build on the wobble numbers above.")


def parse_file(path: Path) -> list[dict]:
    """Pull every capture out of one session log."""
    out: list[dict] = []
    cur: dict | None = None
    samples: list[float] = []
    col = None
    marks: list[dict] = []

    def flush() -> None:
        nonlocal cur, samples, col
        if cur is not None and ("spread" in cur or samples):
            if samples:                       # recompute from raw data
                cur.update(_stats(samples))
                cur["source"] = "samples"
            else:
                cur["source"] = "summary"
            if "nominal_deg" in cur and "spread" in cur:
                out.append(cur)
        cur, samples, col = None, [], None

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#BEGIN"):
            flush()
            cur = {"file": path.name}
            continue
        if line.startswith("#END"):
            flush()
            continue

        if line.startswith("#"):
            body = line.lstrip("#").strip()
            parts = [p.strip() for p in body.split(",")]
            # marks are emitted OUTSIDE any #BEGIN/#END block, so they must be
            # handled before the "are we inside a capture" guard below
            if parts[0] == "mark":
                d = {}
                for i in range(1, len(parts) - 1, 2):
                    try:
                        d[parts[i]] = float(parts[i + 1])
                    except ValueError:
                        pass
                if "true_deg" in d and "measured_deg" in d:
                    marks.append(d)
                continue
            if cur is None:
                continue
            if parts[0] == "test" and len(parts) > 1:
                cur["kind"] = parts[1]
            elif parts[0] == "nominal_deg" and len(parts) > 1:
                try:
                    cur["nominal_deg"] = float(parts[1])
                except ValueError:
                    pass
            elif parts[0] == "sample_hz" and len(parts) > 1:
                try:
                    cur["sample_hz"] = float(parts[1])
                except ValueError:
                    pass
            elif parts[0] == "summary":
                # summary,<kind>,key,value,key,value,...
                cur.setdefault("kind", parts[1] if len(parts) > 1 else "?")
                for i in range(2, len(parts) - 1, 2):
                    try:
                        cur[parts[i]] = float(parts[i + 1])
                    except ValueError:
                        pass
            continue

        # --- data ---
        if cur is None:
            continue
        parts = [p.strip() for p in line.split(",")]
        if col is None:
            if parts[0] == "t_s":
                col = parts
            continue
        if len(parts) != len(col):
            continue                     # torn line from the serial stream
        try:
            samples.append(float(parts[col.index("pivot_deg")]))
        except (ValueError, IndexError):
            continue

    flush()
    for m in marks:
        m["kind"] = "mark"
        out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", help="session log(s) or a directory")
    ap.add_argument("--csv", default=None, help="also write a tidy CSV here")
    a = ap.parse_args()

    paths: list[Path] = []
    for s in a.logs:
        p = Path(s)
        if p.is_dir():
            paths.extend(sorted(q for q in p.iterdir() if q.is_file()))
        elif p.is_file():
            paths.append(p)
    if not paths:
        raise SystemExit("no input files found")

    recs: list[dict] = []
    for p in paths:
        recs.extend(parse_file(p))
    marks = [r for r in recs if r.get("kind") == "mark"]
    recs = [r for r in recs if r.get("kind") != "mark"]
    if not recs and not marks:
        raise SystemExit("nothing found - keep the '#BEGIN'/'#END' and '# mark' "
                         "lines when saving the session log")

    # last capture wins if a position was repeated
    quiet = {r["nominal_deg"]: r for r in recs if r.get("kind") == "yquiet"}
    wob = {r["nominal_deg"]: r for r in recs if r.get("kind") == "ywobble"}
    angles = sorted(set(quiet) | set(wob))

    if marks:
        report_linearity(marks)
    if not recs:
        return

    n_samp = sum(1 for r in recs if r.get("source") == "samples")
    print(f"Parsed {len(recs)} capture(s) from {len(paths)} file(s) "
          f"({n_samp} recomputed from raw samples)\n")
    print(f"  {'angle':>7} {'quiet':>9} {'wobble':>9} {'y-tilt':>9}  verdict")
    print(f"  {'(deg)':>7} {'spread':>9} {'spread':>9} {'alone':>9}")
    print("  " + "-" * 52)

    rows, worst, missing = [], 0.0, False
    for ang in angles:
        q, w = quiet.get(ang), wob.get(ang)
        qs = q["spread"] if q else float("nan")
        ws = w["spread"] if w else float("nan")
        if q is None or w is None:
            missing = True
            ytilt, note = float("nan"), "need BOTH q and g here"
        else:
            ytilt = math.sqrt(max(0.0, ws ** 2 - qs ** 2))
            worst = max(worst, ytilt)
            note = ("ok" if ytilt < GOOD else
                    "marginal" if ytilt < MARGINAL else "TOO LARGE")
        fmt = lambda v: "    --   " if math.isnan(v) else f"{v:9.2f}"
        print(f"  {ang:>7.0f} {fmt(qs)} {fmt(ws)} {fmt(ytilt)}  {note}")
        rows.append((ang, qs, ws, ytilt, note))

    print()
    if missing:
        print("  Some angles lack a pair - run both 'q <deg>' and 'g <deg>'.\n")
    if 180.0 not in angles:
        print("  NOTE: no capture at 180 deg. Upright is the only position that\n"
              "  matters for balancing, and sensitivity varies around the circle.\n")

    if a.csv:
        with Path(a.csv).open("w") as f:
            f.write("nominal_deg,quiet_spread,wobble_spread,ytilt,verdict\n")
            for ang, qs, ws, yt, note in rows:
                f.write(f"{ang},{qs},{ws},{yt},{note}\n")
        print(f"  tidy CSV -> {a.csv}\n")

    if worst == 0.0:
        return
    print(f"  WORST y-tilt contribution: {worst:.2f} deg\n")
    if worst < GOOD:
        print("  VERDICT: y-tilt is second order.")
        print("  The distortion is a repeatable function of the swing angle, so")
        print("  a calibration map inverts it. Build the map - dense around")
        print("  upright, where balancing actually lives.")
    elif worst < MARGINAL:
        print("  VERDICT: marginal.")
        print("  A map still helps, but this much residual must be carried as")
        print("  measurement noise on the pivot channel, not calibrated away.")
        print(f"  measurement_std[0] ~ {math.radians(worst / 2):.4g} rad "
              f"({worst / 2:.2f} deg, half the spread as ~1 sigma)")
        print("  Expect it to cost phase margin - re-check the margins before")
        print("  attempting to balance.")
    else:
        print("  VERDICT: two-variable problem.")
        print("  The reading depends on out-of-plane tilt independently of the")
        print("  swing angle, so one measurement cannot recover the true angle.")
        print("  A lookup table CANNOT fix this. Options: constrain the y play")
        print("  mechanically, or add a sensor that observes it.")


if __name__ == "__main__":
    main()
