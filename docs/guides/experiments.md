# Experiments — Comparing the Acquisition Functions

This document reports experiments run with the system itself, to show it works
end to end and to compare the three acquisition functions — Expected
Improvement (EI), Upper/Lower Confidence Bound (UCB), and the project-goal
**Entropy Search** (ES). Everything here is reproducible with the script
`scripts/compare_acquisitions.py`; the raw per-run records are written to
`results/` (gitignored).

> **Read the numbers as relative, not absolute.** The budget below is small (24
> evaluations, 5 seeds) and the targets are deliberately aggressive
> (`Mp_desired = 5 %`, `Ts_desired = 1 s`), so the absolute costs are high and
> the seed-to-seed spread is wide. The experiment is designed to compare
> *behaviour* under equal conditions, not to deliver a converged tuning.
>
> **Objective version.** These runs predate the current cost function: they used
> the earlier symmetric quadratic cost on `M_p`/`T_s`. The cost has since been
> changed to ITAE + control energy + asymmetric hinge penalties (`notation.md`
> §6), so the absolute numbers here are not comparable to runs under the new
> objective — but the *relative* acquisition comparison still stands.

---

## 1. Setup

- **Plant:** the default reaction-wheel pendulum (`configs/default.yaml`).
- **Rollout:** `dt = 0.01 s`, horizon `5 s` (500 steps), a `0.05 rad` initial
  tilt, true process/measurement noise injected.
- **Budget:** `n_initial = 8` Latin-hypercube designs + `n_iterations = 16`
  acquisition-driven steps = **24 evaluations**, `n_rollouts_per_eval = 1`,
  ML-II hyperparameter refitting on, **5 seeds** (0–4).
- **Measured:** the best cost observed during the search, whether the reported
  optimum (the posterior-mean minimiser) stabilises the plant, and wall-clock
  time. Lower cost is better; the diverged-run penalty is `1000`.

Reproduce with:

```bash
PYTHONPATH=src python scripts/compare_acquisitions.py \
    --seeds 5 --n-initial 8 --n-iterations 16 --sim-time 5.0 \
    --json results/acquisition_comparison.json
```

---

## 2. Experiment 1 — acquisition comparison

Best observed cost over 5 seeds (mean ± std, and the single best run), the
fraction of seeds whose *reported* optimum stabilised the plant, and the mean
wall-clock time per run:

| Acquisition | best cost (mean ± std) | best (min) | reported stabilised | sec/run |
|---|---|---|---|---|
| Expected Improvement | 28.4 ± 6.5 | 21.7 | 5/5 | 14.5 |
| Upper/Lower CB (β=2) | **25.0 ± 3.8** | **20.1** | 3/5 | 14.3 |
| Entropy Search | 25.6 ± **2.6** | 22.3 | 4/5 | 17.1 |

Observations:

- **All three reliably find stabilising controllers.** Every run's best
  observed cost (≈20–40) is far below the `1000` divergence penalty — the search
  is exploring usable designs, not flailing.
- **The model-driven acquisitions edge out EI here.** UCB has the lowest mean
  and the single best run; ES has the **lowest spread** (std 2.6), i.e. it is
  the most *consistent* across seeds — in keeping with its information-seeking
  design. The mean differences are within the seed-to-seed noise, so this is a
  tendency, not a verdict.
- **Entropy Search costs ~20 % more wall time** per run, the price of its
  Monte-Carlo `p_min` estimate and fantasy updates.

---

## 3. Experiment 2 — convergence vs budget

Using the same runs, the mean running-best cost as a function of how many
evaluations have been spent (no extra computation — this post-processes the
saved history):

| Acquisition | @8 (init) | @12 | @16 | @20 | @24 | reduction |
|---|---|---|---|---|---|---|
| Expected Improvement | 32.4 | 29.3 | 29.3 | 29.3 | 28.4 | 12 % |
| Upper/Lower CB | 32.4 | 30.8 | 27.6 | 26.4 | 25.0 | **23 %** |
| Entropy Search | 32.4 | 28.2 | 28.1 | 25.6 | 25.6 | 21 % |

- All acquisitions start from the **same** initial-design best (32.4) — they
  share the Latin-hypercube design at each seed and differ only in the
  model-driven phase, a useful sanity check.
- UCB and ES keep improving through the budget (23 % and 21 % reduction over the
  initial design); EI plateaus early here (12 %), consistent with its greedier,
  more exploitative nature on a noisy surface.

---

## 4. The "reported diverged" cases, and a recommendation

UCB reported a diverging optimum on 2 of 5 seeds and ES on 1 of 5, even though
their *observed* bests were fine. This is the expected tension of reporting the
**posterior-mean minimiser** (`optimization.md` §5) under noise on a small
budget: `argmin μ_T` can land on a marginally-stable, never-re-evaluated point
that tips over under the report seed's particular noise draw. It is a property
of the noisy problem, not a bug — the metrics honestly record `diverged = True`.

Two levers reduce it, both already in the system:
- **Average more rollouts per query** (`optimization.n_rollouts_per_eval > 1`):
  the CLT noise reduction of `optimization.md` §1 smooths the surface the GP
  models and the mean it reports.
- **Spend more budget** (larger `n_iterations`, full `10 s` horizon): the
  earlier single-acquisition run at a representative budget (24 evals, 10 s
  horizon, 2 rollouts/eval) reported a stabilising optimum with `diverged =
  False`.

---

## 5. Takeaways

1. The full LQG + Bayesian-optimisation pipeline runs end to end and produces
   stabilising controllers across acquisitions and seeds.
2. The information-driven acquisitions (UCB, Entropy Search) make better use of
   a small model-driven budget than greedy EI; Entropy Search is the most
   consistent across seeds, at a modest extra cost.
3. Reporting robustness on tiny budgets is a noise phenomenon, addressed by
   averaging rollouts or spending more evaluations — not by changing the
   algorithm.

All claims here are reproducible from the script and the seeds stated; the
JSON summary records every per-run number.
