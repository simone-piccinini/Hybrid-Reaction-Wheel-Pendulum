# Running Experiments — the `experiment` and `io` layers

This guide documents the top of the stack: how a YAML configuration becomes a
reproducible Bayesian-optimisation run that tunes the LQG weight matrices, and
what it leaves on disk. It is the practical companion to the theory briefs in
[`docs/theory/`](../theory/) and the architecture contracts in
[`docs/architecture/`](../architecture/).

By the time control reaches this layer, every lower layer is in place: the
plant and its linearisation, the LQR controller, the Kalman filter, the
simulator, the metrics, and the whole optimisation stack (GP surrogate, ML-II,
acquisitions, the BO outer loop). The `experiment` layer only *orchestrates*
them; the `io` layer only *reads configuration and writes results*.

---

## 1. The two layers at a glance

| Layer | Path | May import | Responsibility |
|---|---|---|---|
| `experiment` | `src/inverted_pendulum/experiment/` | `optimization`, `simulation`, `io`, `core` | build the object graph from a config, run the loop, record the run |
| `io` | `src/inverted_pendulum/io/` | **`core` only** | parse YAML, persist runs, plot diagnostics |

`io` is a **cross-cutting leaf** (`dependency_rules.md` §1): only `experiment`
and `scripts/` may import it, and it may itself import nothing above `core`.
This is why config parsing produces *plain data* and never constructs a plant
or an optimiser — building domain objects would require importing upper layers,
which `io` is forbidden to do. The construction lives one level up, in
`ExperimentManager`.

```
                 configs/*.yaml
                      │  io.config_loader.load_config
                      ▼
              ExperimentConfig  (plain, validated data)
                      │  experiment.ExperimentManager
                      ▼
   plant → engine → objective → GP → acquisition → BayesianOptimizer
                      │  .optimize()  (optimization.md §5)
                      ▼
               ExperimentResult
                      │  .save_results()  →  io.run_logging / io.plotting
                      ▼
        results/<name>/{metadata.json, trajectories.npz, history.npz, *.png}
```

---

## 2. Quick start

From the repository root (tests and scripts run with `PYTHONPATH=src`):

```bash
PYTHONPATH=src python scripts/run_experiment.py configs/default.yaml -o results/run01
```

That loads the config, runs the tuning loop, prints a summary, and writes the
run record to `results/run01/`. Flags:

- `CONFIG` — the experiment YAML (defaults to `configs/default.yaml`).
- `-o/--out` — output directory (defaults to `results/<config name>`).
- `--no-plots` — skip the PNG figures (no matplotlib needed).

Programmatically:

```python
from inverted_pendulum.experiment.manager import ExperimentManager

manager = ExperimentManager.from_config_file("configs/default.yaml")
result = manager.run()                       # deterministic from config.seed
print(result.metrics["objective"], result.metrics["best_observed_cost"])
manager.save_results(result, "results/run01", plots=True)
```

`ExperimentResult` carries everything about the run: `best_config`
(the reported optimum), `best_theta`, the full `history_X`/`history_y` of the
search, a representative `best_result` rollout, and the `metrics` dict.

---

## 3. The configuration schema

The canonical, fully-commented schema is [`configs/default.yaml`](../../configs/default.yaml).
A configuration has eight sections plus a `name` and a master `seed`.

| Section | Key fields | Notes |
|---|---|---|
| (top) | `name`, `seed` | `seed` threads through **all** randomness (AGENTS §7) |
| `plant` | masses, inertias, frictions, `wheel`, `motor` | SI units; inertias and `resistance` are denominators (> 0) |
| `simulation` | `dt`, `simulation_time`, `initial_state`, `divergence_angle`, `disturbances` | `disturbances` is the **true** world noise, *not* the filter's `W`/`V` |
| `objective` | `Mp_desired`, `Ts_desired`, `w1`, `w2`, `penalty?` | the `notation.md` §6 cost; desired values are denominators (> 0) |
| `search_space` | `lower_bounds`, `upper_bounds`, `log_scale?` | length-11 vectors in **log-space** (see below) |
| `gp` | `kernel`, `signal_variance`, `noise_variance`, `lengthscales?` | `kernel ∈ {matern52, squared_exponential}` |
| `acquisition` | `kind`, `params` | `kind ∈ {entropy_search, expected_improvement, ucb}`; `params` is forwarded verbatim |
| `optimization` | `n_initial`, `n_iterations`, `n_rollouts_per_eval`, `optimize_hyperparameters` | the §5 budget |

### The decision vector and its log-space coordinates

The optimiser tunes `θ = vec(Q, R, W, V)`, packed by `LQGConfig.to_vector` in a
fixed order and in **log-space** (every weight is positive, so its logarithm is
the natural unconstrained coordinate):

```
θ = [ log diag(Q)  (4) | log diag(R)  (1) | log diag(W)  (4) | log diag(V)  (2) ]   # length 11
```

So `search_space.lower_bounds` / `upper_bounds` are the natural logs of the
weight limits, and `log_scale` defaults to all-`True`. For the 4-state plant
with two sensors the dimension is fixed at `d = 2·n_x + n_u + n_y = 11`.

### True noise vs assumed noise — the distinction that matters

`simulation.disturbances` is what the **world does** (the noise actually
injected into the rollout). The `W`/`V` covariances the Kalman filter assumes
are part of `θ` and are **searched** by the optimiser. They are deliberately
separate: if the true noise were taken from the candidate config, the optimiser
could "cheat" by proposing a noiseless world. See `simulation/disturbances.py`.

---

## 4. What a run writes

`save_results` (via `io.run_logging.save_run`) writes the AGENTS §7
reproducibility bundle into the run directory:

| File | Contents |
|---|---|
| `metadata.json` | git commit hash, seed, UTC timestamp, the full config, the metrics, and a summary (best `θ`, best `Q/R/W/V` diagonals) |
| `trajectories.npz` | the representative rollout: `time`, `true_states`, `estimated_states`, `controls`, `measurements`, `seed`, `diverged` |
| `history.npz` | the BO dataset: `X` (evaluated `θ`) and `y` (their costs) |
| `trajectory.png` | (with `plots=True`) true vs estimated states and the control history |
| `convergence.png` | (with `plots=True`) observed cost and running best per evaluation |

Everything is JSON or NumPy `.npz`, so a run reloads with the standard library
and NumPy alone — no project code required to inspect results:

```python
import json, numpy as np
meta = json.load(open("results/run01/metadata.json"))
traj = np.load("results/run01/trajectories.npz")
```

The `metrics` dict records, for the reported optimum's representative rollout:
`objective`, `overshoot`, `settling_time`, `control_effort`,
`oscillation_energy`, `trajectory_entropy`, `diverged`, plus the search summary
`best_observed_cost` and `n_evaluations`.

---

## 5. Reproducibility

A run is a pure function of `(config, seed)` (AGENTS §7). `ExperimentManager.run`
derives a single `numpy.random.Generator` from `config.seed` and threads it
through the entire optimisation — the Latin-hypercube initial design, every
rollout's noise, the acquisition's candidate and function-sample draws, and the
ML-II restarts. Re-running the same config reproduces the same `history_y` and
the same `best_config`, and `metadata.json` records the git hash so the exact
code revision is known. (`current_git_hash` degrades to `"unknown"` outside a
git work tree rather than failing.)

---

## 6. Choosing the acquisition

- **`entropy_search`** — the project goal (`optimization.md` §4). Maintains a
  belief over the *location* of the optimum and queries where an observation is
  expected to collapse that belief most. Parameters: `n_optimum_samples`,
  `n_representers`, `n_fantasies`, `n_candidates`. The most informative per
  query and the most expensive.
- **`expected_improvement`** — the classic value-greedy baseline; cheap and
  effective. Parameter: `n_candidates`.
- **`ucb`** — GP lower-confidence-bound (named `UpperConfidenceBound` per the
  class diagram); the optimism baseline. Parameters: `beta`, `n_candidates`.

The reported answer is always the **posterior-mean minimiser** (`argmin μ_T`),
not the lowest observation, since under noise the lowest observation may be a
lucky draw (`optimization.md` §5).

---

## 7. Extending

- **A new acquisition or kernel** — add the class under `optimization/` and
  register it in `ExperimentManager._ACQUISITIONS` / `_KERNELS`; it becomes
  selectable from the config `kind`/`kernel` field immediately.
- **A different plant or sensor set** — change the `plant` section and the
  `search_space` length (`d = 2·n_x + n_u + n_y`). The manager validates the
  packed dimension against the engine.
- **Averaging out noise** — raise `optimization.n_rollouts_per_eval` so each
  oracle query averages several seeded rollouts (the CLT noise reduction of
  `optimization.md` §1).

---

## 8. A note on layering

The implementation honours the rule that keeps the optimiser swappable
(`dependency_rules.md` §4): the optimiser reaches the physical/control/
estimation stack **only** through `engine.run(config) → objective.evaluate(result)`.
A controller-design failure (an unstabilising gain, or a non-convergent Riccati
sweep) is caught inside `SimulationEngine.run` and returned as a `diverged`
result, which the objective scores as the finite penalty — so the optimiser
never imports a control/estimation exception and never crashes on a bad
candidate.
