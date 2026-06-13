# Time-Domain Response — Step Response, Transients, and Modes

This document explains, with the theory, how the project analyses a linear model
**in time**: the step response, the initial-condition (regulation) transient,
the standard transient metrics, and the modal (pole / damping) reading. It is
the companion to [frequency_analysis.md](frequency_analysis.md) — the same model,
seen in the time domain instead of the frequency domain. The code is in
`dynamics/time_response.py`, drawn by `io/plotting.py`, and exercised by
`scripts/step_response.py`.

---

## 1. Regulation vs tracking — what "step response" means here

A *tracking* controller is asked to follow a reference; its step response is the
output's reaction to a step change in that reference. Our system is different:
it is a **regulator** that drives the pendulum back to the upright equilibrium
($\mathbf{x}=0$). It has no reference to track — the "command" is simply *be
upright*.

For a regulator the meaningful time-domain test is therefore the
**initial-condition response**: knock the state to some $\mathbf{x}(0)$ (a tilt)
and watch it return to zero. This is a step *in the initial condition*, and the
classical step-response characteristics — overshoot, peak time, settling time —
apply to it directly. (It is exactly the transient on which the cost metrics
$M_p$ and $T_s$ in `metrics/` are defined.) We also provide the genuine
**unit-step input response** for any model, which is the standard tool for the
stable building blocks.

A second essential point: the **open-loop plant is unstable** (the pendulum
falls), so its open-loop step response diverges and is not useful. Every
meaningful time response here is of the **closed loop**.

---

## 2. The state-space solution in time

For the continuous system $\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$,
$\mathbf{y} = C\mathbf{x} + D\mathbf{u}$, linear-systems theory gives the exact
solution

$$
\mathbf{x}(t) = e^{A t}\,\mathbf{x}(0)
            \;+\; \int_{0}^{t} e^{A(t-\tau)} B\,\mathbf{u}(\tau)\,d\tau .
$$

Two special cases are all we need.

### Free (initial-condition) response — $\mathbf{u}\equiv 0$

$$
\mathbf{x}(t) = e^{A t}\,\mathbf{x}(0),
$$

the **matrix exponential** acting on the initial state. The code computes it
*exactly* at the sample times by forming $\Phi = e^{A\,\Delta t}$ once (with the
from-scratch `numerics.matrix_exp.expm` — the same primitive used for
discretisation) and propagating $\mathbf{x}_{k+1} = \Phi\,\mathbf{x}_k$. There is
no integrator truncation error: this is the analytic solution sampled.

### Step response — unit step from rest

For a constant (zero-order-hold) input the convolution integral collapses to the
discrete recursion on the ZOH-discretised model $(A_d, B_d)$:

$$
\mathbf{x}_{k+1} = A_d\,\mathbf{x}_k + B_d\,\mathbf{u}, \qquad
\mathbf{y}_k = C\,\mathbf{x}_k + D\,\mathbf{u}, \qquad \mathbf{x}_0 = 0,
$$

with $\mathbf{u}$ the unit step. This is precisely the standard step response of
the transfer function, computed without leaving the project's own numerics.

### The closed loop

Substituting the state-feedback law $\mathbf{u} = -K\mathbf{x}$ into the dynamics
gives the autonomous closed loop

$$
\dot{\mathbf{x}} = (A - BK)\,\mathbf{x} \equiv A_{\mathrm{cl}}\,\mathbf{x},
$$

whose free response is the regulation transient. `closed_loop(model, K)` forms
$A_{\mathrm{cl}}$ (it takes the raw gain array, so `dynamics` need not import
`control`).

---

## 3. Transient metrics (Ogata §5-4)

For a response moving from an initial value to a steady-state `final`,
`transient_metrics` reports:

| Metric | Definition |
|---|---|
| **rise time** | time to go from 10 % to 90 % of the total step |
| **peak time** | time of the largest excursion past `final` |
| **percent overshoot** | $100 \cdot (\text{peak excursion past final}) / |\text{step}|$ |
| **settling time** | last time the signal leaves the $\pm$2 % band around `final` |
| **steady state** | the `final` value |

For the canonical second-order system these have closed forms in the damping
ratio $\zeta$ and natural frequency $\omega_n$ — used as exact checks in the
tests:

$$
\%\text{OS} = 100\,e^{-\zeta\pi/\sqrt{1-\zeta^2}},
\qquad
t_{\text{peak}} = \frac{\pi}{\omega_n\sqrt{1-\zeta^2}} .
$$

Pass `final = 0` for a regulation transient (decay to upright); pass
`initial = 0` for a unit step from rest.

---

## 4. Modal analysis — poles as $\omega_n$ and $\zeta$

The eigenvalues of $A$ are the system's poles, the same ones the Bode diagram
shows. `modal_analysis` recasts each pole $\lambda$ into its time-domain reading:

$$
\omega_n = |\lambda|, \qquad \zeta = -\frac{\operatorname{Re}(\lambda)}{|\lambda|}.
$$

So a complex pair $\lambda = -\zeta\omega_n \pm j\omega_n\sqrt{1-\zeta^2}$ gives an
oscillatory mode of frequency $\omega_n$ and damping $\zeta$; a **negative** $\zeta$
flags an **unstable** mode (a pole in the right half-plane); $\zeta \ge 1$ is an
overdamped real pole; $\omega_n = 0$ is an integrator. For a discrete model the
poles are first mapped to continuous-equivalents via $\lambda_c = \ln(\lambda_d)/\Delta t$.

---

## 5. Reading the pendulum's closed-loop response

`scripts/step_response.py` designs an LQR for the default plant and analyses the
closed loop. With $Q = \operatorname{diag}(20, 2, 10^{-2}, 10^{-2})$, $R = 1$ it
reports four **real, stable** closed-loop poles,

$$
\{-0.88,\; -4.36,\; -4.91,\; -25.78\}\ \text{rad/s},
$$

all with $\zeta = 1$ (overdamped) — LQR has placed a well-damped loop, the
slowest mode ($\omega_n \approx 0.88$ rad/s) being the wheel settling back.

The regulation transient from a $0.05$ rad tilt:

- **overshoot $\approx 33\%$, peak at $\approx 0.37$ s, settling $\approx 2.2$ s.**
  The overshoot may look surprising for an all-real-pole (non-oscillatory) loop,
  but it is real and physical: this is a **multi-state initial-condition
  response**, not a textbook second-order step. The reaction wheel spins up to
  arrest the fall and, as it spins back down, it carries the pendulum angle
  slightly *past* upright to the other side — a single zero-crossing "overshoot"
  with no ringing. The metric correctly captures it.
- **the linear and nonlinear responses coincide** to about $10^{-3}$ rad over the
  whole transient. The linear curve is $e^{A_{\mathrm{cl}} t}\mathbf{x}_0$; the
  nonlinear curve is the true $\sin\theta$ plant under the same controller. Their
  agreement at $0.05$ rad is a direct, quantitative confirmation that the
  small-angle linearisation the whole design rests on is valid in the operating
  regime.

This is the same model and the same poles as the Bode analysis, now read in
time: stability, speed, and the validity of the linearisation, all visible at a
glance.

---

## 6. Validation

`tests/validation/test_time_response_validation.py` cross-checks against
`scipy.signal` (allowed only in `tests/validation/`):

- `step_response` against `scipy.signal.step` on a damped second-order system;
- `free_response` against `scipy.signal.lsim` (zero input, with $\mathbf{x}_0$);
- the propagated states against `scipy.linalg.expm` directly.

The unit tests additionally pin the analytic second-order overshoot/peak-time
formulas and the first-order $1 - e^{-t/\tau}$ step.

---

## 7. Using it

```bash
PYTHONPATH=src python scripts/step_response.py configs/default.yaml -o results/time_response
```

```python
import numpy as np
from inverted_pendulum.dynamics.time_response import (
    closed_loop, free_response, step_response, transient_metrics, modal_analysis,
)
# given a continuous StateSpaceModel `model` and an LQR gain `K`:
cl = closed_loop(model, K)
modes = modal_analysis(cl)                                  # poles → ω_n, ζ
t, states, _ = free_response(cl, [0.05, 0, 0, 0], t_end=5.0, dt=0.01)
metrics = transient_metrics(t, states[:, 0], final=0.0)    # overshoot, settling…
t_step, y_step = step_response(cl, t_end=5.0, dt=0.01)      # unit-step response
```

`free_response`, `step_response`, `modal_analysis` work on **any**
`StateSpaceModel`, so the time-domain view — like the frequency-domain one — is
available wherever a linear model is.
