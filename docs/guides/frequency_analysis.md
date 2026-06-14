# Frequency Analysis — From State Space to the Bode Diagram

This document explains, with the theory, how the project produces **Bode
diagrams** of a linear model. It follows exactly the route requested: start
from the **state-space** form, pass to the **frequency** form through the
**Laplace transform**, and use that form to plot the diagram. The code lives in
`dynamics/frequency_response.py` (the computation) and `io/plotting.py`
(the drawing); the entry point is `scripts/frequency_analysis.py`.

---

## 1. What a Bode diagram is

A Bode diagram describes how a linear system responds to a sinusoidal input of
angular frequency $\omega$, in **steady state**: it plots the **gain** (output
amplitude / input amplitude) and the **phase shift** as functions of $\omega$,
with $\omega$ on a logarithmic axis and the gain in **decibels**. It is the
single most informative picture of a linear system's behaviour across
frequencies — where it amplifies, where it attenuates, and how much it delays.

---

## 2. State-space form

The linearised plant is given in state-space form (`physical/pendulum.py`,
`docs/theory/model.md`):

$$
\dot{\mathbf{x}}(t) = A\,\mathbf{x}(t) + B\,\mathbf{u}(t), \qquad
\mathbf{y}(t) = C\,\mathbf{x}(t) + D\,\mathbf{u}(t),
$$

with $\mathbf{x}\in\mathbb{R}^{n_x}$, input $\mathbf{u}\in\mathbb{R}^{n_u}$,
output $\mathbf{y}\in\mathbb{R}^{n_y}$. For our pendulum
$\mathbf{x} = [\theta_p,\dot\theta_p,\theta_w,\dot\theta_w]$, $n_x=4$, the input
is the motor voltage ($n_u=1$), and the sensors are $[\theta_p,\dot\theta_w]$
($n_y=2$).

This is a description **in time**. To see the frequency behaviour we change
representation.

---

## 3. Passing to the frequency form via Laplace

Take the Laplace transform of both equations, with **zero initial conditions**
(we study the response to the input, not to an initial state). Using
$\mathcal{L}\{\dot{\mathbf{x}}\}(s) = s\,\mathbf{X}(s)$:

$$
s\,\mathbf{X}(s) = A\,\mathbf{X}(s) + B\,\mathbf{U}(s), \qquad
\mathbf{Y}(s) = C\,\mathbf{X}(s) + D\,\mathbf{U}(s).
$$

Solve the first equation for $\mathbf{X}(s)$. Group the state terms,

$$
(sI - A)\,\mathbf{X}(s) = B\,\mathbf{U}(s)
\;\;\Longrightarrow\;\;
\mathbf{X}(s) = (sI - A)^{-1} B\,\mathbf{U}(s),
$$

and substitute into the output equation:

$$
\mathbf{Y}(s) = \big[\,C\,(sI - A)^{-1} B + D\,\big]\,\mathbf{U}(s).
$$

The bracket is the **transfer matrix** — the frequency-form description of the
system:

$$
\boxed{\,G(s) = C\,(sI - A)^{-1} B + D\,}
$$

$G(s)$ is $n_y \times n_u$: entry $G_{ij}(s)$ is the transfer function from input
$j$ to output $i$. Its **poles** are the values of $s$ where $sI-A$ is singular —
i.e. the **eigenvalues of $A$** (the system's natural modes), up to any
pole–zero cancellations with $C$ and $B$.

---

## 4. The frequency response: $s = j\omega$

A stable LTI system driven by $u(t)=\sin(\omega t)$ settles to a sinusoid of the
same frequency, scaled by $|G(j\omega)|$ and shifted by $\angle G(j\omega)$.
That is the content of evaluating the transfer function on the **imaginary
axis**, $s = j\omega$: $G(j\omega)$ is the system's *frequency response*. The
Bode diagram plots its two parts,

$$
\text{magnitude}(\omega) = 20\,\log_{10}\big|G(j\omega)\big| \ \text{[dB]},
\qquad
\text{phase}(\omega) = \angle\,G(j\omega) \ \text{[deg]},
$$

against $\omega$ on a logarithmic axis. The decibel and log-frequency scaling
turn products of factors into sums and power-law roll-offs into straight lines —
which is what makes Bode diagrams so readable.

**Discrete-time models.** A discrete model lives on the unit circle instead of
the imaginary axis: the frequency response is $G(e^{j\omega \Delta t})$, i.e.
substitute $s \to z = e^{j\omega\Delta t}$ in $G(z)=C(zI-A)^{-1}B+D$. The code
handles both; for `is_discrete` models it uses the unit-circle point.

---

## 5. How the code computes it

`dynamics/frequency_response.py`:

- `transfer_function(model, omega)` evaluates $G$ at each requested $\omega$ and
  returns a complex array of shape `(n_omega, n_y, n_u)`.
- `bode(model, omega, output_index, input_index)` reduces one channel to
  magnitude (dB) and unwrapped phase (deg) — a `BodeData` record.
- `log_frequencies(omega_min, omega_max, n)` builds a log-spaced sweep.

For each frequency the core step is forming $(sI - A)^{-1}B$, i.e. solving the
linear system

$$
(sI - A)\,X = B, \qquad G = C\,X + D .
$$

### Staying inside the from-scratch numerics

With $s = j\omega$ this is a **complex** linear system, but the project's
hand-written linear algebra is real (`numerics/linalg.py`), and `numpy.linalg`
/ SciPy solvers are banned in `src/` (`AGENTS.md` §3). We therefore use the
standard **complex-to-real embedding**. Writing $s = \sigma + j\omega$,
$X = X_r + jX_i$, and splitting $(sI-A)X=B$ into real and imaginary parts gives
the real $2n_x \times 2n_x$ system

$$
\begin{bmatrix} \sigma I - A & -\omega I \\[2pt] \omega I & \sigma I - A \end{bmatrix}
\begin{bmatrix} X_r \\[2pt] X_i \end{bmatrix}
=
\begin{bmatrix} B \\[2pt] 0 \end{bmatrix},
$$

which the project's real partial-pivot LU (`numerics.linalg.lu_solve_matrix`)
solves directly; then $X = X_r + jX_i$. (For $s=j\omega$, $\sigma=0$.) This is
exact — the validation tests below match a reference complex inverse to machine
precision — and it keeps the whole pipeline on the from-scratch solver.

The magnitude and phase are then plain element-wise operations
($|\cdot|$, $\angle$, $\log_{10}$), and the phase is **unwrapped** so the curve
is continuous. Plotting (`io.plotting.plot_bode`) draws the two stacked traces;
the computation layer never imports matplotlib.

---

## 6. Reading the pendulum's Bode diagram

Running `scripts/frequency_analysis.py` on the default plant linearises it and
draws one Bode diagram per output channel. The eigenvalues of $A$ — the poles of
$G$ — are approximately

$$
\{\,0,\; +4.45,\; -4.93,\; -13.58\,\}\ \text{rad/s},
$$

and they explain everything in the plots:

- **The $+4.45$ pole has positive real part — the plant is open-loop
  unstable.** This is correct: it is the pendulum *wanting to fall*
  (`model.md`'s sanity check). Two consequences for the Bode reading:
  - the voltage → $\theta_p$ magnitude **peaks near $\omega \approx 4$ rad/s**,
    the natural frequency of the inverted-pendulum mode, and rolls off on either
    side;
  - because the system is unstable, $G(j\omega)$ here is the **analytic**
    frequency response (the transfer function evaluated on the imaginary axis),
    *not* a physically measurable steady-state response — an unstable system has
    no bounded sinusoidal steady state. The diagram is still the right object
    for design (it is what loop-shaping and the LQG analysis use), but it is not
    something you could measure open-loop on the bench.

- **The $0$ eigenvalue is the wheel-angle integrator** ($\theta_w$ does not
  appear in the dynamics — column 3 of $A$ is zero). It makes $sI-A$ singular at
  $s=0$, so the routine cannot form $G$ exactly at $\omega=0$ (it raises rather
  than divide by a singular matrix); we sweep $\omega>0$. The sensed outputs
  ($\theta_p$, $\dot\theta_w$) do not expose that integrator directly, so their
  channel magnitudes flatten to finite values as $\omega\to 0$ rather than
  blowing up.

- **The phase** of the $\theta_p$ channel starts near $+90^\circ$ at low
  frequency and decreases through the resonance — the signature of the
  pendulum mode.

So the Bode diagram is a direct, visual restatement of the same facts the
state-space eigen-analysis gives (`numerics.spectral_radius` /
`numerics.eigvals`): an unstable pendulum mode near $4$ rad/s and a wheel-angle
integrator. The two views are consistent because they are the same model in two
representations.

---

## 7. The expert's plot: open-loop gain and stability margins

The Bode diagrams above are of the **plant** $G$. The more telling plot for a
control engineer is the **open-loop gain of the broken loop**, $L = G K$, where
$K$ is the LQG controller — because its shape near the critical point gives the
**stability margins**: how much delay or gain change the real hardware can
absorb before the closed loop goes unstable. (`scripts/stability_margins.py`,
`dynamics.frequency_response.loop_transfer_function` / `stability_margins`.)

### Building the LQG loop gain

The LQG controller is the Kalman observer plus the LQR law $u = -K\hat{\mathbf{x}}$.
Realised (current-estimator, predictor state) as a system from measurement to
control,

$$
A_c = (A_d - B_d K)(I - LC), \quad B_c = (A_d - B_d K)L, \quad
C_c = -K(I - LC), \quad D_c = -KL,
$$

its transfer function $K_c(z)$ composes with the plant into the scalar loop gain,
broken at the (single) plant input,

$$
L(z) = -K_c(z)\,G(z), \qquad z = e^{j\omega\Delta t}.
$$

The leading minus puts it in the standard negative-feedback convention (critical
point $-1$), since the controller already carries the $-K$. This is a
**discrete-time** loop — the controller realised here is exactly the one the
simulator runs (predict → update → $u=-K\hat{\mathbf{x}}$), so the margins are
the margins of the *implemented* controller, sampling lag included.

**Why a reduced model.** The wheel angle is a decoupled integrator —
uncontrollable from the input *and* unobservable from the sensors — so a
steady-state Kalman filter does not exist for the full plant (it is
undetectable). It is dropped, leaving the controllable + observable
$[\theta_p,\dot\theta_p,\dot\theta_w]$ model on which the LQG, and hence the loop
gain, is well-defined. The hidden mode cancels in $L$ anyway, so this changes
nothing about the margins.

### The two margins

- **Phase margin (PM)** $= 180^\circ + \angle L(j\omega_{gc})$ at the gain
  crossover $|L|=1$: the extra phase lag (e.g. microcontroller latency) the loop
  tolerates before instability. Want $\gtrsim 30\text{–}45^\circ$.
- **Gain margin (GM)** $= -20\log_{10}|L(j\omega_{pc})|$ at the phase crossover
  $\angle L = -180^\circ$: how much the loop gain may change (a weaker $K_t$, a
  drooping battery) before instability. Want $\gtrsim 6$ dB.

### What the pendulum's LQG loop shows — and Doyle's warning

For the sensible-pendulum design the loop gain reports **PM $\approx 7^\circ$**
and **GM $\approx 3.3$ dB** — *both well below* the rules of thumb. The loop is
stabilising but **fragile**: a few degrees of unmodelled phase lag, or a ~45 %
gain change, would destabilise it. This is the textbook lesson of Doyle's 1978
result, *"Guaranteed Margins for LQG Regulators: None"*: LQR alone has
excellent guaranteed margins ($\geq 6$ dB, $\geq 60^\circ$) and the Kalman filter
is robust too, but **their combination carries no such guarantee**. Plotting
$L = GK$ for the *full* LQG is the only way to find out — which is exactly why an
expert runs it, and a reason to fold a margin constraint into the tuning
objective.

---

## 8. Validation

`tests/validation/` cross-checks the hand-written code against the banned
reference libraries (allowed only there):

- the transfer function against a `numpy.linalg.inv` reference
  $C(j\omega I - A)^{-1}B + D$, continuous and discrete — they agree to
  ~$10^{-15}$ (the embedding solve is exact) — and against `scipy.signal.bode`
  for a SISO channel;
- the loop gain $L = -K_c G$ against `control.frequency_response` of the same
  controller·plant series — again to machine precision;
- the margin extractor against `control.margin` on analytic continuous loops and
  for the LQG **phase** margin. Note the from-scratch, grid-based extractor
  finds a genuine $-180^\circ$ phase crossing of the *discrete* loop (a real
  ~3.3 dB gain margin) that `control.margin` **misses** — a small vindication of
  computing it directly and verifying on a dense grid.

---

## 9. Using it

Command line:

```bash
PYTHONPATH=src python scripts/frequency_analysis.py configs/default.yaml -o results/bode
PYTHONPATH=src python scripts/stability_margins.py  configs/pendulum_sensible.yaml -o results/margins
```

Programmatically:

```python
import numpy as np
from inverted_pendulum.dynamics.frequency_response import bode, log_frequencies
from inverted_pendulum.experiment.manager import ExperimentManager

model = ExperimentManager.from_config_file("configs/default.yaml") \
            .build_engine().linear_model.continuous   # the (A, B, C, D) form
omega = log_frequencies(1e-2, 1e3, 600)
curve = bode(model, omega, output_index=0, input_index=0)  # voltage → θ_p
# curve.magnitude_db, curve.phase_deg  →  io.plotting.plot_bode
```

The same `transfer_function` / `bode` work on **any** `StateSpaceModel` — the
open-loop plant, its discretisation, or a closed-loop model — so the frequency
view is available wherever a linear model is.
