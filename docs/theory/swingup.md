# Implementation Brief — Energy-Shaping Swing-Up and the LQG Hand-Off

## Goal

Bring the pendulum from **hanging** ($\theta_p = \pi$) up into the small-angle
basin around **upright** ($\theta_p = 0$), then hand control to the balancing LQG
that holds it there. This is the one maneuver the rest of the project does *not*
cover: everything else (LQR, Kalman, the whole Bayesian-optimisation loop) lives
in the small-angle linearisation about upright ([model.md](model.md)), which is
meaningless once the pendulum is far from the top. Swing-up is a **global,
nonlinear** controller that runs on the *true* `sin θ` plant
([`dynamics/nonlinear_model.py`](../../src/inverted_pendulum/dynamics/nonlinear_model.py))
and only needs to deliver the state into the LQR's region of attraction — after
which [lqr.md](lqr.md) + [kalman.md](kalman.md) take over.

Code: [`control/swingup_controller.py`](../../src/inverted_pendulum/control/swingup_controller.py)
(the law) and [`scripts/swing_up.py`](../../scripts/swing_up.py) (the full
two-phase rollout, the switch, and the feasibility check).

## Where it comes from (the one-paragraph intuition)

You cannot muscle a torque-limited pendulum straight up — on this build the wheel
can produce only a fraction of the gravity torque at the horizontal (see
"Feasibility"). Instead you control its **energy**. The pendulum has a total
mechanical energy that is lowest hanging and highest upright; if you can drive
that energy to exactly the upright level while keeping the motion on the right
orbit, the pendulum arrives at the top with almost no speed — caught, not thrown.
Energy is pumped a little at a time by pushing the wheel *in the direction the
pendulum is already moving*, so each swing is bigger than the last (resonant
pumping), until the energy matches the top and the supervisor hands off to the
balancer. This is the Åström–Furuta energy-control method (Åström & Furuta,
"Swinging up a pendulum by energy control", *Automatica* 2000), adapted to the
reaction wheel.

## What it consumes

The plant scalars (SI; [notation.md](notation.md) §2) — pendulum mass $m$, length
$\ell$, gravity $g$, body inertia $I_b$, the motor's voltage→torque gain
$E = K_t/R_a$, and the rail $V_{\max}$ — plus one tuning gain $k>0$. It reads the
pendulum angle $\theta_p$ and rate $\dot\theta_p$ (directly sensed / differentiated;
no Kalman filter, which is a small-angle estimator and invalid here). The output
is a motor voltage. The balancing LQG it hands off to consumes the usual
$(A_d, B_d, C)$ and the tuned $(Q,R,W,V)$.

## The equations to implement

Measure $\theta_p$ from upright, so the pendulum's mechanical energy (the body;
the wheel torque is the *control*, not part of the energy) is

$$
E(\theta_p, \dot\theta_p) = \tfrac12 I_b\,\dot\theta_p^{\,2} + m g \ell \cos\theta_p,
$$

maximal at upright ($E_{\text{up}} = m g \ell$, at $\theta_p = 0,\ \dot\theta_p = 0$)
and minimal hanging ($-m g \ell$). Write the **energy error**
$\tilde E = E - E_{\text{up}}$ (so $\tilde E \le 0$ everywhere below the top).

The pendulum's equation of motion
([pendulum.py](../../src/inverted_pendulum/physical/pendulum.py):
$I_b\ddot\theta_p = m g \ell \sin\theta_p - b_p\dot\theta_p - \tau_w$, with $\tau_w$
the net wheel torque on the body) gives the rate of change of the energy error:

$$
\frac{d\tilde E}{dt}
  = I_b\,\dot\theta_p\ddot\theta_p - m g \ell \sin\theta_p\,\dot\theta_p
  = -\,b_p\,\dot\theta_p^{\,2} \;-\; \tau_w\,\dot\theta_p .
$$

The control is the wheel torque $\tau_w$. Choose it proportional to the energy
error along the motion,

$$
\boxed{\;\tau_w = k\,\tilde E\,\dot\theta_p\;}, \qquad k > 0,
$$

so that

$$
\frac{d\tilde E}{dt} = -\,b_p\,\dot\theta_p^{\,2} \;-\; k\,\tilde E\,\dot\theta_p^{\,2}.
$$

The second term drives $\tilde E \to 0$ for **any** $k>0$: when $\tilde E < 0$
(below the top) it is $+k|\tilde E|\dot\theta_p^{\,2} \ge 0$, adding energy every
swing regardless of the direction of motion; the first term is the friction
drain. Finally invert the (unsaturated) voltage→torque map $\tau_w \approx E\,V$
to a **command**, and clamp it to the rail:

$$
\boxed{\;V = \mathrm{sat}_{V_{\max}}\!\Big(\frac{k\,\tilde E\,\dot\theta_p}{E}\Big)\;}
\qquad E = \frac{K_t}{R_a}.
$$

Two fixed points fall out correctly: $\tilde E = 0$ (already at the upright energy
— on the homoclinic orbit) and $\dot\theta_p = 0$ (instantaneously at rest)
both command $V = 0$. The reaction torque on the body is $-E\,V = -k\tilde E\dot\theta_p$,
which for $\tilde E<0$ has the sign of $\dot\theta_p$ — it always pushes *with* the
motion, which is exactly "pump energy in."

## The hand-off — a switching supervisor

Swing-up only has to reach the LQR's basin; the balancer does the rest. A latching
supervisor watches the state and switches **once**:

$$
\text{switch to balance when}\quad
\lvert \operatorname{wrap}(\theta_p)\rvert < \theta_{\text{sw}}
\ \text{ and }\ \lvert\dot\theta_p\rvert < \dot\theta_{\text{sw}},
$$

with $\operatorname{wrap}(\cdot)$ mapping the angle to $(-\pi,\pi]$ (the true state
may be near a multiple of $2\pi$). Defaults: $\theta_{\text{sw}} = 0.2$ rad
($\approx 11^\circ$), $\dot\theta_{\text{sw}} = 4$ rad/s. Because energy control
delivers the pendulum to the top *slow* (as $\tilde E \to 0$, $\dot\theta_p \to 0$
at $\theta_p = 0$), the incoming velocity is naturally small and inside the basin.
On the switch, seed the Kalman filter at the current (wrapped) state and run the
tuned LQG ([kalman.md](kalman.md) → [lqr.md](lqr.md)); the estimator's
wheel-angle state is unobservable and irrelevant, so the balancer uses a near-zero
LQR weight on it (its estimate drifts — do not regulate it hard).

## Feasibility — the torque-vs-friction condition

Swing-up is **not always possible**: it is gated by whether the actuator can
out-torque the friction. The peak reaction torque is
$\tau_{\max} = E\,V_{\max}$. Pumping stalls at the bottom speed where this balances
pivot friction, $\dot\theta_{\text{cap}} \approx \tau_{\max}/b_p$. Reaching the top
requires enough energy to climb $2m g \ell$, i.e. a bottom speed

$$
\dot\theta_{\text{need}} = \sqrt{\frac{4 m g \ell}{I_b}}.
$$

So swing-up is feasible only when $\dot\theta_{\text{cap}} > \dot\theta_{\text{need}}$,
i.e. the pivot friction is below a critical value

$$
b_p < b_p^{\text{crit}} = \frac{E\,V_{\max}}{\dot\theta_{\text{need}}}.
$$

For the measured build ([`configs/pendulum_measured.yaml`](../../configs/pendulum_measured.yaml)):
$\tau_{\max} \approx 0.044$ N·m, $\dot\theta_{\text{need}} \approx 16.2$ rad/s, so
$b_p^{\text{crit}} \approx 0.0027$ N·m·s/rad. Equivalently, a free ring-down whose
damping ratio $\zeta = b_p/(2 I_b\omega_n)$ (with $\omega_n=\sqrt{mg\ell/I_b}$)
exceeds $\approx 0.06$ cannot be swung up. The config's **guessed**
`pivot_friction = 0.01` (marked "not measured") gives $\zeta \approx 0.22$ and is
infeasible; a realistic ball-bearing pivot ($b_p \lesssim 10^{-3}$, $\zeta \lesssim
0.02$) swings up in a few seconds. **Measure $b_p$ before trusting swing-up on
hardware** — a free-decay (ring-down) test on the encoder gives it directly.
`scripts/swing_up.py` prints this feasibility line up front.

## Implementation order

1. **Build the law** from the plant scalars and a gain $k$
   ([`EnergySwingUpController`](../../src/inverted_pendulum/control/swingup_controller.py));
   it needs no state and no filter.
2. **Roll out on the nonlinear plant** from hanging $[\pi, 0, 0, 0]$ using the ZOH
   stepper. There is no "diverged" guard on angle here — large angles are the
   normal regime (unlike the balancing simulator).
3. **Each step:** if still swinging up, command $V = \mathrm{sat}(k\tilde E\dot\theta_p/E)$;
   test the switch condition; once inside the basin, latch to balance.
4. **Balance phase:** seed the Kalman filter at the current wrapped state, then run
   measure → filter → $u = -K\hat{\mathbf x}$, saturating to the rail.
5. **Report** the swing-up time, the upright error and rate after the catch, and the
   peak wheel speed during pumping (the momentum the balancer inherits).

## Practical cautions

- **Feasibility first.** Check $b_p < b_p^{\text{crit}}$ before anything else; a stiff
  pivot or a weak motor makes swing-up impossible no matter the gain (see above).
- **The wheel spins up while pumping.** Its speed climbs monotonically-ish and must
  not saturate (back-EMF kills torque) *before* the catch — the demo reaches
  ~1000 RPM. This inherited momentum is what the balancer must then bleed off, and
  it ties to the wheel angle being uncontrollable in steady state
  ([kalman.md](kalman.md), the detectability caveat).
- **Capture basin.** The pendulum arrives moving; if $\dot\theta_{\text{sw}}$ is too
  large the LQR may overshoot back down. Keep the switch window inside the LQR's
  actual region of attraction (tighten $\theta_{\text{sw}}$/$\dot\theta_{\text{sw}}$
  if it fails to settle).
- **The hanging equilibrium cannot self-start.** At $\theta_p=\pi,\ \dot\theta_p=0$
  the law commands $V=0$ and the state is an equilibrium — in practice sensor noise
  or a tiny perturbation breaks the symmetry, but a deterministic sim started
  *exactly* at rest needs a nudge.
- **Bang-bang, by necessity.** With $\tau_{\max}\ll$ the gravity torque, the command
  saturates for most of the swing, so $k$ mostly sets *when* saturation kicks in,
  not the pumped power; do not over-tune it.
- **Friction is not purely viscous.** The model lumps friction into $b_p\dot\theta_p$;
  real bearings are largely Coulomb (a constant $\tau_c$). Coulomb friction is a
  fixed drain rather than a speed cap, so hardware is often *more* forgiving than
  the viscous feasibility bound suggests — measure the ring-down envelope shape to
  tell which dominates.
