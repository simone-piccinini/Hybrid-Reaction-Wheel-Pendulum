# Implementation Brief — Reaction-Wheel Pendulum Plant Model

## Goal

Produce the linear state-space matrices $(A, B)$ that describe the pendulum near its upright equilibrium. Everything downstream — LQR gain, Kalman filter, simulation — consumes these matrices. This brief specifies the model to build; it omits the Lagrangian derivation that produced it.

## Where it comes from (the one-paragraph intuition)

The plant is a pendulum free to swing, with a reaction wheel mounted on it driven by a brushless motor (modelled as a DC motor). The motor cannot push against the ground — it spins the wheel, and the wheel's angular acceleration produces an equal-and-opposite reaction torque on the pendulum body. That reaction torque is the only control authority. The full dynamics are nonlinear (gravity enters as $\sin\theta$), so we **linearise about the upright equilibrium** $\theta = 0$ using $\sin\theta \approx \theta$, valid for small angles. This is exactly the regime where LQR is optimal, and balancing keeps the system there.

## The state, input, and output

$$

\mathbf{x} = \begin{bmatrix} \theta \\ \dot{\theta} \\ \phi \\ \dot{\phi} \end{bmatrix}, \qquad u = V_a \ (\text{motor voltage}), \qquad \dot{\mathbf{x}} = A\mathbf{x} + Bu.

$$

- $\theta$ — pendulum angle (upright is $\theta = 0$, the equilibrium to stabilise).
- $\dot\theta$ — pendulum angular rate.
- $\phi$ — wheel angle relative to the pendulum.
- $\dot\phi$ — wheel angular rate.
- $u$ — the single control input: voltage applied to the motor. The system is single-input.

Note the wheel angle $\phi$ itself does not appear in the dynamics (column 3 of $A$ is zero) — only its rate matters. It is kept in the state because the estimator/controller track it, but it is uncontrollable and that is expected.

## Parameters the model needs

| Symbol | Meaning |
|---|---|
| $I_p$ | moment of inertia of the pendulum |
| $I_w$ | moment of inertia of the wheel |
| $b_p$ | pendulum (pivot) friction coefficient |
| $b_w$ | wheel friction coefficient |
| $m$ | mass of the system |
| $g$ | gravitational acceleration |
| $l$ | distance from pivot to centre of mass |
| $K_t$ | motor torque constant |
| $K_e$ | motor back-EMF constant |
| $R_a$ | motor armature resistance |

These are physical constants supplied by config. Compute two lumped scalars from them first:

$$

E = \frac{K_t}{R_a}, \qquad D = \frac{K_t K_e}{R_a} + b_w.

$$

$E$ converts voltage to motor torque; $D$ lumps the motor's back-EMF damping together with wheel friction.

## The matrices to build

$$

A = \begin{bmatrix}
0 & 1 & 0 & 0 \\[4pt]
\dfrac{mgl}{I_p} & -\dfrac{b_p}{I_p} & 0 & \dfrac{D}{I_p} \\[8pt]
0 & 0 & 0 & 1 \\[4pt]
-\dfrac{mgl}{I_p} & \dfrac{b_p}{I_p} & 0 & -\dfrac{D(I_w + I_p)}{I_w I_p}
\end{bmatrix}, \qquad
B = \begin{bmatrix}
0 \\[4pt]
-\dfrac{E}{I_p} \\[8pt]
0 \\[4pt]
\dfrac{E(I_w + I_p)}{I_w I_p}
\end{bmatrix}.

$$

Rows 1 and 3 are the trivial kinematic relations $\dot{x}_1 = x_2$ and $\dot{x}_3 = x_4$. Rows 2 and 4 are the linearised equations of motion for the pendulum and wheel respectively. The opposite signs between rows 2 and 4 (e.g. $+mgl/I_p$ vs $-mgl/I_p$, $-E/I_p$ vs $+E(\cdots)$) encode the action–reaction coupling: torque that accelerates the wheel decelerates the pendulum.

## Measurement model (for the Kalman filter)

The dynamics give $A, B$ in continuous time. The estimator also needs a measurement matrix $C$ such that $\mathbf{y} = C\mathbf{x} + \mathbf{v}$. Set $C$ to select whatever the hardware actually senses (commonly $\theta$ and $\dot\phi$, or an IMU-derived angle plus a wheel encoder rate); pull the exact sensor layout from config. $C$ must map the state into the sensor's real units and sign convention.

## Implementation order

1. **Load parameters** from config; validate all inertias and $R_a$ are nonzero (they appear in denominators).
2. **Compute the lumped scalars** $E$ and $D$.
3. **Assemble $A$ and $B$** exactly as above (continuous-time, linearised at $\theta = 0$).
4. **Build $C$** from the sensor configuration.
5. **Discretise** $(A, B)$ to $(A_d, B_d)$ at the control sampling period (via the matrix exponential primitive in the numerics layer) — the LQR and Kalman recursions run in discrete time.
6. **Hand off** $(A_d, B_d, C)$ to the control and estimation layers.

## Practical cautions

- **This model is only valid near $\theta = 0$.** It is a small-angle approximation; it does not describe swing-up from hanging-down. The agent stabilises an already-near-upright pendulum.
- **Sanity-check before trusting it.** The open-loop $A$ should be unstable (an eigenvalue with positive real part) — that is the pendulum wanting to fall, and it is correct. A fully stable $A$ means a sign or assembly error.
- **Confirm controllability** of $(A, B)$ before computing an LQR gain; the wheel-angle state being uncontrollable is fine, but the angle/rate states must be reachable through the motor.
- **Units must be consistent** (SI throughout) across all parameters, or the matrices will be silently wrong.
- **Keep continuous and discrete matrices distinct.** $(A, B)$ from this brief are continuous; the control loop uses the discretised $(A_d, B_d)$. Do not mix them.