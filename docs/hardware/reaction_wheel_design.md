# Designing the reaction wheel

How to size and model the reaction wheel (the disk the motor spins), with the
engineering reasoning behind every dimension. Companion to
[Sizing the pendulum](sizing_the_pendulum.md) and the
[Bill of Materials](bill_of_materials.md); the physical quantities map to
[`model.md`](../theory/model.md) / [`notation.md`](../theory/notation.md) §2.

---

## 1. What the wheel does, and the one number that matters

The pendulum has no way to push against the ground. The **only** control torque
comes from action–reaction: the motor accelerates the wheel, and the wheel
pushes back on the body with an equal-and-opposite torque. The wheel is an
**angular-momentum store**.

Two facts decide the whole design:

1. **The reaction torque on the body equals the motor torque** — *independent of
   the wheel's inertia.* A heavier wheel does **not** give you more push.
2. **What the wheel's inertia `I_w` buys is momentum capacity:**

$$
H_{max} = I_w\,\omega_{max}
$$

   — how much angular momentum the wheel can soak up before it hits its top
   speed `ω_max` and **saturates** (after which it can no longer help and the
   pendulum falls).

So the design driver is the **moment of inertia `I_w`**, sized from a
*momentum* requirement — not the diameter that "looks right."

---

## 2. Requirement → target inertia

To catch a fall from a tilt `θ_c`, the wheel must absorb the angular momentum the
body builds up while falling, `≈ I_b\,\dot\theta_{catch}`, with margin. From the
[sizing analysis](sizing_the_pendulum.md), for a ~15 cm arm with a ~0.11 kg tip
falling through ~12°, that momentum is `≈ 4×10⁻³ kg·m²/s`.

The motor's usable top speed (GBM2804H-100T, no-load ≈ 1550 rpm ≈ 162 rad/s,
derated under load) is `ω_max ≈ 120 rad/s`. Asking for a ~2.5–3× momentum
margin, `H_max ≈ 1.2×10⁻² kg·m²/s`, gives the target:

$$
I_w \;\gtrsim\; \frac{H_{max}}{\omega_{max}}
   = \frac{1.2\times10^{-2}}{120}
   \;\approx\; \mathbf{1\times10^{-4}\ kg\,m^2}.
$$

That single number is the spec the geometry must hit.

---

## 3. The geometry: inertia formulas

These are the shapes you'll sketch in Onshape; the moment of inertia about the
spin axis is:

| Shape | `I` about the axis | Comment |
|---|---|---|
| Solid uniform disk (radius `r`) | `½ m r²` | half the mass is wasted near the centre |
| Thin ring / all mass at `r` | `m r²` | **twice the inertia per gram** of a solid disk |
| Annulus (inner `r₁`, outer `r₂`) | `½ m (r₁² + r₂²)` | the realistic disk-with-hole |
| `N` point masses (bolts) at `r_b` | `N\,m_b\,r_b²` | easy to tune / balance with screws |
| Hub + rim (superpose) | `I_{hub} + I_{rim}` | the practical build |

Two levers jump out: inertia scales as **`r²`** (geometry) but only **linearly**
with mass. Therefore:

> **Golden rule — put the mass at the rim, and make the radius as large as
> clearance allows.** You reach the target `I_w` with far *less* mass.

This matters because (from the sizing doc) **every gram at the tip increases the
gravity torque and shortens the arm you can balance.** A rim-loaded wheel is the
lightest way to get the inertia you need.

---

## 4. Recommended dimensions

| Quantity | Value | Why |
|---|---|---|
| **Radius `r`** | **~50 mm** (Ø100 mm) | inertia ∝ `r²`; bigger than the Ø35 motor; leaves clearance |
| **Mass `m`** | **~40–60 g** | the minimum that hits `I_w`; light ⇒ longer arm |
| **Mass distribution** | **rim-loaded** (steel ring or bolts at the edge; light hub/spokes) | max inertia per gram |
| **Target `I_w`** | **~1×10⁻⁴ kg·m²** | the momentum requirement (§2), ~3× margin |
| **Thickness** | 3–5 mm printed + a metal rim, *or* ~3 mm solid aluminium | houses the rim mass |
| **Material** | PLA/PETG disk + steel/brass rim (cheap, high inertia), or solid aluminium | — |

### A worked example (PLA disk + steel rim, Ø100)

- PLA disk Ø100 × 3 mm, solid: volume `≈ 23.6 cm³`, density `≈ 1.24 g/cm³`
  → `m ≈ 29 g`, `I_{disk} = ½·0.029·0.05² ≈ 3.6×10⁻⁵`.
- Add a **steel ring at `r ≈ 48 mm`** to reach the target: need
  `ΔI ≈ 6.4×10⁻⁵`, so `m_{ring} = ΔI/r² = 6.4×10⁻⁵/0.0023 ≈ 28 g`.
- **Total ≈ 57 g, `I_w ≈ 1×10⁻⁴ kg·m²`.** ✓

To shave mass, replace the solid PLA centre with **spokes** (hub + 3–5 thin
arms): cuts ~15 g for a small inertia loss, landing around 40 g / `≈9×10⁻⁵`.
Bolts at the rim (e.g. 8× M3 nut-and-bolt) are a convenient way to *fine-tune*
inertia and **balance** the wheel afterwards.

### Sanity checks with these numbers

- Momentum capacity: `H_max = 1×10⁻⁴ · 120 ≈ 1.2×10⁻² kg·m²/s` > the ~4×10⁻³
  needed → ~3× margin. ✓
- Spin-up time at peak torque (`τ ≈ 0.04 N·m`): `t = I_w ω/τ = 1×10⁻⁴·120/0.04
  ≈ 0.3 s`. Fast enough to react; the wheel reaches useful speed well within a
  fall. ✓

---

## 5. The trade-offs (size it to the requirement, no more)

| If you increase… | You gain | You pay |
|---|---|---|
| `I_w` (heavier / bigger wheel) | more momentum capacity, longer before saturation | more tip mass → shorter arm; slower to reach a given speed (but **same** reaction torque) |
| Radius `r` | inertia "for free" (∝ r²) | clearance with the arm/base; balance harder; aesthetics |
| Wheel mass at fixed `I_w` (solid vs rim) | — | a solid disk needs **2×** the mass of a rim → worse for the arm |

Conclusion: hit `I_w ≈ 1×10⁻⁴` with the **largest practical radius** and the
**least mass** (rim-loaded). Don't over-build inertia — extra mass hurts the arm
and the reaction torque doesn't improve.

---

## 6. Mounting — yes, the motor is at the centre

The GBM2804 is an **outrunner**: the outer **bell rotates**, the inner base
(stator) is fixed. So the assembly is **concentric** with the motor axis:

```
        ┌──────── reaction wheel ────────┐   ← bolts to the rotating BELL, spins
        │   ╔══════════════════════╗     │
        │   ║   motor bell (rotor) ║     │   ← rotates with the wheel
        │   ║   ┌──────────────┐   ║     │
        │   ║   │ stator/base  │   ║     │   ← FIXED
        │   ╚═══╪══════════════╪═══╝     │
        └───────┼──────────────┼─────────┘
                │  pendulum arm │           ← stator base bolted here (fixed)
                │   (rear shaft: AS5600 magnet for θ_w)
```

- **Stator base → top of the pendulum arm** (the fixed part).
- **Reaction wheel → the rotating bell**, concentric, on the motor axis.
- **Motor-angle magnet (AS5600 #2) → the rear hollow shaft** (the "encoder
  seat", Ø7), reading `θ_w`/`θ̇_w` — see [sensors_and_i2c.md](sensors_and_i2c.md).

**Measure your motor's bell bolt pattern** (typically M2.5 on a small circle) and
model the wheel's hub holes to match, coaxial with the shaft.

---

## 7. CAD / build rules (Onshape)

- **Use the mass-properties tool.** Sketch the wheel, assign the correct
  **material density** (PLA ≈ 1.24, PETG ≈ 1.27, aluminium ≈ 2.70, steel ≈ 7.85,
  brass ≈ 8.5 g/cm³), and Onshape reports the **mass and moment of inertia**
  directly — iterate the rim until `I_w ≈ 1×10⁻⁴`. No hand-integration needed.
- **Balance it.** Make it symmetric and **concentric** with the spin axis. A
  static/dynamic imbalance creates a once-per-revolution disturbance force
  `≈ m_{unbal}·e·ω²` (grows with the *square* of speed) — exactly the kind of
  periodic torque the controller must fight. Keep tolerances tight on the bore
  and bolt circle; tune with rim screws.
- **Clearance.** With `r = 50 mm` on a ~150 mm arm, check the wheel never hits
  the base or arm when the pendulum tilts ±20–30°. Shorter arm ⇒ smaller wheel.
- **Mount plane perpendicular to the axis.** A tilted wheel adds gyroscopic
  cross-coupling not in the planar model.
- **Export STL → slice (Cura/PrusaSlicer/Bambu) → print** in PLA/PETG; press-fit
  or glue the metal rim, then balance.

---

## 8. Close the loop with the simulator

Onshape gives you the **real** mass and inertia. Put them in the config and let
the optimiser confirm the design works *before* you print and buy:

```yaml
# configs/pendulum_real.yaml
plant:
  wheel:
    mass: 0.057          # from Onshape mass properties
    radius: 0.05
    inertia: 1.0e-4      # I_w from Onshape (about the spin axis)
    friction_coefficient: 1.0e-4
  pendulum_mass: 0.12    # tip assembly
  pendulum_length: 0.15  # the arm L
  # ... motor: resistance 10, torque_constant 0.037, max_voltage 12 ...
objective: {U_max: 12.0, ...}
```

Run `scripts/run_experiment.py`: if the run diverges, or the controller
saturates `U_max`, the design is short of authority somewhere. **Do not assume
that means the wheel** — saturating `U_max` is a *torque* symptom, and on this
build the wheel turned out never to be the binding constraint (§9). Check which
limit actually binds before adding rim mass. That paper → CAD → simulation loop
is the whole design method.

---

## 9. The as-built wheel (v2) — and why adding wire is the wrong fix

The printed PLA wheel came out **lighter and less inertial** than §4's estimate:

| | §4 estimate | **as built (Onshape)** |
|---|---|---|
| wheel mass | 57–77 g | **41.0 g** |
| $I_w$ | $1.0\text{--}1.35\times10^{-4}$ | **$6.117\times10^{-5}$ kg·m²** (61.17 kg·mm²) |
| $m_{tip}$ | ~120–137 g | **101 g** |
| $I_b = m_{tip}L^2$ | ~3.0×10⁻³ | **2.27×10⁻³ kg·m²** |

That is **64 %** of the §2 target $I_w \approx 1.1\times10^{-4}$, which by the
momentum argument alone drops the catch angle from 12° to **7.6°**. The obvious
fix is to add rim mass — galvanized wire around the Ø102 mm rim, where
$\Delta I = m r^2$ is most efficient (~20 g of wire: 2.5 turns of 2 mm, or 10
turns of 1 mm, closes the gap). **That fix does not work, and it is worth
understanding why.**

### The momentum requirement is not the binding constraint

Two limits set the catch angle, and they pull in *opposite* directions as mass
goes onto the rim:

$$\underbrace{I_w\,\omega_{max} \ge 3\,I_b\,\omega_n\theta_c}_{\text{momentum — wire HELPS}}
\qquad\qquad
\underbrace{\tau \ge m_{tip}\,g\,L\sin\theta_c}_{\text{torque — wire HURTS}}$$

Wire adds inertia (raising the momentum ceiling), but it also adds tip mass, so
gravity's torque about the pivot grows and the *torque* ceiling falls. With the
arm fixed at $L = 15$ cm and $\tau_{design} = 0.02$ N·m:

| wire | $m_{tip}$ | $I_w$ (kg·mm²) | $\theta_c$ momentum | $\theta_c$ torque | usable |
|---|---|---|---|---|---|
| **0 g (as built)** | 101 g | 61.2 | 7.63° | 7.73° | **7.63°** |
| +20 g | 121 g | 113.2 | 11.78° | 6.45° | 6.45° |
| +42 g | 143 g | 170.4 | 15.01° | 5.45° | 5.45° |

The two curves cross at **0.3 g of wire**. The as-built wheel is, by accident,
almost exactly matched to the motor's design torque — 7.63° of momentum capacity
against 7.73° of torque authority. **Past that point every gram of wire buys
momentum the motor has no torque to spend, while making the pendulum harder to
hold.** (If you are willing to spend the brief *stall* torque 0.044 N·m on the
catch rather than the derated 0.02 N·m, the crossing moves out to ~29 g — the
only regime in which wire helps at all.)

### The simulator agrees

Sweeping the largest recoverable initial tilt on the **nonlinear** plant, with
one fixed balancing LQG and the 12 V rail enforced
([`configs/pendulum_build_v2.yaml`](../../configs/pendulum_build_v2.yaml)):

| rail | as built (41 g) | +20 g wire | +42 g wire |
|---|---|---|---|
| **12 V** | **10.5°** | 10.8° | 10.0° |
| 18 V | 15.9° | 16.3° | 15.1° |
| 24 V | 21.3° | 21.9° | 20.3° |
| 36 V | 32.5° | 33.6° | 31.2° |

Read the table both ways. **Across a row** (adding wire) nothing happens: +20 g
buys **+0.3°**, and +42 g is *worse* than the bare wheel. **Down a column**
(raising the rail) the catch angle scales almost linearly — tripling the rail
triples it. The system is **voltage/torque limited, not momentum limited**,
exactly as the crossing analysis predicts. In these rollouts the wheel peaks near
185 rad/s while the *commanded* voltage clips hard against 12 V: the wheel still
has momentum headroom when the motor has already run out of torque.

### Verdict

- **Do not add the wire.** It is the right fix for a momentum-limited design, and
  this one is not momentum-limited. Save the 20 g — that mass costs you torque
  headroom you cannot spare.
- **The wheel as printed is fine**, and better than the §2 momentum estimate
  suggests: the honest figure is a **~10.5° catch angle at 12 V**, not 7.6°
  (§2's 3× safety factor is deliberately conservative).
- **If you want a bigger catch angle, buy volts, not grams** — the rail is the
  only lever in that table that moves. Failing that, shorten the arm (the $L$
  that appears in *both* constraints), per
  [sizing_the_pendulum.md](sizing_the_pendulum.md).

Control verification of the as-built plant — tuning, stability margins, and the
delay-fragility caveat — is in
[`docs/papers/robustness_lqg_measured.md`](../papers/robustness_lqg_measured.md).

---

## Sources

- Datasheet figures for the iPower GBM2804H-100T (as supplied).
- Moments of inertia of standard shapes (any classical-mechanics text).
- Reaction-wheel pendulum reference build: [GitHub](https://github.com/simplefoc/Arduino-FOC-reaction-wheel-inverted-pendulum), [project docs](https://docs.simplefoc.com/simplefoc_pendulum).
