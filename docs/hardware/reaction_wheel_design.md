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

Run `scripts/run_experiment.py`: if the controller saturates `U_max` or the run
diverges, the wheel has too little momentum (raise `I_w` / `ω_max`) or the arm is
too long/heavy (see [sizing_the_pendulum.md](sizing_the_pendulum.md)). That
paper → CAD → simulation loop is the whole design method.

---

## Sources

- Datasheet figures for the iPower GBM2804H-100T (as supplied).
- Moments of inertia of standard shapes (any classical-mechanics text).
- Reaction-wheel pendulum reference build: [GitHub](https://github.com/simplefoc/Arduino-FOC-reaction-wheel-inverted-pendulum), [project docs](https://docs.simplefoc.com/simplefoc_pendulum).
