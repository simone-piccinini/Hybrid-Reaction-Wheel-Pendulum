# Sensors, the I²C bus, and the magnets

A practical explainer for the angle sensing of the reaction-wheel pendulum: how
many sensors you need and why, how the I²C bus works (and the one trap with two
identical AS5600), and what the magnets are for. Companion to the
[Bill of Materials](bill_of_materials.md).

---

## 1. How many sensors — two, and why

The controller does **not** need a sensor per state. The model's measurement
matrix `C` (see [`notation.md`](../theory/notation.md) §3) reads only **two**
quantities; the Kalman filter reconstructs the rest.

| Sensor | Mounted on | Measures | The model gives the rest |
|---|---|---|---|
| **AS5600 #1** | the **pivot** axle | `θ_p` — pendulum angle | `θ̇_p` by differentiating/filtering |
| **AS5600 #2** | the **motor** shaft | `θ_w` → `θ̇_w` — wheel speed | `θ_w` tracked; also drives FOC commutation |

So the full state `[θ_p, θ̇_p, θ_w, θ̇_w]` is recovered from **2 angle sensors**:

- **Velocities are not separate sensors** — `θ̇_p` and `θ̇_w` are computed from
  the angle readings (SimpleFOC already estimates the motor velocity).
- **The Kalman filter fills the gaps** — that is exactly its job: estimate the
  unmeasured states from the two measurements plus the dynamics model.
- **The motor sensor is required anyway** — a BLDC under FOC needs rotor
  position to commutate, so AS5600 #2 is not "extra"; the motor won't run well
  without it.

You **cannot** do it with one: the pivot and the motor are two different shafts.
**Two sensors, minimum.** (Swapping the pivot AS5600 for an MPU6050 IMU is still
one module → still two sensors total.)

---

## 2. How the I²C bus works

**I²C** (Inter-Integrated Circuit) is a two-wire serial bus that lets one
**controller** (the microcontroller) talk to many **peripherals** (sensors) over
just two shared signal lines, plus ground:

```
        MCU (controller)
         │      │
   SDA ──┼──────┼─────────●───────────●──────   (data,  bidirectional)
   SCL ──┼──────┼─────────●───────────●──────   (clock, driven by the MCU)
         │      │      AS5600 #1   AS5600 #2 …
        Rp     Rp      (0x36)      (0x36) ← same address!
       (pull-up resistors to 3.3 V; usually on the breakout boards)
```

- **SDA** carries data, **SCL** the clock. Both are *open-drain* — devices only
  pull them low, and **pull-up resistors** hold them high (the Grove/most AS5600
  boards include them).
- Every peripheral has a **7-bit address**. To read a sensor the MCU puts that
  address on the bus; only the matching device responds. That is how many
  devices coexist on the same two wires.
- It is **3.3 V** here (the AS5600 is a 3.3 V part; the Teensy 4.0 / ESP32 are
  3.3 V logic — keep the whole bus at 3.3 V).

### The trap: two AS5600 share one address

The AS5600 has a **fixed I²C address `0x36`** — there are no address-select pins.
So **two AS5600 on the same bus both answer to `0x36` and collide**: the MCU
cannot tell them apart. Four ways to solve it:

1. **Use two (or three) hardware I²C buses — the clean, free fix.** A
   microcontroller often has more than one I²C peripheral, each with its own
   SDA/SCL pins. Put one AS5600 on each bus; both keep address `0x36` but live
   on physically separate wires.
   - **Teensy 4.0**: **three** I²C buses (`Wire`, `Wire1`, `Wire2`) — plenty.
   - **ESP32**: **two** I²C buses (`Wire`, `Wire1`) — exactly enough.
   SimpleFOC's `MagneticSensorI2C` accepts which bus to use (`sensor.init(&Wire1)`).
2. **I²C multiplexer** (e.g. **TCA9548A**, ~€2–3): one upstream bus, eight
   switchable downstream channels; you select a channel, talk to the AS5600 on
   it, switch. Useful if your MCU has only one bus.
3. **Non-I²C output**: the AS5600 also outputs the angle as a **PWM or analog**
   signal on its `OUT` pin — read one sensor that way (a timer/ADC pin) and free
   the I²C address.
4. **Mixed sensors**: pair one AS5600 (`0x36`) with an **MPU6050** IMU
   (`0x68`) — different addresses, so they share one bus with no clash (at the
   cost of accel+gyro fusion for `θ_p`).

This project uses **option 1** — two hardware buses, one sensor each.

### Bus / pin recap

| MCU | Bus 0 (motor sensor) | Bus 1 (pivot sensor) | Spare |
|---|---|---|---|
| **Teensy 4.0** | `Wire` — SDA 18 / SCL 19 | `Wire1` — SDA 17 / SCL 16 | `Wire2` — SDA 25 / SCL 24 |
| **ESP32 DevKit** | `Wire` — SDA 21 / SCL 22 | `Wire1` — SDA 19 / SCL 18 | — |

(Always tie **all grounds together** — MCU, both sensors, driver, power supply.)

---

## 3. What the magnets are for

The AS5600 is **non-contact**: it does not touch the shaft. Instead it senses the
**direction of a magnetic field** with internal Hall sensors and reports that
angle (12-bit, 4096 steps/turn). The field comes from a small **magnet glued to
the end of the rotating shaft**, spinning ~1–2 mm under the chip.

So **each sensor needs its own magnet** — that is why you buy **extra magnets**:
**one per AS5600** (two for this build; grab a spare or two, they are cheap and
easy to lose or crack when gluing).

### It must be a *diametric* magnet

This is the detail people get wrong. The magnet must be magnetised **across its
diameter** (north on one side, south on the other), **not** through its
thickness (axially):

```
  DIAMETRIC (correct)          AXIAL (wrong)
      ┌───────┐                  ┌───────┐
      │ N │ S │   ← field        │   N   │   ← field points
      └───────┘     rotates      ├───────┤      straight up;
   the in-plane field            │   S   │      the sensor sees
   direction = the angle         └───────┘      (almost) no rotation
```

An axial magnet (an ordinary fridge/disc magnet) gives the AS5600 nothing to
track — the readings will be junk. A typical part is a **Ø6 × 3 mm diametric
neodymium magnet**. They are usually **not included** with the breakout board,
so they go on the shopping list separately.

### Mounting rules (from the datasheet)

- **Centred**: the magnet's centre must sit on the **rotation axis**, aligned
  with the chip centre — off-centre mounting corrupts the angle.
- **Air gap 0.5–3 mm** between the magnet face and the chip.
- **Size** similar to the chip package (≈6 mm).
- A small **3D-printed holder** pressed/glued onto each shaft end keeps the
  magnet centred and at the right height (one for the pivot axle, one for the
  motor shaft).

---

## 4. Summary

- **Two AS5600** (pivot → `θ_p`, motor → `θ_w/θ̇_w`); the Kalman filter
  reconstructs the other two states.
- **I²C** = two shared wires + addresses; but **two AS5600 = same address `0x36`**
  → put them on **separate hardware buses** (Teensy 4.0 has 3, ESP32 has 2).
- **One diametric magnet per sensor** (buy 2 + spares); diametric, centred on the
  shaft, 0.5–3 mm under the chip. The "extra" magnets are simply these — the
  sensors are sold without them.

See the full parts list and pinout in the [Bill of Materials](bill_of_materials.md).
