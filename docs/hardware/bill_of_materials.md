# Hardware — Bill of Materials and Wiring

This is the complete hardware list to physically build the reaction-wheel
inverted pendulum that the rest of this repository simulates and tunes. The
design follows the well-documented **SimpleFOC reaction-wheel pendulum**
(CAD + reference firmware exist), adapted to the sensor set our control model
assumes. Target budget: **€100–150** all-in.

Prices are approximate (incl. VAT, EU/AliExpress sources, mid-2026) and
fluctuate — treat them as ranges. Links are in the [Sources](#sources) list.

---

## 1. How the hardware maps to the control model

Our model (see [`docs/theory/model.md`](../theory/model.md),
[`notation.md`](../theory/notation.md) §2–3):

| In the model | Real part |
|---|---|
| Motor `τ = K_t·i`, voltage input, `R_a`, `K_t`, `K_e` | a **gimbal BLDC** run in **FOC torque mode** — FOC makes a BLDC behave as a current→torque source `τ ≈ K_t·i`, *exactly* the DC-motor model |
| Reaction wheel `I_w` | a balanced disk on the motor shaft |
| State `θ_p` (pendulum angle), `θ̇_p` | absolute angle sensor at the **pivot** |
| State `θ_w`, `θ̇_w` (wheel angle/rate) | absolute angle sensor on the **motor** (also required by FOC) |
| Measured outputs `[θ_p, θ̇_w]` (the `C` matrix) | the two sensors above |
| Controller at `dt = 0.01 s` (100 Hz) | the microcontroller running the discrete LQG |
| `max_voltage` | the DC bus (12 V) |

Because FOC turns the BLDC into a torque source, **the LQR/Kalman gains tuned in
simulation transfer directly** — you feed the optimiser's `K` (and the Kalman
filter) the same way, commanding motor current/voltage.

---

## 2. Recommended build (BLDC + SimpleFOC)

### Electronics

| # | Component | Role | Qty | Price (€) |
|---|---|---|---|---|
| 1 | **ESP32 DevKit v1** (38-pin) | runs the discrete LQG @100 Hz; 2 hardware I²C buses; Wi-Fi for logging | 1 | 8–12 |
| 2 | **SimpleFOCMini** (DRV8313) driver | 3-PWM BLDC driver, 8–30 V, ~2.5 A | 1 | 10–17 |
| 3 | **Gimbal BLDC motor** iPower GM4108-120T | the actuator (reaction-wheel drive) | 1 | 28–35 |
| 4 | **AS5600** magnetic encoder board | motor angle → `θ_w`, `θ̇_w` (and FOC commutation) | 1 | 5–7 |
| 5 | **AS5600** magnetic encoder board | pivot angle → `θ_p`, `θ̇_p` | 1 | 5–7 |
| 6 | **Diametric magnet** 6×2.5 mm (axially mounted, *diametrically* magnetised) | one per AS5600 | 2 | 2–4 |
| 7 | **12 V / 3 A DC power supply** (barrel jack) | the DC bus | 1 | 10–14 |
| 8 | **Electrolytic capacitor** 220–470 µF / 35 V | decouples the driver bus (back-EMF spikes) | 1 | 1 |
| 9 | **Breadboard** (half-size) + **Dupont jumper wires** (M-M, M-F) | wiring | 1 set | 5–7 |
| 10 | **USB cable** (micro-USB or USB-C, per your ESP32) | flashing + bench power for the ESP32 | 1 | 2–3 |

### Mechanics

| # | Component | Role | Qty | Price (€) |
|---|---|---|---|---|
| 11 | **Reaction wheel** — disk ~80–120 g (3D-printed with a steel rim, or a machined aluminium disk Ø~110–120 mm) | the inertia `I_w` | 1 | 4–10 |
| 12 | **Ball bearings** 8×16×5 mm (608-style is 8×22×7; pick to match your shaft) | the pivot (and wheel hub if needed) | 2–3 | 5–8 |
| 13 | **Pivot shaft** Ø8 mm steel/aluminium, ~80–100 mm | the pendulum pivot axle | 1 | 3–5 |
| 14 | **3D-printed parts**: base, pendulum arm, motor mount, two magnet holders | the structure | set | 4–8 (filament) |
| 15 | **M3 screws + nuts + nylon standoffs** assortment | fastening / mounting the boards | set | 5–7 |

**Recommended subtotal: ≈ €97–124** — comfortably inside €150, leaving margin
for shipping or a spare motor/sensor.

> If you do **not** have a 3D printer, add ~€15–25 for a printing service (items
> 11 and 14), or substitute an aluminium **2020 extrusion** base + a turned
> metal disk.

---

## 3. Alternatives (swap any row above)

| Category | Cheaper | Recommended | Higher-end |
|---|---|---|---|
| **Motor** | GBM2208 (~€15–20) — light pendulums (~0.2 kg) | **GBM2804H-100T** (~€23) | **GM4108-120T** (~€30) — more torque, the reference build's motor (best for the ~0.5 kg "sensible" config) |
| **Driver** | SimpleFOCMini clone / bare DRV8313 (~€9) | **SimpleFOCMini** (~€12) | **SimpleFOCShield v2** (~€28–35) — higher current, screw terminals |
| **MCU** | Arduino Nano clone (~€5, 8-bit — tight) | **ESP32** (~€10) | **Teensy 4.0** (~€25) / STM32 — headroom for the full Kalman filter + fast logging |
| **Pendulum sensor** | **MPU6050** IMU (~€4) — needs accel+gyro fusion for `θ_p` | **AS5600** at the pivot (~€6) — direct angle, matches the `C` matrix | **AMT103-V** optical encoder (~€22) — high resolution (used by the original project; pricey) |
| **Power** | bench/wall **12 V** adapter (~€12) | same | **3S LiPo (11.1 V)** + charger (~€25–30) — portable/standalone |

### "Literal DC-motor" variant (~€60–90)
If you want the *brushed* DC motor of the model literally: a small DC motor
**with a quadrature encoder** + an H-bridge driver
(**TB6612FNG** ~€5, or **BTS7960** ~€8 for more current), plus AS5600/MPU6050 for
the pendulum. Cheaper, but a good high-speed *encodered* brushed motor is harder
to source well, and FOC gives smoother torque — hence the BLDC route is the
primary recommendation.

---

## 4. Wiring and pinout

### The one gotcha: two AS5600 on I²C

The AS5600 has a **fixed I²C address `0x36`** (no address pins), so **two of them
cannot share one bus**. Three ways out:

1. **Two I²C buses (recommended, free):** the ESP32 has two hardware I²C
   peripherals — put one AS5600 on `Wire` and the other on `Wire1` with separate
   pins. SimpleFOC's `MagneticSensorI2C` accepts a `TwoWire*`.
2. **I²C multiplexer:** add a **TCA9548A** (~€2–3) and select channels.
3. **Mixed sensors:** use one AS5600 (motor, I²C) + one **MPU6050** (pendulum,
   address `0x68` → no clash), at the cost of sensor fusion for `θ_p`.

This BOM uses option 1.

### Pin map (ESP32 DevKit v1)

| ESP32 GPIO | Connects to | Notes |
|---|---|---|
| GPIO 25 | SimpleFOCMini **IN1** | LEDC PWM |
| GPIO 26 | SimpleFOCMini **IN2** | LEDC PWM |
| GPIO 27 | SimpleFOCMini **IN3** | LEDC PWM |
| GPIO 13 | SimpleFOCMini **EN** | driver enable (avoid strapping pins 0/2/12/15) |
| GPIO 21 / GPIO 22 | **AS5600 #1 (motor)** SDA / SCL | `Wire` (bus 0) |
| GPIO 18 / GPIO 19 | **AS5600 #2 (pivot)** SDA / SCL | `Wire1` (bus 1, custom pins) |
| 3V3 | VCC of both AS5600 | AS5600 is a 3.3 V part |
| GND | common ground | **tie ESP32 / driver / sensors / PSU grounds together** |

### Power / motor

```
 12 V PSU (+) ──┬──────────────► SimpleFOCMini  VM (motor supply +)
                │
              [ 220–470 µF ]  (across VM–GND, close to the driver)
                │
 12 V PSU (−) ──┴──────────────► SimpleFOCMini  GND ── common GND ── ESP32 GND ── AS5600 GND

 SimpleFOCMini  A / B / C  ─────► motor phase wires (any order; FOC calibration sorts it)
 ESP32 (USB)   ───────────────►  ESP32 logic power + flashing
```

For a **standalone** (no-USB) build, add an **MP1584 buck** (~€2) set to 5 V from
the 12 V bus into the ESP32 `5V`/`VIN` pin; keep grounds common.

### Sensor placement

- **Motor AS5600**: a diametric magnet glued to the motor shaft's free end, the
  sensor board fixed ~1–2 mm above it. Gives `θ_w` (and FOC commutation);
  velocity `θ̇_w` from SimpleFOC.
- **Pivot AS5600**: a diametric magnet on the pivot shaft, sensor on the fixed
  frame. Gives `θ_p` directly — **zero it at the upright equilibrium** in
  firmware (so `θ_p = 0` is balanced, as in the model).

---

## 5. Firmware / bring-up notes

1. **SimpleFOC** ([docs.simplefoc.com](https://docs.simplefoc.com)): configure
   `BLDCMotor`, `BLDCDriver3PWM(25, 26, 27, 13)`, and the motor
   `MagneticSensorI2C(AS5600_I2C)` on `Wire`. Run the alignment/calibration once
   to find pole pairs and the zero electrical angle.
2. **Torque mode**: set `motor.controller = MotionControlType::torque` and
   command current/voltage. This is the `τ = K_t·i` input your LQG expects.
3. **Pivot sensor** on `Wire1`: `Wire1.begin(18, 19); MagneticSensorI2C as_pivot(AS5600_I2C); as_pivot.init(&Wire1);`
4. **Control loop @100 Hz**: read `θ_p` (pivot), `θ̇_p` (differentiate/filter),
   `θ_w`, `θ̇_w` (motor) → run the discrete Kalman + `u = −K x̂` (the gains from
   `scripts/run_experiment.py`) → command the motor. Match `dt = 0.01 s`.
5. **Safety**: software limit on `|θ_p|` (cut the motor past ~±30°, the
   small-angle validity edge) and a current/voltage cap (`U_max` = 12 V).

> **Identify the real parameters.** Measure the actual `m_p, ℓ_p, I_b, I_w, R_a,
> K_t, K_e, b_*` of your build and put them in a config (e.g.
> `configs/pendulum_real.yaml`), then re-run the optimiser — the simulation is
> only as good as these numbers (see the analysis in
> [`docs/guides/experiments.md`](../guides/experiments.md)).

---

## 6. Tools needed (not in the budget)

3D printer or a printing service · soldering iron + solder · small hex/Allen
keys + screwdrivers · a multimeter · (optional) calipers to measure parts for
parameter identification.

---

## 7. Budget summary

| Build | Approx. total |
|---|---|
| **Recommended (BLDC + SimpleFOC, GM4108)** | **€100–125** |
| Tight (GBM2804, MPU6050 for pendulum, printed wheel) | €80–100 |
| Literal brushed-DC (DC motor + encoder + H-bridge) | €60–90 |

All three sit inside the €150 ceiling. The recommended build is favoured because
its CAD and reference firmware already exist, and FOC torque mode matches the
simulated DC-motor model.

---

## Sources

- [SimpleFOC reaction-wheel inverted pendulum — GitHub (BOM + firmware)](https://github.com/simplefoc/Arduino-FOC-reaction-wheel-inverted-pendulum)
- [SimpleFOC pendulum — project docs](https://docs.simplefoc.com/simplefoc_pendulum)
- [Reaction-wheel pendulum CAD — Thingiverse](https://www.thingiverse.com/thing:4416233)
- [Community build with two AS5600](https://community.simplefoc.com/t/reaction-wheel-inverted-pendulum-with-two-as5600/6445)
- [SimpleFOCMini — specs & purchase](https://docs.simplefoc.com/simplefocmini) · [eBay listing](https://www.ebay.com/itm/186547994843)
- [iPower GM4108-120T gimbal motor](https://shop.iflight.com/ipower-gimbal-brushless-motor-gbm4108h-120t-pro220) · [GBM2804H-100T](https://shop.iflight.com/ipower-gimbal-brushless-motor-gbm2804h-100t-pro218)
- [AS5600 module (Amazon)](https://www.amazon.com/Magnetic-Encoder-Induction-Measurement-Precision/dp/B0CJ83FNL4) · [AS5600 datasheet (fixed address 0x36)](https://ams.com/as5600)
- [ESP32 DevKit board](https://robocraze.com/products/esp32-development-board)
- [SimpleFOC documentation](https://docs.simplefoc.com)
