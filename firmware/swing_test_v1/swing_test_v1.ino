// SWING TEST / PARAMETER IDENTIFICATION  -  v1
// Reaction-wheel pendulum, Teensy 4.1 + GBM2804H-100T + 2x AS5600
//
// PURPOSE
//   Stages 0-3 of docs/hardware/bringup_pipeline.md. This sketch MEASURES the
//   plant and makes the arm swing a little. It does NOT balance - there is no
//   LQR and no Kalman filter here on purpose. Balancing is stage 5, after the
//   numbers this produces have been fed back into the simulation.
//
//   Every logging mode streams CSV to serial. Capture it to a file and run
//   scripts/identify_parameters.py on it to get a ready config fragment.
//
// INHERITED FROM v6 (closed_loop_v6_dualsensor.ino) - do not weaken:
//   closed-loop commutation from the measured rotor angle, the machine-driven
//   self-calibration sweep, sensorDir detection, spike rejection, overspeed,
//   stall detect, thermal budget, FAULT pin, session limit. Each corresponds to
//   a failure that actually happened during bring-up (handoff section 9).
//
// SENSOR ROLES (note: B is the pendulum, A is the wheel)
//   A: AS5600 ANALOG, pin 23   -> WHEEL / rotor angle, drives commutation
//   B: AS5600 I2C,   18/19     -> PIVOT angle = the pendulum angle theta_p
//
// WIRING: identical to v6. GPO must FLOAT, encoder VCC must be 3.3 V.
//
// COMMANDS (115200 baud)
//   s            status
//   n [secs]     STAGE 0: sensor noise floor, motor never enabled
//   w [secs]     STAGE 1: free-swing log, motor never enabled
//   r            arm (self-calibrate encoder A, ~8 s, wheel must spin freely)
//   k <duty>     STAGE 2: wheel spin-up step         [CLAMP THE ARM]
//   d <duty>     STAGE 2: spin up then coast down    [CLAMP THE ARM]
//   t <duty> <ms>  STAGE 3: single torque pulse, log the arm's response
//   e <gain> <deg> STAGE 3: gentle resonant swing, abort above <deg>
//   c <duty>     duty cap        x  disarm        h  coast
//   z            flip encoder A direction
//
// SAFETY NOTE ON MODE e: it deliberately adds energy to a swinging arm. It
// aborts on amplitude, on session time, and on every v6 layer. Start with a low
// gain and a low cap. Keep clear of the arm.

#include <math.h>
#include <Wire.h>

// ---- Sensor A: analog (WHEEL) ----
const int PIN_SENSOR_OUT = 23;
const int ADC_BITS = 12;
const int ADC_MAX  = (1 << ADC_BITS) - 1;

// Nominal counts per full turn, used ONLY before self-calibration so that
// stage 0 can measure encoder A's noise floor without arming the motor. The
// handoff measured 4060-4065 counts consistently across runs. Do not use this
// for commutation - that needs the machine-driven sweep (handoff section 4.3).
const float NOMINAL_SPAN_COUNTS = 4062.0f;

// ---- Sensor B: I2C (PENDULUM PIVOT) ----
#define AS5600_ADDR 0x36
#define REG_ANGLE   0x0E
#define REG_STATUS  0x0B
#define REG_AGC     0x1A
bool  sensorB_present = false;
float sensorB_deg = -1;
float sensorB_cont = 0, sensorB_prev = 0;
bool  sensorB_started = false;
float pivotZero = 0;                 // set by 'o' or at log start

// ---- Driver pins ----
const int PIN_IN1 = 2, PIN_IN2 = 3, PIN_IN3 = 4, PIN_EN = 5, PIN_FAULT = 8;
const bool FAULT_WIRED = true;

const int POLE_PAIRS = 7;
const int PWM_BITS = 10;
const int PWM_MAX  = (1 << PWM_BITS) - 1;
const unsigned long CONTROL_PERIOD_US = 1000;   // 1 kHz

// ---- Safety ----
const int HARD_DUTY_CEILING = 450;   // stays at the v6 value for BRING-UP.
                                     // thermal_and_duty_limits.md justifies
                                     // raising it, but only AFTER the stage-2
                                     // bench current check. Do not raise it
                                     // before that measurement.
int dutyCap = 250;
const float SPIKE_REJECT_DEG = 8.0f;
const float MAX_SAFE_DPS = 1500.0f;
const int           STALL_DUTY_MIN   = 100;
const float         STALL_DPS_MAX    = 15.0f;
const unsigned long STALL_TIMEOUT_MS = 1500;
const unsigned long THERMAL_WINDOW_MS = 3000;
const float DUTY_BUDGET_THROTTLE = 200.0f;
const float DUTY_BUDGET_CUT      = 340.0f;
const unsigned long MAX_SESSION_MS = 120000;

// ---- Self-calibration sweep ----
const int ALIGN_DUTY = 130, SWEEP_DUTY = 170, SWEEP_STEP = 2, SWEEP_DELAY = 6;
const int MIN_VALID_SPAN = 2500;
const float DUTY_RAMP_PER_SEC = 400.0f;
const float VEL_ALPHA = 0.02f;

// ---- Logging ----
const unsigned long LOG_PERIOD_US = 5000;   // 200 Hz CSV
const unsigned long MAX_LOG_MS    = 60000;

enum Mode { IDLE, TORQUE, PULSE, SPINUP, COAST, PUMP };
enum LogKind { LOG_NONE, LOG_NOISE, LOG_SWING, LOG_SPIN, LOG_PULSE, LOG_PUMP };

Mode mode = IDLE;
LogKind logKind = LOG_NONE;
bool armed = false, calibrated = false;
int  calMin = -1, calMax = -1, sensorDir = 1;

float offsetDeg = 0, contAngle = 0, prevAngle = 0, velocity = 0;
float dutyRequest = 0, dutyActual = 0, viTerm = 0;
unsigned long sessionStart = 0, lastMicros = 0, thermalStart = 0;
float dutyAccum = 0, dutyAvg = 0;
unsigned long movingSince = 0;

// logging / test state
unsigned long logStart = 0, logUntil = 0, lastLogUs = 0;
unsigned long pulseUntil = 0;
float pumpGain = 8.0f, pumpAbortDeg = 15.0f;
float pivotVel = 0, pivotPrev = 0;
unsigned long pivotPrevUs = 0;
float lastCmd = 0;

// ---------------------------------------------------------------- setup

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000) {}

  pinMode(PIN_IN1, OUTPUT); pinMode(PIN_IN2, OUTPUT); pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_EN, OUTPUT);
  if (FAULT_WIRED) pinMode(PIN_FAULT, INPUT_PULLUP);

  analogWriteFrequency(PIN_IN1, 20000);
  analogWriteFrequency(PIN_IN2, 20000);
  analogWriteFrequency(PIN_IN3, 20000);
  analogWriteResolution(PWM_BITS);
  killOutputs();

  analogReadResolution(ADC_BITS);
  analogReadAveraging(8);
  pinMode(PIN_SENSOR_OUT, INPUT_DISABLE);

  Wire.begin();
  Wire.setClock(1000000);   // Fast-Mode-Plus; AS5600 supports it. A 2-byte read
                            // is ~40 us, so the pivot can be sampled in-loop.

  Serial.println();
  Serial.println("=========================================");
  Serial.println("  SWING TEST / PARAM ID  v1");
  Serial.println("  A(analog,23)=WHEEL   B(I2C,18/19)=PIVOT");
  Serial.println("=========================================");
  scanSensorB();
  Serial.println("  Stages 0-1 need NO motor power.");
  printHelp();
}

// ---------------------------------------------------------------- loop

void loop() {
  handleSerial();

  // pivot is sampled every cycle - it is the pendulum state
  static unsigned long lastB = 0;
  if (micros() - lastB >= 1000) { lastB = micros(); updateSensorB(); }

  // ---- unarmed logging modes (stage 0 and 1): motor never enabled ----
  if (!armed) {
    killOutputs();
    // Encoder A is logged from the RAW ADC here: it is not calibrated until
    // arming, but stage 0 must still see its noise floor - it is the analog
    // channel and the one that can pick up slip-ring modulation.
    if (logKind == LOG_NOISE || logKind == LOG_SWING)
      serviceLog(rawAngleA_deg(), 0.0f);
    else {
      static unsigned long lastIdle = 0;
      if (millis() - lastIdle >= 1000) {
        lastIdle = millis();
        if (sensorB_present) {
          Serial.print("# idle,pivot_deg,");
          Serial.println(sensorB_cont - pivotZero, 2);
        }
      }
      delay(2);
    }
    return;
  }

  unsigned long now = millis();
  if (now - sessionStart > MAX_SESSION_MS) { stop("session time limit"); return; }
  if (FAULT_WIRED && digitalRead(PIN_FAULT) == LOW) { stop("driver FAULT"); return; }

  unsigned long nowUs = micros();
  if (nowUs - lastMicros < CONTROL_PERIOD_US) return;
  float dt = (nowUs - lastMicros) / 1000000.0f;
  if (dt > 0.05f) dt = 0.05f;
  lastMicros = nowUs;

  float a = readAngleA();
  if (a < 0) { stop("lost sensor A"); return; }
  float d = a - prevAngle;
  if (d >  180.0f) d -= 360.0f;
  if (d < -180.0f) d += 360.0f;
  if (fabsf(d) > SPIKE_REJECT_DEG) return;
  prevAngle = a;
  contAngle += d;
  velocity = (1.0f - VEL_ALPHA) * velocity + VEL_ALPHA * (d / dt);
  if (fabsf(velocity) > MAX_SAFE_DPS) { stop("overspeed"); return; }

  float pos = contAngle - offsetDeg;
  float vSigned = velocity * sensorDir;

  // ---- mode dispatch ----
  if (mode == PULSE && now >= pulseUntil) { dutyRequest = 0; mode = TORQUE; }

  if (mode == COAST) dutyRequest = 0;

  if (mode == PUMP) {
    // Energy pump: push the wheel so its REACTION torque acts along the arm's
    // current direction of travel. Small, bang-bang, amplitude-limited. This is
    // swingup.md's law with a hard cap - it builds a gentle oscillation, not a
    // swing-up.
    float amp = fabsf(pivotAngle());
    if (amp > pumpAbortDeg) { stop("pump amplitude cap reached (expected)"); return; }
    float drive = pumpGain * pivotVel;              // deg/s -> duty
    if (drive >  dutyCap) drive =  dutyCap;
    if (drive < -dutyCap) drive = -dutyCap;
    dutyRequest = drive;
  }

  // ---- ramp, cap, thermal ----
  float rampStep = DUTY_RAMP_PER_SEC * dt;
  if (dutyActual < dutyRequest) { dutyActual += rampStep; if (dutyActual > dutyRequest) dutyActual = dutyRequest; }
  else if (dutyActual > dutyRequest) { dutyActual -= rampStep; if (dutyActual < dutyRequest) dutyActual = dutyRequest; }

  int cap = dutyCap;
  if (dutyAvg > DUTY_BUDGET_THROTTLE) {
    float over = (dutyAvg - DUTY_BUDGET_THROTTLE) / (DUTY_BUDGET_CUT - DUTY_BUDGET_THROTTLE);
    if (over > 1) over = 1;
    cap = (int)(dutyCap * (1.0f - 0.7f * over));
  }
  float cmd = dutyActual;
  if (cmd >  cap) cmd =  cap;
  if (cmd < -cap) cmd = -cap;
  lastCmd = cmd;

  // stall check is suppressed while coasting on purpose (zero duty is expected)
  if (mode != COAST) {
    if (fabsf(velocity) > STALL_DPS_MAX || fabsf(cmd) < STALL_DUTY_MIN) movingSince = now;
    else if (now - movingSince > STALL_TIMEOUT_MS) { stop("stalled - try 'z' then repeat"); return; }
  }

  dutyAccum += fabsf(cmd) * dt;
  if (now - thermalStart >= THERMAL_WINDOW_MS) {
    dutyAvg = dutyAccum / (THERMAL_WINDOW_MS / 1000.0f);
    dutyAccum = 0; thermalStart = now;
    if (dutyAvg > DUTY_BUDGET_CUT) { stop("thermal budget - let it cool"); return; }
  }

  float elec = wrap360(sensorDir * pos * POLE_PAIRS);
  writePhases(wrap360(elec + (cmd >= 0 ? 90.0f : -90.0f)), (int)fabsf(cmd));

  if (logKind != LOG_NONE) serviceLog(contAngle - offsetDeg, vSigned);
  else {
    static unsigned long lastPrint = 0;
    if (now - lastPrint >= 400) {
      lastPrint = now;
      Serial.print("  wheel "); Serial.print(vSigned, 0); Serial.print(" dps");
      Serial.print("  duty "); Serial.print((int)cmd); Serial.print("/"); Serial.print(cap);
      Serial.print("  pivot "); Serial.print(pivotAngle(), 2); Serial.print(" deg");
      Serial.print("  avg "); Serial.println(dutyAvg, 0);
    }
  }
}

// ---------------------------------------------------------------- logging

const char* logName(LogKind k) {
  switch (k) {
    case LOG_NOISE: return "noise";
    case LOG_SWING: return "freeswing";
    case LOG_SPIN:  return "spin";
    case LOG_PULSE: return "pulse";
    case LOG_PUMP:  return "pump";
    default:        return "none";
  }
}

void startLog(LogKind k, unsigned long secs) {
  logKind = k;
  logStart = millis();
  unsigned long ms = secs * 1000UL;
  if (ms > MAX_LOG_MS) ms = MAX_LOG_MS;
  logUntil = logStart + ms;
  lastLogUs = micros();
  Serial.print("# TEST,");    Serial.println(logName(k));
  Serial.print("# DURATION_S,"); Serial.println(secs);
  Serial.println("# columns: t_s,pivot_deg,pivot_dps,wheel_deg,wheel_dps,duty");
  Serial.println("t_s,pivot_deg,pivot_dps,wheel_deg,wheel_dps,duty");
}

void serviceLog(float wheelDeg, float wheelDps) {
  unsigned long nowUs = micros();
  if (nowUs - lastLogUs < LOG_PERIOD_US) return;
  lastLogUs = nowUs;
  unsigned long now = millis();
  if (now >= logUntil) { endLog(); return; }
  Serial.print((now - logStart) / 1000.0f, 4); Serial.print(',');
  Serial.print(pivotAngle(), 3);               Serial.print(',');
  Serial.print(pivotVel, 2);                   Serial.print(',');
  Serial.print(wheelDeg, 3);                   Serial.print(',');
  Serial.print(wheelDps, 1);                   Serial.print(',');
  Serial.println((int)lastCmd);
}

void endLog() {
  Serial.println("# END");
  logKind = LOG_NONE;
  if (armed) { dutyRequest = 0; mode = TORQUE; }
  Serial.println("  log complete.");
}

// ---------------------------------------------------------------- sensors

float rawAngleA_deg() {
  // Uncalibrated scale - noise-floor use only, never for commutation.
  return analogRead(PIN_SENSOR_OUT) * 360.0f / NOMINAL_SPAN_COUNTS;
}

float readAngleA() {
  if (!calibrated) return -1;
  int raw = analogRead(PIN_SENSOR_OUT);
  int span = calMax - calMin;
  if (span < 500) return -1;
  if (raw < calMin - span / 8 || raw > calMax + span / 8) return -1;   // silent-fault guard
  float dg = (raw - calMin) * 360.0f / span;
  if (dg < 0) dg = 0;
  if (dg > 359.99f) dg = 359.99f;
  return dg;
}

float pivotAngle() { return sensorB_cont - pivotZero; }

void scanSensorB() {
  Wire.beginTransmission(AS5600_ADDR);
  sensorB_present = (Wire.endTransmission() == 0);
  Serial.print("  pivot encoder B (I2C): ");
  if (!sensorB_present) {
    Serial.println("NOT FOUND");
    Serial.println("    check SDA=18 SCL=19, 3.3V, shared GND, pull-ups.");
    Serial.println("    Stages 0-3 all need B - fix this before continuing.");
    return;
  }
  Serial.println("found at 0x36");
  int st = readRegB(REG_STATUS), agc = readRegB(REG_AGC);
  if (st >= 0) {
    bool MD = st & 0x20, ML = st & 0x10, MH = st & 0x08;
    Serial.print("    magnet: ");
    if (!MD)     Serial.println("NOT DETECTED - fit or reposition");
    else if (ML) Serial.println("too weak - move closer");
    else if (MH) Serial.println("too strong - move further away");
    else         Serial.println("detected, strength OK");
    Serial.print("    AGC "); Serial.println(agc);
  }
}

int readRegB(uint8_t reg) {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return -1;
  Wire.requestFrom(AS5600_ADDR, 1);
  if (Wire.available()) return Wire.read();
  return -1;
}

void updateSensorB() {
  if (!sensorB_present) return;
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(REG_ANGLE);
  if (Wire.endTransmission(false) != 0) { sensorB_deg = -1; return; }
  Wire.requestFrom(AS5600_ADDR, 2);
  if (Wire.available() == 2) {
    int hi = Wire.read(), lo = Wire.read();
    float deg = (((hi << 8) | lo) & 0x0FFF) * 360.0f / 4096.0f;
    sensorB_deg = deg;
    unsigned long us = micros();
    if (!sensorB_started) {
      sensorB_prev = deg; sensorB_cont = deg; sensorB_started = true; pivotPrevUs = us;
    } else {
      float dd = deg - sensorB_prev;
      if (dd >  180.0f) dd -= 360.0f;
      if (dd < -180.0f) dd += 360.0f;
      sensorB_prev = deg;
      sensorB_cont += dd;
      float dtp = (us - pivotPrevUs) / 1000000.0f;
      if (dtp > 1e-5f) {
        // light filter only - the identification scripts differentiate the RAW
        // angle column themselves. Never feed this into a state estimator
        // (handoff section 4.4).
        float raw = dd / dtp;
        pivotVel = 0.85f * pivotVel + 0.15f * raw;
        pivotPrevUs = us;
      }
    }
  }
}

// ---------------------------------------------------------------- motor

float wrap360(float a) {
  while (a < 0)       a += 360.0f;
  while (a >= 360.0f) a -= 360.0f;
  return a;
}

void writePhases(float elecDeg, int duty) {
  if (duty < 0) duty = 0;
  if (duty > HARD_DUTY_CEILING) duty = HARD_DUTY_CEILING;
  float r = elecDeg * PI / 180.0f;
  analogWrite(PIN_IN1, (int)((sinf(r) + 1.0f) * 0.5f * duty));
  analogWrite(PIN_IN2, (int)((sinf(r + 2.0f * PI / 3.0f) + 1.0f) * 0.5f * duty));
  analogWrite(PIN_IN3, (int)((sinf(r + 4.0f * PI / 3.0f) + 1.0f) * 0.5f * duty));
}

void killOutputs() {
  analogWrite(PIN_IN1, 0); analogWrite(PIN_IN2, 0); analogWrite(PIN_IN3, 0);
  digitalWrite(PIN_EN, LOW);
}

void stop(const char* why) {
  killOutputs();
  armed = false; mode = IDLE;
  dutyRequest = dutyActual = viTerm = 0; lastCmd = 0;
  if (logKind != LOG_NONE) { Serial.println("# ABORT"); logKind = LOG_NONE; }
  Serial.print("\n  *** STOPPED: "); Serial.println(why);
  Serial.println("  Send 'r' to re-arm.\n");
}

bool selfCalibrate() {
  Serial.println("  Aligning rotor...");
  digitalWrite(PIN_EN, HIGH);
  for (int dd = 0; dd <= ALIGN_DUTY; dd += 3) {
    if (FAULT_WIRED && digitalRead(PIN_FAULT) == LOW) {
      killOutputs(); Serial.println("  FAULT during alignment."); return false;
    }
    writePhases(0.0f, dd); delay(12);
  }
  delay(600);
  Serial.println("  Self-calibrating encoder A (one mechanical revolution)...");
  int lo = ADC_MAX + 1, hi = -1;
  int totalSteps = 360 * POLE_PAIRS / SWEEP_STEP;
  int prevRaw = analogRead(PIN_SENSOR_OUT);
  float net = 0;
  for (int i = 0; i <= totalSteps; i++) {
    if (FAULT_WIRED && digitalRead(PIN_FAULT) == LOW) {
      killOutputs(); Serial.println("  FAULT during sweep."); return false;
    }
    writePhases((float)((i * SWEEP_STEP) % 360), SWEEP_DUTY);
    delay(SWEEP_DELAY);
    int raw = analogRead(PIN_SENSOR_OUT);
    if (raw < lo) lo = raw;
    if (raw > hi) hi = raw;
    int dRaw = raw - prevRaw;
    if (dRaw >  ADC_MAX / 2) dRaw -= ADC_MAX;
    if (dRaw < -ADC_MAX / 2) dRaw += ADC_MAX;
    net += dRaw; prevRaw = raw;
    if (i % (totalSteps / 8) == 0) Serial.print(".");
  }
  Serial.println();
  killOutputs();
  int span = hi - lo;
  Serial.print("  raw span "); Serial.println(span);
  if (span < MIN_VALID_SPAN) {
    Serial.println("  *** Sweep did not cover a full revolution.");
    Serial.println("  Loose magnet, an obstruction, or SWEEP_DUTY too low.");
    return false;
  }
  calMin = lo; calMax = hi; calibrated = true;
  sensorDir = (net > 0) ? 1 : -1;
  Serial.print("  encoder A direction "); Serial.println(sensorDir);
  return true;
}

bool arm() {
  if (!selfCalibrate()) return false;
  float a = readAngleA();
  if (a < 0) { killOutputs(); Serial.println("  Encoder A read failed."); return false; }
  contAngle = a; prevAngle = a; offsetDeg = a;
  velocity = 0; dutyRequest = dutyActual = viTerm = 0; lastCmd = 0;
  mode = TORQUE;
  digitalWrite(PIN_EN, HIGH);
  armed = true;
  sessionStart = thermalStart = movingSince = millis();
  lastMicros = micros();
  dutyAccum = dutyAvg = 0;
  Serial.println("  ARMED.\n");
  return true;
}

// ---------------------------------------------------------------- serial

void printHelp() {
  Serial.println();
  Serial.println("  --- no motor power needed ---");
  Serial.println("  n [s]          stage 0: sensor noise floor");
  Serial.println("  w [s]          stage 1: free-swing log (hang, displace, release)");
  Serial.println("  o              zero the pivot angle here");
  Serial.println("  --- motor ---");
  Serial.println("  r              arm / self-calibrate");
  Serial.println("  k <duty>       stage 2: spin-up step     [CLAMP THE ARM]");
  Serial.println("  d <duty>       stage 2: coast-down       [CLAMP THE ARM]");
  Serial.println("  t <duty> <ms>  stage 3: torque pulse");
  Serial.println("  e <gain> <deg> stage 3: resonant swing, abort above <deg>");
  Serial.println("  c <duty>  cap    z  flip A dir    h  coast    x  disarm    s  status");
  Serial.println();
}

void printStatus() {
  Serial.println();
  Serial.print("  armed ");  Serial.println(armed ? "YES" : "no");
  Serial.print("  encoder A ");
  if (calibrated) { Serial.print("cal "); Serial.print(calMin); Serial.print("..");
                    Serial.print(calMax); Serial.print(" dir "); Serial.println(sensorDir); }
  else Serial.println("NOT calibrated - send 'r'");
  Serial.print("  pivot B ");
  if (sensorB_present) { Serial.print(pivotAngle(), 2); Serial.println(" deg"); }
  else Serial.println("NOT FOUND");
  Serial.print("  duty cap "); Serial.print(dutyCap);
  Serial.print("/"); Serial.println(HARD_DUTY_CEILING);
  Serial.print("  logging "); Serial.println(logName(logKind));
  Serial.println();
}

// Accepts CR, LF or CRLF. PuTTY sends CR on Enter, the Arduino Serial Monitor
// sends LF. Accepting only LF made 'x' (disarm) unreachable from PuTTY, which
// matters because that is the stop command.
void handleSerial() {
  static char buf[48];
  static int idx = 0;
  static bool lastWasCR = false;
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r' || ch == '\n') {
      if (ch == '\n' && lastWasCR) { lastWasCR = false; continue; }
      lastWasCR = (ch == '\r');
      buf[idx] = 0;
      if (idx > 0) command(buf);
      idx = 0;
      continue;
    }
    lastWasCR = false;
    if (ch == 8 || ch == 127) { if (idx > 0) idx--; continue; }   // backspace
    if (idx < (int)sizeof(buf) - 1) buf[idx++] = ch;
  }
}

void command(char* c) {
  while (*c == ' ') c++;

  if (c[0] == 'x') {
    killOutputs(); armed = false; mode = IDLE; logKind = LOG_NONE;
    dutyRequest = dutyActual = 0; lastCmd = 0;
    Serial.println("  Disarmed."); return;
  }
  if (c[0] == 's' && c[1] == 0) { printStatus(); return; }
  if (c[0] == '?') { printHelp(); return; }
  if (c[0] == 'o' && c[1] == 0) {
    pivotZero = sensorB_cont;
    Serial.println("  pivot zeroed here."); return;
  }
  if (c[0] == 'r' && c[1] == 0) {
    if (!armed) arm(); else Serial.println("  Already armed.");
    return;
  }
  if (c[0] == 'h' && c[1] == 0) {
    mode = COAST; dutyRequest = 0;
    Serial.println("  Coasting."); return;
  }
  if (c[0] == 'z' && c[1] == 0) {
    sensorDir = -sensorDir; dutyRequest = dutyActual = 0; mode = TORQUE;
    Serial.print("  encoder A direction "); Serial.println(sensorDir); return;
  }

  float v1 = atof(c + 1);
  char* sp = strchr(c + 1, ' ');
  float v2 = sp ? atof(sp + 1) : 0;

  switch (c[0]) {
    case 'n':
      if (!sensorB_present) { Serial.println("  pivot encoder missing."); break; }
      killOutputs(); armed = false;
      Serial.println("  STAGE 0: hold everything STILL.");
      startLog(LOG_NOISE, v1 > 0 ? (unsigned long)v1 : 10);
      break;

    case 'w':
      if (!sensorB_present) { Serial.println("  pivot encoder missing."); break; }
      killOutputs(); armed = false;
      pivotZero = sensorB_cont;   // zero at the hanging rest position
      Serial.println("  STAGE 1: motor OFF. Displace ~20-30 deg and RELEASE.");
      startLog(LOG_SWING, v1 > 0 ? (unsigned long)v1 : 20);
      break;

    case 'k':
      if (!armed) { Serial.println("  Arm first with 'r'."); break; }
      Serial.println("  STAGE 2 spin-up. IS THE ARM CLAMPED?");
      mode = TORQUE; dutyRequest = v1; movingSince = millis();
      startLog(LOG_SPIN, 5);
      break;

    case 'd':
      if (!armed) { Serial.println("  Arm first with 'r'."); break; }
      Serial.println("  STAGE 2 coast-down: spinning up, then releasing.");
      mode = TORQUE; dutyRequest = v1; movingSince = millis();
      delay(2500);                       // reach steady state, then coast
      mode = COAST; dutyRequest = 0;
      startLog(LOG_SPIN, 8);
      break;

    case 't':
      if (!armed) { Serial.println("  Arm first with 'r'."); break; }
      mode = PULSE; dutyRequest = v1;
      pulseUntil = millis() + (unsigned long)(v2 > 0 ? v2 : 200);
      movingSince = millis();
      Serial.print("  STAGE 3 pulse: duty "); Serial.print((int)v1);
      Serial.print(" for "); Serial.print((int)(v2 > 0 ? v2 : 200)); Serial.println(" ms");
      startLog(LOG_PULSE, 6);
      break;

    case 'e':
      if (!armed) { Serial.println("  Arm first with 'r'."); break; }
      pumpGain     = (v1 > 0) ? v1 : 8.0f;
      pumpAbortDeg = (v2 > 0) ? v2 : 15.0f;
      if (pumpAbortDeg > 45.0f) { pumpAbortDeg = 45.0f; Serial.println("  cap clamped to 45 deg."); }
      mode = PUMP; movingSince = millis();
      Serial.print("  STAGE 3 resonant swing: gain "); Serial.print(pumpGain, 1);
      Serial.print(", abort above "); Serial.print(pumpAbortDeg, 0); Serial.println(" deg. KEEP CLEAR.");
      startLog(LOG_PUMP, 20);
      break;

    case 'c':
      if (v1 < 0) v1 = 0;
      if (v1 > HARD_DUTY_CEILING) { v1 = HARD_DUTY_CEILING; Serial.println("  Clamped."); }
      dutyCap = (int)v1;
      Serial.print("  duty cap "); Serial.println(dutyCap);
      break;

    default:
      killOutputs(); armed = false; mode = IDLE; logKind = LOG_NONE;
      dutyRequest = dutyActual = 0;
      Serial.println("  Unrecognised - disarmed as a precaution.");
      break;
  }
}
