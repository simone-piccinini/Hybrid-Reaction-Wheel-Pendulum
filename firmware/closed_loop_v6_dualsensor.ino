// CLOSED-LOOP MOTOR CONTROL  -  v6  (DUAL SENSOR + FEEDFORWARD)
// Motor:   iPower GBM2804H-100T gimbal motor, 12N14P (7 pole pairs)
// Sensor A: AS5600 ANALOG on pin 23 (A9)   - wheel/rotor, drives commutation
// Sensor B: AS5600 I2C on pins 18/19       - second axis, read alongside
//
// WHY TWO SENSORS ON DIFFERENT INTERFACES
//   Every AS5600 is fixed at I2C address 0x36, so two of them can never
//   share a bus. Running one on analog and one on I2C is the way round
//   that. It is also the arrangement the pendulum needs: the wheel angle
//   for commutation, the pivot angle for the balancing state.
//
// WHAT IS NEW IN v6
//   1. Sensor B is read and reported alongside A, so you can confirm
//      both track correctly while the motor runs.
//
//   2. VELOCITY FEEDFORWARD. In v5 the PI loop had to generate the
//      entire duty command, and at 90 dps that meant working down at
//      duty 15-60 where cogging torque is a large fraction of what is
//      being asked for. The loop surged between 57 and 132 dps chasing
//      its own ripple.
//
//      Your f 150 run gives the plant gain directly: duty 150 produced
//      about 1000 dps steady, so duty ~= 0.15 * dps. Feeding that in
//      directly means the PI loop only has to trim a small correction
//      instead of building the whole command, which is far more stable
//      at low speed.
//
//   3. Integral gain lowered. The velocity signal is filtered with a
//      ~50ms time constant, which puts real phase lag in the feedback
//      path, and VKI 0.40 was far too much gain to sit behind that lag.
//
// WIRING
//   Driver:   GND->GND, IN1->2, IN2->3, IN3->4, EN->5, FAULT->8
//             RESET and SLEEP tied to the driver's own 3.3V
//   Sensor A: VCC->3.3V, GND->GND, DIR->GND, GPO->FLOATING, OUT->pin 23
//   Sensor B: VCC->3.3V, GND->GND, DIR->GND, GPO->FLOATING
//             SDA->pin 18, SCL->pin 19
//
//   Sensor B is optional. If it does not respond the sketch says so and
//   carries on - the motor only depends on sensor A.
//
// COMMANDS (115200 baud)
//   r          arm - self-calibrates sensor A, takes ~8 seconds
//   f <duty>   torque mode, e.g. f 150
//   v <dps>    velocity mode with feedforward, e.g. v 90
//   h          coast to a stop
//   z          flip sensor A direction
//   c <duty>   duty cap
//   p <val>    velocity KP    i <val>  velocity KI    g <val>  feedforward gain
//   b          re-scan for sensor B
//   s          status
//   x          stop / disarm

#include <math.h>
#include <Wire.h>

#define USE_I2C false      // sensor A (commutation) is analog, always

// ---- Sensor A: analog ----
const int PIN_SENSOR_OUT = 23;              // A9
const int ADC_BITS = 12;
const int ADC_MAX  = (1 << ADC_BITS) - 1;

// ---- Sensor B: I2C ----
#define AS5600_ADDR   0x36
#define REG_ANGLE     0x0E
#define REG_STATUS    0x0B
#define REG_AGC       0x1A
bool sensorB_present = false;
float sensorB_deg = -1;
float sensorB_cont = 0, sensorB_prev = 0;
bool  sensorB_started = false;

// ---- Driver pins ----
const int PIN_IN1   = 2;
const int PIN_IN2   = 3;
const int PIN_IN3   = 4;
const int PIN_EN    = 5;
const int PIN_FAULT = 8;
const bool FAULT_WIRED = true;

// ---- Motor ----
const int POLE_PAIRS = 7;

// ---- PWM ----
const int PWM_BITS = 10;
const int PWM_MAX  = (1 << PWM_BITS) - 1;

const unsigned long CONTROL_PERIOD_US = 1000;   // 1 kHz

// ---- Safety ----
const int HARD_DUTY_CEILING = 450;
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
const int ALIGN_DUTY   = 130;
const int SWEEP_DUTY   = 170;
const int SWEEP_STEP   = 2;
const int SWEEP_DELAY  = 6;
const int MIN_VALID_SPAN = 2500;

const float DUTY_RAMP_PER_SEC = 400.0f;
const float VEL_ALPHA = 0.02f;

// ---- Velocity loop ----
// Feedforward does most of the work now, so the PI gains are much
// smaller than in v5. FF_GAIN comes from the measured plant: duty 150
// gave about 1000 dps, so 150/1000 = 0.15.
float FF_GAIN = 0.15f;
float VKP = 0.12f;
float VKI = 0.06f;
const float VI_LIMIT = 120.0f;

// ---- State ----
enum Mode { IDLE, TORQUE, VELOCITY };
Mode mode = IDLE;

bool armed = false, calibrated = false;
int  calMin = -1, calMax = -1;
int  sensorDir = 1;

float offsetDeg = 0, contAngle = 0, prevAngle = 0, velocity = 0;
float dutyRequest = 0, dutyActual = 0, speedTarget = 0, viTerm = 0;

unsigned long sessionStart = 0, lastMicros = 0, thermalStart = 0;
float dutyAccum = 0, dutyAvg = 0;
unsigned long movingSince = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000) {}

  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
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
  Wire.setClock(400000);

  Serial.println();
  Serial.println("=====================================");
  Serial.println("  CLOSED-LOOP v6 - DUAL SENSOR");
  Serial.println("=====================================");
  Serial.println("  A: analog pin 23  (drives commutation)");
  Serial.println("  B: I2C 18/19      (read alongside)");
  Serial.println();

  scanSensorB();

  Serial.println("  Shaft must be FREE TO TURN, then send 'r'.");
  Serial.println();
  printHelp();
}

void loop() {
  handleSerial();

  // Sensor B is read whether armed or not, so you can watch it move.
  static unsigned long lastB = 0;
  if (millis() - lastB >= 20) { lastB = millis(); updateSensorB(); }

  if (!armed) {
    killOutputs();
    static unsigned long lastIdle = 0;
    if (millis() - lastIdle >= 1000) {
      lastIdle = millis();
      if (sensorB_present) {
        Serial.print("  idle   B ");
        Serial.print(sensorB_deg, 1);
        Serial.print(" deg  (cont ");
        Serial.print(sensorB_cont, 0);
        Serial.println(")");
      }
    }
    delay(2);
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

  if (mode == VELOCITY) {
    // ---- feedforward + PI trim ----
    // The feedforward supplies most of the duty from the known plant
    // gain, so the integrator only has to correct a small residual.
    // That is what stops the surging seen in v5 at low speed.
    float ff = FF_GAIN * speedTarget;
    float verr = speedTarget - vSigned;
    viTerm += VKI * verr * dt;
    if (viTerm >  VI_LIMIT) viTerm =  VI_LIMIT;
    if (viTerm < -VI_LIMIT) viTerm = -VI_LIMIT;
    dutyRequest = ff + VKP * verr + viTerm;
    if (fabsf(speedTarget) < 1.0f && fabsf(vSigned) < 5.0f) { dutyRequest = 0; viTerm = 0; }
  }

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

  if (fabsf(velocity) > STALL_DPS_MAX || fabsf(cmd) < STALL_DUTY_MIN) movingSince = now;
  else if (now - movingSince > STALL_TIMEOUT_MS) { stop("stalled - try 'z' then 'f' again"); return; }

  dutyAccum += fabsf(cmd) * dt;
  if (now - thermalStart >= THERMAL_WINDOW_MS) {
    dutyAvg = dutyAccum / (THERMAL_WINDOW_MS / 1000.0f);
    dutyAccum = 0;
    thermalStart = now;
    if (dutyAvg > DUTY_BUDGET_CUT) { stop("thermal budget - let it cool"); return; }
  }

  float elec = wrap360(sensorDir * pos * POLE_PAIRS);
  writePhases(wrap360(elec + (cmd >= 0 ? 90.0f : -90.0f)), (int)fabsf(cmd));

  // ---- telemetry: both sensors ----
  static unsigned long lastPrint = 0;
  if (now - lastPrint >= 400) {
    lastPrint = now;
    Serial.print("  ");
    Serial.print(mode == TORQUE ? "TORQ" : (mode == VELOCITY ? "VEL " : "idle"));
    Serial.print("  A ");    Serial.print(vSigned, 0);
    if (mode == VELOCITY) { Serial.print("/"); Serial.print(speedTarget, 0); }
    Serial.print(" dps");
    Serial.print("  duty ");  Serial.print((int)cmd);
    Serial.print("/");        Serial.print(cap);
    if (sensorB_present) {
      Serial.print("   B ");  Serial.print(sensorB_deg, 1);
      Serial.print(" deg");
    }
    Serial.print("  avg ");   Serial.println(dutyAvg, 0);
  }
}

// ---------------------------------------------------------------
// sensor A - analog, drives commutation
// ---------------------------------------------------------------

float readAngleA() {
  if (!calibrated) return -1;
  int raw = analogRead(PIN_SENSOR_OUT);
  int span = calMax - calMin;
  if (span < 500) return -1;
  if (raw < calMin - span / 8 || raw > calMax + span / 8) return -1;
  float dg = (raw - calMin) * 360.0f / span;
  if (dg < 0) dg = 0;
  if (dg > 359.99f) dg = 359.99f;
  return dg;
}

// ---------------------------------------------------------------
// sensor B - I2C, read only
// ---------------------------------------------------------------

void scanSensorB() {
  Wire.beginTransmission(AS5600_ADDR);
  sensorB_present = (Wire.endTransmission() == 0);

  Serial.print("  sensor B (I2C): ");
  if (!sensorB_present) {
    Serial.println("NOT FOUND");
    Serial.println("    check SDA=18, SCL=19, 3.3V, shared GND,");
    Serial.println("    and that the board has pull-up resistors.");
    Serial.println("    The motor does not depend on B, so this is");
    Serial.println("    not fatal - 'b' rescans.");
    return;
  }
  Serial.println("found at 0x36");

  int st = readRegB(REG_STATUS);
  int agc = readRegB(REG_AGC);
  if (st >= 0) {
    bool MD = st & 0x20, ML = st & 0x10, MH = st & 0x08;
    Serial.print("    magnet: ");
    if (!MD)      Serial.println("NOT DETECTED - fit or reposition it");
    else if (ML)  Serial.println("too weak - move closer");
    else if (MH)  Serial.println("too strong - move further away");
    else          Serial.println("detected, strength OK");
    Serial.print("    AGC "); Serial.println(agc);
  }
  Serial.println();
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
    if (!sensorB_started) { sensorB_prev = deg; sensorB_cont = deg; sensorB_started = true; }
    else {
      float dd = deg - sensorB_prev;
      if (dd >  180.0f) dd -= 360.0f;
      if (dd < -180.0f) dd += 360.0f;
      sensorB_prev = deg;
      sensorB_cont += dd;
    }
  }
}

// ---------------------------------------------------------------

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
  analogWrite(PIN_IN1, 0);
  analogWrite(PIN_IN2, 0);
  analogWrite(PIN_IN3, 0);
  digitalWrite(PIN_EN, LOW);
}

void stop(const char* why) {
  killOutputs();
  armed = false; mode = IDLE;
  dutyRequest = dutyActual = speedTarget = viTerm = 0;
  Serial.println();
  Serial.print("  *** STOPPED: ");
  Serial.println(why);
  Serial.println("  Send 'r' to re-arm.");
  Serial.println();
}

bool selfCalibrate() {
  Serial.println("  Aligning rotor...");
  digitalWrite(PIN_EN, HIGH);
  for (int dd = 0; dd <= ALIGN_DUTY; dd += 3) {
    if (FAULT_WIRED && digitalRead(PIN_FAULT) == LOW) {
      killOutputs(); Serial.println("  FAULT during alignment."); return false;
    }
    writePhases(0.0f, dd);
    delay(12);
  }
  delay(600);

  Serial.print("  Self-calibrating sensor A: one full revolution (");
  Serial.print(POLE_PAIRS);
  Serial.println(" electrical revs)...");

  int lo = ADC_MAX + 1, hi = -1;
  int totalSteps = 360 * POLE_PAIRS / SWEEP_STEP;
  int prevRaw = analogRead(PIN_SENSOR_OUT);
  float netProvisional = 0;

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
    netProvisional += dRaw;
    prevRaw = raw;

    if (i % (totalSteps / 8) == 0) Serial.print(".");
  }
  Serial.println();
  killOutputs();

  int span = hi - lo;
  Serial.print("  raw range ");
  Serial.print(lo); Serial.print(" .. "); Serial.print(hi);
  Serial.print("   span "); Serial.println(span);

  if (span < MIN_VALID_SPAN) {
    Serial.println("  *** Sweep did not cover a full revolution.");
    Serial.println("  The rotor did not follow: loose magnet, an");
    Serial.println("  obstruction, or SWEEP_DUTY too low for the load.");
    Serial.println();
    return false;
  }

  calMin = lo; calMax = hi; calibrated = true;
  sensorDir = (netProvisional > 0) ? 1 : -1;

  Serial.print("  sensor A direction ");
  Serial.println(sensorDir > 0 ? "+1" : "-1");
  Serial.println();
  return true;
}

bool arm() {
  if (!selfCalibrate()) return false;

  float a = readAngleA();
  if (a < 0) { killOutputs(); Serial.println("  Sensor A read failed."); return false; }

  contAngle = a; prevAngle = a; offsetDeg = a;
  velocity = 0;
  dutyRequest = dutyActual = speedTarget = viTerm = 0;
  mode = IDLE;

  digitalWrite(PIN_EN, HIGH);
  armed = true;
  sessionStart = thermalStart = movingSince = millis();
  lastMicros = micros();
  dutyAccum = dutyAvg = 0;

  Serial.println("  ARMED. 'f 150' for torque, 'v 90' for speed.");
  Serial.println();
  return true;
}

// ---------------------------------------------------------------

void printHelp() {
  Serial.println("  r = arm (self-calibrates)     x = stop");
  Serial.println("  f <duty> = torque             v <dps> = speed");
  Serial.println("  h = coast   z = flip A dir    c <duty> = cap");
  Serial.println("  p <vkp>  i <vki>  g <ff>      b = rescan B   s = status");
  Serial.println();
}

void printStatus() {
  Serial.println();
  Serial.print("  armed ");  Serial.print(armed ? "YES" : "no");
  Serial.print("   mode ");
  Serial.println(mode == TORQUE ? "torque" : (mode == VELOCITY ? "velocity" : "idle"));
  Serial.print("  A calibrated ");
  if (calibrated) { Serial.print(calMin); Serial.print(".."); Serial.print(calMax);
                    Serial.print("  dir "); Serial.println(sensorDir); }
  else Serial.println("NO - send 'r'");
  Serial.print("  B ");
  if (sensorB_present) { Serial.print("present, "); Serial.print(sensorB_deg, 1); Serial.println(" deg"); }
  else Serial.println("not found");
  Serial.print("  duty cap "); Serial.print(dutyCap);
  Serial.print("/");           Serial.println(HARD_DUTY_CEILING);
  Serial.print("  FF ");       Serial.print(FF_GAIN, 3);
  Serial.print("   VKP ");     Serial.print(VKP, 3);
  Serial.print("   VKI ");     Serial.println(VKI, 3);
  Serial.println();
}

void handleSerial() {
  static char buf[40];
  static int idx = 0;
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n' || idx >= (int)sizeof(buf) - 1) {
      buf[idx] = 0;
      if (idx > 0) command(buf);
      idx = 0;
    } else buf[idx++] = ch;
  }
}

void command(char* c) {
  while (*c == ' ') c++;

  if (c[0] == 'x') {
    killOutputs(); armed = false; mode = IDLE;
    dutyRequest = dutyActual = speedTarget = viTerm = 0;
    Serial.println("  Disarmed."); return;
  }
  if (c[0] == 's') { printStatus(); return; }
  if (c[0] == 'b' && c[1] == 0) { scanSensorB(); return; }
  if (c[0] == 'r' && c[1] == 0) {
    if (!armed) arm(); else Serial.println("  Already armed.");
    return;
  }
  if (c[0] == 'h' && c[1] == 0) {
    mode = IDLE; dutyRequest = speedTarget = viTerm = 0;
    Serial.println("  Coasting."); return;
  }
  if (c[0] == 'z' && c[1] == 0) {
    sensorDir = -sensorDir;
    dutyRequest = dutyActual = viTerm = 0; mode = IDLE;
    Serial.print("  A direction flipped to "); Serial.println(sensorDir);
    return;
  }

  float val = atof(c + 1);
  switch (c[0]) {
    case 'f':
      if (!armed) { Serial.println("  Arm first with 'r'."); break; }
      mode = TORQUE; viTerm = 0; dutyRequest = val; movingSince = millis();
      Serial.print("  torque mode, duty "); Serial.println((int)val);
      break;
    case 'v':
      if (!armed) { Serial.println("  Arm first with 'r'."); break; }
      mode = VELOCITY; speedTarget = val; movingSince = millis();
      Serial.print("  velocity mode, target "); Serial.print(val, 0);
      Serial.print(" dps  (feedforward duty ");
      Serial.print(FF_GAIN * val, 0);
      Serial.println(")");
      break;
    case 'c':
      if (val < 0) val = 0;
      if (val > HARD_DUTY_CEILING) { val = HARD_DUTY_CEILING; Serial.println("  Clamped."); }
      dutyCap = (int)val;
      Serial.print("  duty cap "); Serial.println(dutyCap);
      break;
    case 'p': VKP = val; Serial.print("  VKP "); Serial.println(VKP, 3); break;
    case 'i': VKI = val; Serial.print("  VKI "); Serial.println(VKI, 3); break;
    case 'g': FF_GAIN = val; Serial.print("  FF gain "); Serial.println(FF_GAIN, 3); break;
    default:
      killOutputs(); armed = false; mode = IDLE;
      dutyRequest = dutyActual = speedTarget = viTerm = 0;
      Serial.println("  Unrecognised - disarmed as a precaution.");
      break;
  }
}
