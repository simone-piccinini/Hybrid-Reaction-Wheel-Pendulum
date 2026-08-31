// Y-WOBBLE TEST  -  v1
// Reaction-wheel pendulum, pivot encoder (AS5600 on I2C, pins 18/19)
//
// PURPOSE
//   Decide whether the pivot encoder's error is a function of ONE variable
//   (the swing angle) or TWO (swing angle AND out-of-plane tilt).
//
//   The 4-point linearity check showed the reading is badly distorted but
//   REPEATABLE at a held position (systematic error up to 80 deg, scatter only
//   1-7 deg). If that holds, the distortion is deterministic and a calibration
//   map inverts it. If instead the reading swings wildly when the arm is tilted
//   out of plane at a FIXED swing angle, then one measurement cannot recover
//   the true angle and no lookup table can help.
//
//   This sketch measures exactly that: hold the swing angle fixed, push the arm
//   through its full y play, and see how much the reading moves.
//
// THE MOTOR IS NEVER ENABLED. There is no commutation code in this file at all;
// the driver pins are driven low once at boot and never touched again.
//
// METHOD - read this before running
//   The swing angle MUST NOT MOVE during a capture. If the arm also rotates in
//   the swing plane, the reading changes for that reason instead and the result
//   is worthless. Rest the arm against a fixed stop, then push it only OUT OF
//   PLANE (the y direction), through the full slop available.
//
//   For each position, take TWO captures:
//     q <deg>   QUIET baseline - do not touch the arm at all
//     g <deg>   WOBBLE         - push through the full y play, back and forth
//   The difference between the two spreads is the y-tilt contribution.
//
//   Do it at several swing angles - at minimum near hanging (0), near
//   horizontal (90), and near upright (180), since sensitivity varies around
//   the circle.
//
// COMMANDS (115200 baud)
//   o            zero the pivot angle here
//   q <deg>      start a QUIET capture, labelled with the true swing angle
//   g <deg>      start a WOBBLE capture, labelled with the true swing angle
//   <enter>      any input while capturing STOPS it and prints the summary
//   s            status / magnet health
//   b            re-scan the encoder
//   ?            help
//
// READING THE RESULT
//   The summary prints the peak-to-peak SPREAD of the reading. Compare the
//   wobble capture against the quiet one at the same angle:
//     spread < ~3 deg   -> y-tilt is second order; a calibration map will work
//     3 - 10 deg        -> marginal; usable only with an inflated Kalman R
//     > ~10 deg         -> two-variable problem; a map cannot fix it

#include <Wire.h>
#include <math.h>

// ---- pivot encoder (I2C) ----
#define AS5600_ADDR 0x36
#define REG_ANGLE   0x0E
#define REG_STATUS  0x0B
#define REG_AGC     0x1A

// ---- wheel encoder (analog) - logged for reference only ----
const int PIN_SENSOR_OUT = 23;
const int ADC_BITS = 12;
const float NOMINAL_SPAN_COUNTS = 4062.0f;   // uncalibrated scale, reference only

// ---- driver pins: forced safe, never driven ----
const int PIN_IN1 = 2, PIN_IN2 = 3, PIN_IN3 = 4, PIN_EN = 5;

const unsigned long SAMPLE_US   = 5000;      // 200 Hz logging
const unsigned long MAX_CAPTURE_MS = 60000;  // safety stop

bool  encoderPresent = false;
float pivotDeg = 0, pivotPrev = 0, pivotCont = 0, pivotZero = 0;
bool  pivotStarted = false;

bool capturing = false;
bool captureIsWobble = false;
float nominalDeg = 0;
unsigned long capStart = 0, lastSample = 0;
long  nSamples = 0;
float vMin = 0, vMax = 0, vSum = 0, vSumSq = 0;

// ---------------------------------------------------------------- setup

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000) {}

  // driver held inert for the whole sketch
  pinMode(PIN_IN1, OUTPUT); pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT); pinMode(PIN_EN, OUTPUT);
  digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW); digitalWrite(PIN_EN, LOW);

  analogReadResolution(ADC_BITS);
  analogReadAveraging(8);
  pinMode(PIN_SENSOR_OUT, INPUT_DISABLE);

  Wire.begin();
  Wire.setClock(1000000);

  Serial.println();
  Serial.println("=====================================");
  Serial.println("  Y-WOBBLE TEST  v1   (motor never enabled)");
  Serial.println("=====================================");
  scanEncoder();
  Serial.println();
  Serial.println("  Pin the SWING angle against a stop.");
  Serial.println("  Then push the arm only OUT OF PLANE.");
  printHelp();
}

// ---------------------------------------------------------------- loop

void loop() {
  handleSerial();

  static unsigned long lastRead = 0;
  if (micros() - lastRead >= 1000) { lastRead = micros(); updateEncoder(); }

  if (capturing) {
    if (millis() - capStart > MAX_CAPTURE_MS) {
      Serial.println("# auto-stop: 60 s limit");
      endCapture();
      return;
    }
    unsigned long now = micros();
    if (now - lastSample >= SAMPLE_US) {
      lastSample = now;
      float a = pivotAngle();
      if (nSamples == 0) { vMin = vMax = a; }
      if (a < vMin) vMin = a;
      if (a > vMax) vMax = a;
      vSum += a; vSumSq += a * a;
      nSamples++;
      Serial.print((millis() - capStart) / 1000.0f, 4); Serial.print(',');
      Serial.print(a, 3);                                Serial.print(',');
      Serial.println(rawWheelDeg(), 3);
    }
  } else {
    static unsigned long lastIdle = 0;
    if (millis() - lastIdle >= 1000) {
      lastIdle = millis();
      Serial.print("  idle   pivot ");
      Serial.print(pivotAngle(), 2);
      Serial.println(" deg");
    }
  }
}

// ---------------------------------------------------------------- capture

void startCapture(bool wobble, float deg) {
  if (!encoderPresent) { Serial.println("  encoder not found - fix that first."); return; }
  capturing = true; captureIsWobble = wobble; nominalDeg = deg;
  capStart = millis(); lastSample = micros();
  nSamples = 0; vSum = vSumSq = 0; vMin = vMax = 0;
  Serial.println();
  Serial.print("# TEST,");        Serial.println(wobble ? "ywobble" : "yquiet");
  Serial.print("# NOMINAL_DEG,"); Serial.println(deg, 1);
  Serial.println(wobble ? "# NOW: push the arm through its full y play."
                        : "# NOW: do not touch the arm.");
  Serial.println("# send any line to stop");
  Serial.println("t_s,pivot_deg,wheel_deg");
}

void endCapture() {
  capturing = false;
  if (nSamples < 2) { Serial.println("# too few samples"); return; }
  float mean = vSum / nSamples;
  float var  = vSumSq / nSamples - mean * mean;
  if (var < 0) var = 0;
  float sd = sqrtf(var);
  float spread = vMax - vMin;

  Serial.println("# END");
  Serial.println("#");
  Serial.print("# summary,");
  Serial.print(captureIsWobble ? "ywobble" : "yquiet");
  Serial.print(",nominal_deg,");  Serial.print(nominalDeg, 1);
  Serial.print(",n,");            Serial.print(nSamples);
  Serial.print(",mean,");         Serial.print(mean, 3);
  Serial.print(",min,");          Serial.print(vMin, 3);
  Serial.print(",max,");          Serial.print(vMax, 3);
  Serial.print(",spread,");       Serial.print(spread, 3);
  Serial.print(",std,");          Serial.println(sd, 4);
  Serial.println("#");
  Serial.print("  SPREAD ");
  Serial.print(spread, 2);
  Serial.print(" deg   (std ");
  Serial.print(sd, 3);
  Serial.print(" deg, ");
  Serial.print(nSamples);
  Serial.println(" samples)");

  if (captureIsWobble) {
    if (spread < 3.0f) {
      Serial.println("  -> y-tilt is SECOND ORDER. A calibration map should work.");
    } else if (spread < 10.0f) {
      Serial.println("  -> MARGINAL. Map may work, but the residual must go into");
      Serial.println("     the Kalman measurement covariance as real uncertainty.");
    } else {
      Serial.println("  -> TWO-VARIABLE PROBLEM. One reading cannot recover the");
      Serial.println("     true angle; a lookup table will not fix this.");
    }
    Serial.println("  Compare against the 'q' quiet capture at the same angle:");
    Serial.println("  the DIFFERENCE is what the y-tilt actually costs you.");
  }
  Serial.println();
}

// ---------------------------------------------------------------- encoder

float pivotAngle() { return pivotCont - pivotZero; }

float rawWheelDeg() {
  return analogRead(PIN_SENSOR_OUT) * 360.0f / NOMINAL_SPAN_COUNTS;
}

void scanEncoder() {
  Wire.beginTransmission(AS5600_ADDR);
  encoderPresent = (Wire.endTransmission() == 0);
  Serial.print("  pivot encoder (I2C): ");
  if (!encoderPresent) {
    Serial.println("NOT FOUND - check SDA 18, SCL 19, 3.3 V, GND, pull-ups.");
    return;
  }
  Serial.println("found at 0x36");
  int st = readReg(REG_STATUS), agc = readReg(REG_AGC);
  if (st >= 0) {
    bool MD = st & 0x20, ML = st & 0x10, MH = st & 0x08;
    Serial.print("    magnet: ");
    if (!MD)      Serial.println("NOT DETECTED");
    else if (ML)  Serial.println("too weak");
    else if (MH)  Serial.println("too strong");
    else          Serial.println("strength OK");
    Serial.print("    AGC "); Serial.print(agc);
    Serial.println("   (3.3 V range is 0-128; 128 = no headroom left)");
  }
}

int readReg(uint8_t reg) {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return -1;
  Wire.requestFrom(AS5600_ADDR, 1);
  if (Wire.available()) return Wire.read();
  return -1;
}

void updateEncoder() {
  if (!encoderPresent) return;
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(REG_ANGLE);
  if (Wire.endTransmission(false) != 0) return;
  Wire.requestFrom(AS5600_ADDR, 2);
  if (Wire.available() == 2) {
    int hi = Wire.read(), lo = Wire.read();
    float deg = (((hi << 8) | lo) & 0x0FFF) * 360.0f / 4096.0f;
    pivotDeg = deg;
    if (!pivotStarted) { pivotPrev = deg; pivotCont = deg; pivotStarted = true; }
    else {
      float dd = deg - pivotPrev;
      if (dd >  180.0f) dd -= 360.0f;
      if (dd < -180.0f) dd += 360.0f;
      pivotPrev = deg;
      pivotCont += dd;
    }
  }
}

// ---------------------------------------------------------------- serial

void printHelp() {
  Serial.println();
  Serial.println("  o          zero the pivot here");
  Serial.println("  q <deg>    QUIET capture  (do not touch the arm)");
  Serial.println("  g <deg>    WOBBLE capture (push through the full y play)");
  Serial.println("  <enter>    stop the capture and print the summary");
  Serial.println("  s status   b rescan encoder   ? help");
  Serial.println();
  Serial.println("  Suggested run: at each of 0, 90 and 180 deg ->  q <deg>");
  Serial.println("  then  g <deg>, and compare the two spreads.");
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
      // ANY input stops a running capture
      if (capturing) { endCapture(); idx = 0; continue; }
      if (idx > 0) command(buf);
      idx = 0;
    } else buf[idx++] = ch;
  }
}

void command(char* c) {
  while (*c == ' ') c++;
  if (c[0] == 'o' && c[1] == 0) {
    pivotZero = pivotCont;
    Serial.println("  pivot zeroed here.");
    return;
  }
  if (c[0] == 's' && c[1] == 0) {
    Serial.print("  pivot "); Serial.print(pivotAngle(), 2); Serial.println(" deg");
    scanEncoder();
    return;
  }
  if (c[0] == 'b' && c[1] == 0) { scanEncoder(); return; }
  if (c[0] == '?') { printHelp(); return; }

  float v = atof(c + 1);
  switch (c[0]) {
    case 'q': startCapture(false, v); break;
    case 'g': startCapture(true,  v); break;
    default:
      Serial.println("  ? for help");
      break;
  }
}
