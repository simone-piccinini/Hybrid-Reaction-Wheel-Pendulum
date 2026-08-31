// Y-WOBBLE TEST  -  v1.1   (PuTTY-friendly, machine-readable output)
// Reaction-wheel pendulum, pivot encoder (AS5600 on I2C, pins 18/19)
//
// PURPOSE
//   Decide whether the pivot encoder's error is a function of ONE variable
//   (the swing angle) or TWO (swing angle AND out-of-plane tilt).
//
//   One variable  -> the map is invertible, a calibration table recovers truth.
//   Two variables -> one reading cannot recover two unknowns; no table helps.
//
//   Hold the swing angle fixed, push the arm through its full y play, and see
//   how far the reading moves.
//
// THE MOTOR IS NEVER ENABLED. There is no commutation code in this file; the
// driver pins are driven low once at boot and never touched again.
//
// ---------------------------------------------------------------------------
// PUTTY SETUP  (Windows, COM3)
//   Session
//     Connection type : Serial
//     Serial line     : COM3
//     Speed           : 115200
//   Connection > Serial
//     Data bits 8, Stop bits 1, Parity None, Flow control NONE
//     ^ Flow control MUST be None. XON/XOFF or RTS/CTS can stall the stream.
//   Terminal
//     Local echo          : Force on     <- so you can see what you type
//     Local line editing  : Force off    <- so keys are sent immediately
//   Session > Logging
//     Session logging     : All session output
//     Log file name       : e.g. C:\...\results\hw\wobble.log
//     "Always append to the end of it"  <- keeps every capture in one file
//
//   NOTE ON SPEED: this is a Teensy USB CDC port. The baud rate is ignored by
//   the hardware - any value works, both ends just have to agree that a number
//   exists. 115200 is fine. If you want a different LOG RATE, that is the `f`
//   command below, which is a real setting.
// ---------------------------------------------------------------------------
//
// METHOD - read before running
//   The swing angle MUST NOT MOVE during a capture. If the arm also rotates in
//   the swing plane the reading changes for that reason instead, and the number
//   is worthless. Rest the arm against a fixed stop, then push it only OUT OF
//   PLANE, through the full slop available.
//
//   At each position take TWO captures:
//     q <deg>   QUIET  - do not touch the arm
//     g <deg>   WOBBLE - push through the full y play
//   The difference between the spreads is the y-tilt contribution.
//
//   Do this near hanging (0), horizontal (90) and upright (180): sensitivity
//   varies around the circle, and upright is the only one balancing cares about.
//
// COMMANDS
//   o            zero the pivot angle here
//   q <deg>      QUIET capture at true swing angle <deg>
//   g <deg>      WOBBLE capture at true swing angle <deg>
//   <Enter>      stop the running capture and print the summary
//   f <hz>       set log rate, 10..1000 Hz (default 200)
//   s            status / magnet health
//   b            re-scan the encoder
//   ?            help
//
// OUTPUT FORMAT (designed for scripts/analyze_wobble.py)
//   Every non-data line starts with '#', so a parser can skip them wholesale.
//   Data lines are bare CSV. Each capture is bracketed:
//
//     #BEGIN
//     # test,ywobble
//     # nominal_deg,180.0
//     # sample_hz,200
//     t_s,pivot_deg,wheel_deg
//     0.0050,0.000,34.830
//     ...
//     # summary,ywobble,nominal_deg,180.0,n,3534,mean,...,spread,9.141,std,2.63
//     #END

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
const float NOMINAL_SPAN_COUNTS = 4062.0f;   // uncalibrated, reference only

// ---- driver pins: forced safe, never driven ----
const int PIN_IN1 = 2, PIN_IN2 = 3, PIN_IN3 = 4, PIN_EN = 5;

const unsigned long MAX_CAPTURE_MS = 60000;
const long  LOG_HZ_MIN = 10, LOG_HZ_MAX = 1000, LOG_HZ_DEFAULT = 200;

long  logHz = LOG_HZ_DEFAULT;
unsigned long sampleUs = 1000000UL / LOG_HZ_DEFAULT;

bool  encoderPresent = false;
float pivotPrev = 0, pivotCont = 0, pivotZero = 0;
bool  pivotStarted = false;

bool capturing = false, captureIsWobble = false;
float nominalDeg = 0;
unsigned long capStart = 0, lastSample = 0;
long  nSamples = 0;
float vMin = 0, vMax = 0, vSum = 0, vSumSq = 0;

// ---------------------------------------------------------------- setup

void setup() {
  Serial.begin(115200);            // ignored by USB CDC; PuTTY just needs a value
  while (!Serial && millis() < 4000) {}

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
  Serial.println("# =====================================");
  Serial.println("# y_wobble_v1.1   (motor never enabled)");
  Serial.println("# =====================================");
  scanEncoder();
  Serial.print("# log rate,"); Serial.println(logHz);
  Serial.println("# Pin the SWING angle against a stop, then push only");
  Serial.println("# OUT OF PLANE. Enter stops a running capture.");
  printHelp();
}

// ---------------------------------------------------------------- loop

void loop() {
  handleSerial();

  static unsigned long lastRead = 0;
  if (micros() - lastRead >= 1000) { lastRead = micros(); updateEncoder(); }

  if (capturing) {
    if (millis() - capStart > MAX_CAPTURE_MS) {
      Serial.println("# note,auto-stop at 60 s");
      endCapture();
      return;
    }
    unsigned long now = micros();
    if (now - lastSample >= sampleUs) {
      lastSample += sampleUs;                    // no drift accumulation
      if (now - lastSample > 5UL * sampleUs) lastSample = now;   // resync if late
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
      Serial.print("# idle,pivot_deg,");
      Serial.println(pivotAngle(), 2);
    }
  }
}

// ---------------------------------------------------------------- capture

void startCapture(bool wobble, float deg) {
  if (!encoderPresent) {
    Serial.println("# error,encoder not found - fix that first");
    return;
  }
  capturing = true; captureIsWobble = wobble; nominalDeg = deg;
  capStart = millis(); lastSample = micros();
  nSamples = 0; vSum = vSumSq = 0; vMin = vMax = 0;
  Serial.println();
  Serial.println("#BEGIN");
  Serial.print("# test,");        Serial.println(wobble ? "ywobble" : "yquiet");
  Serial.print("# nominal_deg,"); Serial.println(deg, 1);
  Serial.print("# sample_hz,");   Serial.println(logHz);
  Serial.println(wobble ? "# action,push the arm through its full y play"
                        : "# action,do not touch the arm");
  Serial.println("t_s,pivot_deg,wheel_deg");
}

void endCapture() {
  capturing = false;
  if (nSamples < 2) {
    Serial.println("# error,too few samples");
    Serial.println("#END");
    return;
  }
  float mean = vSum / nSamples;
  float var  = vSumSq / nSamples - mean * mean;
  if (var < 0) var = 0;
  float sd = sqrtf(var);
  float spread = vMax - vMin;

  Serial.print("# summary,");
  Serial.print(captureIsWobble ? "ywobble" : "yquiet");
  Serial.print(",nominal_deg,"); Serial.print(nominalDeg, 1);
  Serial.print(",n,");           Serial.print(nSamples);
  Serial.print(",mean,");        Serial.print(mean, 3);
  Serial.print(",min,");         Serial.print(vMin, 3);
  Serial.print(",max,");         Serial.print(vMax, 3);
  Serial.print(",spread,");      Serial.print(spread, 3);
  Serial.print(",std,");         Serial.println(sd, 4);
  Serial.print("# result,spread_deg,"); Serial.print(spread, 2);
  Serial.print(",std_deg,");            Serial.println(sd, 3);

  if (captureIsWobble) {
    if (spread < 3.0f)
      Serial.println("# verdict,ok,y-tilt second order - a calibration map should work");
    else if (spread < 10.0f)
      Serial.println("# verdict,marginal,map may work but the residual is real uncertainty");
    else
      Serial.println("# verdict,too_large,one reading cannot recover the true angle");
    Serial.println("# note,compare against the q capture at the same angle");
  }
  Serial.println("#END");
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
  if (!encoderPresent) {
    Serial.println("# encoder,NOT_FOUND");
    Serial.println("# hint,check SDA 18, SCL 19, 3.3 V, GND, pull-ups");
    return;
  }
  Serial.println("# encoder,found,0x36");
  int st = readReg(REG_STATUS), agc = readReg(REG_AGC);
  if (st >= 0) {
    bool MD = st & 0x20, ML = st & 0x10, MH = st & 0x08;
    Serial.print("# magnet,");
    if (!MD)      Serial.println("NOT_DETECTED");
    else if (ML)  Serial.println("too_weak");
    else if (MH)  Serial.println("too_strong");
    else          Serial.println("ok");
    Serial.print("# agc,"); Serial.print(agc);
    Serial.println(",max_at_3v3,128");
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
  Serial.println("#");
  Serial.println("# o          zero the pivot here");
  Serial.println("# q <deg>    QUIET capture  (do not touch the arm)");
  Serial.println("# g <deg>    WOBBLE capture (push through the full y play)");
  Serial.println("# <Enter>    stop the capture, print the summary");
  Serial.println("# f <hz>     log rate 10..1000 (now: see '# log rate' above)");
  Serial.println("# s status   b rescan   ? help");
  Serial.println("#");
  Serial.println("# suggested: at each of 0, 90, 180 deg ->  q <deg>  then  g <deg>");
  Serial.println("#");
}

// Accepts CR, LF or CRLF. PuTTY sends CR on Enter; the Arduino Serial Monitor
// sends LF. v1.0 only accepted LF, so Enter did nothing under PuTTY.
void handleSerial() {
  static char buf[40];
  static int idx = 0;
  static bool lastWasCR = false;

  while (Serial.available()) {
    char ch = Serial.read();

    if (ch == '\r' || ch == '\n') {
      if (ch == '\n' && lastWasCR) { lastWasCR = false; continue; }  // CRLF pair
      lastWasCR = (ch == '\r');
      buf[idx] = 0;
      if (capturing) {                 // ANY line, including empty, stops it
        endCapture();
      } else if (idx > 0) {
        Serial.print("# > "); Serial.println(buf);   // echo into the log
        command(buf);
      }
      idx = 0;
      continue;
    }

    lastWasCR = false;
    if (ch == 8 || ch == 127) { if (idx > 0) idx--; continue; }      // backspace
    if (idx < (int)sizeof(buf) - 1) buf[idx++] = ch;
  }
}

void command(char* c) {
  while (*c == ' ') c++;
  if (c[0] == 'o' && c[1] == 0) {
    pivotZero = pivotCont;
    Serial.println("# ok,pivot zeroed");
    return;
  }
  if (c[0] == 's' && c[1] == 0) {
    Serial.print("# status,pivot_deg,"); Serial.println(pivotAngle(), 2);
    Serial.print("# status,log_hz,");    Serial.println(logHz);
    scanEncoder();
    return;
  }
  if (c[0] == 'b' && c[1] == 0) { scanEncoder(); return; }
  if (c[0] == '?') { printHelp(); return; }

  float v = atof(c + 1);
  switch (c[0]) {
    case 'q': startCapture(false, v); break;
    case 'g': startCapture(true,  v); break;
    case 'f': {
      long hz = (long)v;
      if (hz < LOG_HZ_MIN || hz > LOG_HZ_MAX) {
        Serial.print("# error,log rate must be ");
        Serial.print(LOG_HZ_MIN); Serial.print("..");
        Serial.println(LOG_HZ_MAX);
        break;
      }
      logHz = hz;
      sampleUs = 1000000UL / (unsigned long)hz;
      Serial.print("# ok,log_hz,"); Serial.println(logHz);
      break;
    }
    default:
      Serial.println("# error,unknown command - send ? for help");
      break;
  }
}
