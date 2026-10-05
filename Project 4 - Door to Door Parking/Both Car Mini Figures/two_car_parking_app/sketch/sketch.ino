// Phase 0 - UNO Q MCU side: motors + LED matrix, controlled from Python via Bridge
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

// ---- Wiring (Cytron Maker Drive) ----
const int M1A = 10, M1B = 9;   // Motor 1
const int M2A = 6,  M2B = 5;   // Motor 2

// Flip these after the self-test if a wheel spins the wrong way
const bool INVERT_M1 = true;
const bool INVERT_M2 = true;

// Safety: stop motors if no drive command arrives for this long
const unsigned long WATCHDOG_MS = 1000;

Arduino_LED_Matrix matrix;
uint8_t frame[104];            // 13 columns x 8 rows
volatile unsigned long lastCmd = 0;
volatile bool moving = false;

// speed: -255 (full reverse) .. 0 (stop) .. 255 (full forward)
void setMotor(int pinA, int pinB, int speed, bool invert) {
  if (invert) speed = -speed;
  speed = constrain(speed, -255, 255);
  if (speed > 0)      { analogWrite(pinA, speed); analogWrite(pinB, 0); }
  else if (speed < 0) { analogWrite(pinA, 0);     analogWrite(pinB, -speed); }
  else                { analogWrite(pinA, 0);     analogWrite(pinB, 0); }
}

// Called from Python: Bridge.call("drive", m1, m2)
void drive(int m1, int m2) {
  setMotor(M1A, M1B, m1, INVERT_M1);
  setMotor(M2A, M2B, m2, INVERT_M2);
  moving = (m1 != 0 || m2 != 0);
  lastCmd = millis();
}

// Called from Python: Bridge.call("dot", col, row)  (col 0-12, row 0-7; -1 clears)
void dot(int col, int row) {
  memset(frame, 0, sizeof(frame));
  if (col >= 0 && col < 13 && row >= 0 && row < 8) {
    frame[row * 13 + col] = 255;
  }
  matrix.draw(frame);
}

void setup() {
  pinMode(M1A, OUTPUT); pinMode(M1B, OUTPUT);
  pinMode(M2A, OUTPUT); pinMode(M2B, OUTPUT);
  drive(0, 0);

  matrix.begin();
  matrix.setGrayscaleBits(8);
  dot(-1, -1);

  Bridge.begin();
  Bridge.provide("drive", drive);
  Bridge.provide("dot", dot);
}

void loop() {
  if (moving && millis() - lastCmd > WATCHDOG_MS) {
    drive(0, 0);   // lost contact -> stop
  }
}
