// The Impenetrable Safe - MCU side (STM32 on the UNO Q)
//
// Button D7 (to GND, uses the internal pull-up)  -> tells Python "button_pressed" (short press)
//                                                  or "button_held" (held down 3 s)
// LEDs  D2 D3 D4 (each through its 120 ohm resistor to GND)
// Servo signal D9 (brown -> GND, red -> 5V, orange -> D9)
//
// Python calls set_state(mode, digits, servoOpen) whenever something changes:
//   mode 0 IDLE      all LEDs off, servo at LOCKED_ANGLE
//   mode 1 READING   flash all 3 LEDs at the start, then D2, D3, D4 light up as digits are accepted
//   mode 2 UNLOCKED  all 3 LEDs flash, servo turns to OPEN_ANGLE
//   mode 3 DENIED    all 3 LEDs blink fast
//   mode 4 LOCKOUT   LEDs chase back and forth
//   mode 5 NEW CODE  choosing a new code: entered digits solid, the next LED blinks slowly, servo open
//   mode 6 SAVED     new code saved: LEDs sweep D2 -> D3 -> D4, servo open

#include "Arduino_RouterBridge.h"

const int LEDS[3] = {2, 3, 4};
const int BUTTON = 7;
const int SERVO_PIN = 9;
const int LOCKED_ANGLE = 180;  // locked = all the way to the other side (the opposite of 0)
const int OPEN_ANGLE = 90;     // open = pointing UP. It only ever moves between 90 and 180.

// volatile: set_state runs on the Bridge's thread, loop() reads them
volatile int mode = 0, digits = 0;
volatile bool servoOpen = false;

int shownMode = -1;
unsigned long modeStart = 0;
int lastButton = HIGH;              // debounced button state
int lastRaw = HIGH;
unsigned long rawChangedMs = 0;
unsigned long pressStartMs = 0;
bool holdSent = false;
const unsigned long HOLD_MS = 3000;   // hold this long = "button_held"

// servo: we make the 50 Hz pulses ourselves (works on any core), for 1.5 s after each move
volatile int servoAngle = LOCKED_ANGLE;
volatile unsigned long servoMoveMs = 0;
unsigned long lastPulseUs = 0;

void set_state(int newMode, int newDigits, bool newServoOpen) {
  mode = newMode;
  digits = newDigits;
  int angle = newServoOpen ? OPEN_ANGLE : LOCKED_ANGLE;
  if (angle != servoAngle) {
    servoAngle = angle;
    servoMoveMs = millis();
  }
  servoOpen = newServoOpen;
}

void setup() {
  for (int i = 0; i < 3; i++) pinMode(LEDS[i], OUTPUT);
  pinMode(BUTTON, INPUT_PULLUP);
  pinMode(SERVO_PIN, OUTPUT);
  servoMoveMs = millis();          // move to the locked position at startup
  Bridge.begin();
  Bridge.provide("set_state", set_state);   // set_state only stores numbers; loop() does the pins
}

void showLeds(bool a, bool b, bool c) {
  digitalWrite(LEDS[0], a ? HIGH : LOW);
  digitalWrite(LEDS[1], b ? HIGH : LOW);
  digitalWrite(LEDS[2], c ? HIGH : LOW);
}

void updateLeds() {
  if (mode != shownMode) {
    shownMode = mode;
    modeStart = millis();
  }
  unsigned long t = millis() - modeStart;
  if (mode == 1) {                       // READING
    if (t < 900) {                       // start: flash all three, 3 times
      bool on = (t / 150) % 2 == 0;
      showLeds(on, on, on);
    } else {
      showLeds(digits >= 1, digits >= 2, digits >= 3);
    }
  } else if (mode == 2) {                // UNLOCKED: all three flash
    bool on = (t / 250) % 2 == 0;
    showLeds(on, on, on);
  } else if (mode == 3) {                // DENIED
    bool on = (t / 100) % 2 == 0;
    showLeds(on, on, on);
  } else if (mode == 5) {                // NEW CODE: next LED blinks slowly
    bool blink = (t / 400) % 2 == 0;
    showLeds(digits >= 1 || (digits == 0 && blink),
             digits >= 2 || (digits == 1 && blink),
             digits >= 3 || (digits == 2 && blink));
  } else if (mode == 6) {                // SAVED: sweep D2 -> D3 -> D4 -> all on
    int k = (t / 120) % 4;
    showLeds(k == 0 || k == 3, k == 1 || k == 3, k == 2 || k == 3);
  } else if (mode == 4) {                // LOCKOUT
    int k = (t / 150) % 4;
    int led = (k == 3) ? 1 : k;          // 0,1,2,1,0,1,2...
    showLeds(led == 0, led == 1, led == 2);
  } else {                               // IDLE
    showLeds(false, false, false);
  }
}

void updateServo() {
  if (millis() - servoMoveMs > 1500) return;      // stop pulsing once it has arrived (no buzzing)
  if (micros() - lastPulseUs >= 20000) {          // one pulse every 20 ms
    lastPulseUs = micros();
    int pulse = map(servoAngle, 0, 180, 500, 2500);
    digitalWrite(SERVO_PIN, HIGH);
    delayMicroseconds(pulse);
    digitalWrite(SERVO_PIN, LOW);
  }
}

void readButton() {
  // Debounce: only believe the pin once it has stayed the same for 30 ms.
  // A short press is sent when you LET GO; holding 3 s sends "button_held" instead.
  int raw = digitalRead(BUTTON);
  unsigned long now = millis();
  if (raw != lastRaw) {
    lastRaw = raw;
    rawChangedMs = now;
  }
  if (now - rawChangedMs < 30) return;            // still bouncing
  if (raw == LOW && lastButton == HIGH) {         // pressed down
    pressStartMs = now;
    holdSent = false;
  }
  if (raw == LOW && !holdSent && now - pressStartMs >= HOLD_MS) {   // still held after 3 s
    holdSent = true;
    Bridge.notify("button_held");
  }
  if (raw == HIGH && lastButton == LOW && !holdSent) {   // let go before 3 s
    Bridge.notify("button_pressed");
  }
  lastButton = raw;
}

void loop() {
  readButton();
  updateLeds();
  updateServo();
}
