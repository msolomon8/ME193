// Step Counter - UNO Q MCU side.
// Samples an MPU-6050 (accelerometer + gyro) at 50 Hz and an HC-SR04 ultrasonic sensor
// (one ping every 60 ms, the last distance is repeated in between) into a ring buffer.
// Python on the Linux side collects new samples with Bridge.call("imu") and shows the
// laptop's step count with Bridge.call("count", steps).
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>

// ---- HC-SR04 ----
// Wired straight in: the UNO Q digital pins (STM32 side) take the 5 V Echo signal.
const int TRIG_PIN = 9;
const int ECHO_PIN = 10;
const uint32_t PING_EVERY = 3;          // samples: one ping per 60 ms (the HC-SR04 needs ~60 ms between pings)
const uint32_t ECHO_START_US = 3000;    // echo never started: sensor missing or busy
const uint32_t ECHO_TIMEOUT_US = 5900;  // nothing within ~1 m (sound goes there and back)
int16_t lastDistMm = 0;                 // 0 = no echo within range

int16_t ping() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  uint32_t t0 = micros();
  while (digitalRead(ECHO_PIN) == LOW)
    if (micros() - t0 > ECHO_START_US) return 0;
  uint32_t t1 = micros();
  while (digitalRead(ECHO_PIN) == HIGH)
    if (micros() - t1 > ECHO_TIMEOUT_US) return 0;
  return (int16_t)((micros() - t1) * 0.1715f);   // mm: 343 m/s, halved for the round trip
}

// ---- IMU ----
const uint32_t SAMPLE_MS = 20;   // 50 Hz
const int RING = 200;            // 4 s of samples, plenty if Python is slow for a moment
const int MAX_PER_CALL = 50;

TwoWire *bus = nullptr;          // whichever bus the MPU-6050 answered on
uint8_t addr = 0;
int16_t ring[RING][7];           // ax ay az gx gy gz (raw counts), distance (mm)
uint32_t head = 0;               // total samples taken (sequence number of the next one)
uint32_t tail = 0;               // sequence number of the next sample to hand to Python
uint32_t nextSample = 0;

void writeReg(uint8_t reg, uint8_t val) {
  bus->beginTransmission(addr);
  bus->write(reg);
  bus->write(val);
  bus->endTransmission();
}

bool findImu() {
  TwoWire *buses[] = {&Wire1, &Wire};   // Qwiic connector first, then the A4/A5 header
  uint8_t addrs[] = {0x68, 0x69};
  for (TwoWire *b : buses) {
    b->begin();
    b->setClock(400000);
    for (uint8_t a : addrs) {
      b->beginTransmission(a);
      if (b->endTransmission() == 0) {
        bus = b;
        addr = a;
        return true;
      }
    }
  }
  return false;
}

void setupImu() {
  writeReg(0x6B, 0x00);   // wake up
  delay(50);
  writeReg(0x1A, 0x03);   // low-pass filter ~44 Hz
  writeReg(0x1B, 0x18);   // gyro +-2000 deg/s (16.4 counts per deg/s)
  writeReg(0x1C, 0x18);   // accel +-16 g (2048 counts per g)
}

bool readImu(int16_t out[7]) {
  bus->beginTransmission(addr);
  bus->write(0x3B);
  if (bus->endTransmission(false) != 0) return false;
  if (bus->requestFrom(addr, (uint8_t)14) != 14) return false;
  uint8_t b[14];
  for (int i = 0; i < 14; i++) b[i] = bus->read();
  out[0] = (b[0] << 8) | b[1];
  out[1] = (b[2] << 8) | b[3];
  out[2] = (b[4] << 8) | b[5];
  // b[6..7] is temperature
  out[3] = (b[8] << 8) | b[9];
  out[4] = (b[10] << 8) | b[11];
  out[5] = (b[12] << 8) | b[13];
  out[6] = lastDistMm;
  return true;
}

// "seq;ax,ay,az,gx,gy,gz,dist;ax,..."  (seq = sequence number of the first sample)
// "NOIMU" if the MPU-6050 was not found.
String imu() {
  if (!bus) return String("NOIMU");
  if (head - tail > RING) tail = head - RING;   // Python fell behind: oldest samples are gone
  String s = String(tail);
  int n = 0;
  while (tail < head && n < MAX_PER_CALL) {
    int16_t *r = ring[tail % RING];
    s += ';';
    for (int i = 0; i < 7; i++) {
      if (i) s += ',';
      s += r[i];
    }
    tail++;
    n++;
  }
  return s;
}

// ---- LED matrix: the step count ----
Arduino_LED_Matrix matrix;
uint8_t frame[104];              // 13 columns x 8 rows

// 3x5 digits, one byte per row, bit 2 = leftmost column
const uint8_t DIGITS[][5] = {
  {7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7}, {5, 5, 7, 1, 1},
  {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 2, 2, 2}, {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7},
  {0, 0, 7, 0, 0},   // 10: -  (waiting for the laptop)
  {7, 4, 7, 4, 7},   // 11: E  (no IMU)
};

void glyph(int g, int col0) {
  for (int r = 0; r < 5; r++)
    for (int c = 0; c < 3; c++)
      if (DIGITS[g][r] & (4 >> c)) frame[(r + 2) * 13 + col0 + c] = 255;
}

// steps >= 0: last three digits, plus one dim dot per thousand along the top row.
// -1: "---" (waiting for the laptop), -2: "E" (no IMU).
void count(int steps) {
  memset(frame, 0, sizeof(frame));
  const int cols[3] = {1, 5, 9};
  if (steps == -2) {
    glyph(11, 5);
  } else if (steps < 0) {
    for (int i = 0; i < 3; i++) glyph(10, cols[i]);
  } else {
    int v = steps % 1000;
    int digits[3] = {v / 100, (v / 10) % 10, v % 10};
    int first = (steps >= 1000 || v >= 100) ? 0 : (v >= 10 ? 1 : 2);   // hide leading zeros below 1000
    for (int i = first; i < 3; i++) glyph(digits[i], cols[i]);
    for (int k = 0; k < steps / 1000 && k < 13; k++) frame[k] = 60;
  }
  matrix.draw(frame);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(8);
  count(-1);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  if (findImu()) setupImu();
  else count(-2);

  Bridge.begin();
  Bridge.provide_safe("imu", imu);   // runs between loop() passes, so the ring buffer is never half-written
  Bridge.provide_safe("count", count);
}

void loop() {
  if (!bus) return;
  uint32_t now = millis();
  if ((int32_t)(now - nextSample) < 0) return;
  nextSample = (now - nextSample > 100) ? now + SAMPLE_MS : nextSample + SAMPLE_MS;   // resync after a long stall
  if (head % PING_EVERY == 0) lastDistMm = ping();   // blocks at most ~9 ms, inside the 20 ms sample period
  if (readImu(ring[head % RING])) head++;
}
