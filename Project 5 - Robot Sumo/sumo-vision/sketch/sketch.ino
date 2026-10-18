// Robot Sumo - UNO Q MCU side: only the LED matrix, used as a debug display.
// Python calls Bridge.call("dot", col, row) to show where the opponent is.
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

Arduino_LED_Matrix matrix;
uint8_t frame[104];            // 13 columns x 8 rows

// col 0-12, row 0-7; -1 clears the matrix
void dot(int col, int row) {
  memset(frame, 0, sizeof(frame));
  if (col >= 0 && col < 13 && row >= 0 && row < 8) {
    frame[row * 13 + col] = 255;
  }
  matrix.draw(frame);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(8);
  dot(-1, -1);

  Bridge.begin();
  Bridge.provide("dot", dot);
}

void loop() {
}
