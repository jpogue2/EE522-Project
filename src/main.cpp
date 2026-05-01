#include <SPI.h>

const int CS_PIN = 10;  // Chip select

// Timing
const unsigned long WINDOW_MS = 3000;
unsigned long windowStart = 0;

// Data accumulation
float sum = 0;
float peak = 0;
int count = 0;

// Baseline tracking
float baseline = 0;
bool baselineSet = false;

float readGSR() {
  int raw = analogRead(A0);
  return (float)raw;
}

void setup() {
  Serial.begin(115200);
  SPI.begin();
  analogReadResolution(12);  // Teensy 4.1 → 0–4095

  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  windowStart = millis();

  Serial.println("Starting GSR capture...");
}

void loop() {
  int gsr = analogRead(A0);
  Serial.println(gsr);
  delay(50); // ~20 Hz
}