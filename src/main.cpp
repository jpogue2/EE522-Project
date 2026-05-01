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

// Replace with your sensor's SPI read function
float readGSR() {
  digitalWrite(CS_PIN, LOW);

  // Example: read 2 bytes (adjust for your sensor)
  byte highByte = SPI.transfer(0x00);
  byte lowByte  = SPI.transfer(0x00);

  digitalWrite(CS_PIN, HIGH);

  int raw = (highByte << 8) | lowByte;

  // Convert to conductance (depends on your module!)
  return (float)raw;
}

void setup() {
  Serial.begin(115200);
  SPI.begin();

  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  windowStart = millis();

  Serial.println("Starting GSR capture...");
}

void loop() {
  float gsr = readGSR();

  // Accumulate stats
  sum += gsr;
  if (gsr > peak) peak = gsr;
  count++;

  unsigned long now = millis();

  // Every 3 seconds
  if (now - windowStart >= WINDOW_MS) {
    float avg = sum / count;

    // Establish baseline (first few windows)
    if (!baselineSet) {
      baseline = avg;
      baselineSet = true;
    }

    float delta = avg - baseline;

    // Output in CSV format (easy to log/analyze)
    Serial.print("AVG:");
    Serial.print(avg);
    Serial.print(",PEAK:");
    Serial.print(peak);
    Serial.print(",DELTA:");
    Serial.println(delta);

    // Reset window
    sum = 0;
    peak = 0;
    count = 0;
    windowStart = now;
  }

  delay(50); // ~20 Hz sampling
}