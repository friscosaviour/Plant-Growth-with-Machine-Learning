/*
 * Arduino sketch to test a soil moisture sensor.
 *
 * This program reads the analog value from the sensor and prints it to the
 * Serial Monitor. The raw value gives an indication of moisture level.
 *
 * Connections:
 * - Soil Moisture Sensor VCC pin to Arduino 5V
 * - Soil Moisture Sensor GND pin to Arduino GND
 * - Soil Moisture Sensor A0 pin to Arduino Analog Pin A0
 *
 * NOTE: The raw values will vary based on the specific sensor and calibration.
 * A lower value typically means the soil is wetter, and a higher value means it's drier.
 */

// Define the analog pin connected to the soil moisture sensor
const int soilMoisturePin = A0;

void setup() {
  // Start serial communication
  Serial.begin(9600);
  Serial.println("Soil Moisture Sensor Test");
  Serial.println("-------------------------");
}

void loop() {
  // Read the analog value from the sensor.
  // The value will be between 0 (wettest) and 1023 (driest).
  int sensorValue = analogRead(soilMoisturePin);

  // Convert the raw value to a moisture percentage for easier understanding.
  // The map() function is used here.
  // The values (0, 1023) are typical for most sensors, but may need
  // calibration for your specific sensor.
  int moisturePercentage = map(sensorValue, 1023, 0, 0, 100);

  // Print the raw value and the calculated percentage
  Serial.print("Raw Value: ");
  Serial.print(sensorValue);
  Serial.print(" | ");
  Serial.print("Moisture: ");
  Serial.print(moisturePercentage);
  Serial.println(" %");
  Serial.println("-------------------------");

  // Wait for a second before the next reading
  delay(1000);
}
