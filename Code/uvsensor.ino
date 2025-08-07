/*
  Arduino code for reading two SD12D UV intensity sensors.

  Connections:
  - Sensor 1 SIG pin to Arduino Analog Pin A0
  - Sensor 2 SIG pin to Arduino Analog Pin A1
  - Both sensors' GND pins to Arduino GND
  - Both sensors' VCC pins to Arduino 5V

  The code reads the analog voltage from the SIG pin of each sensor,
  converts it to a UV index, and prints the values to the Serial Monitor.
*/

// Define the analog pins connected to the SIG pins of the UV sensors
const int uvSensorPin1 = A0;
const int uvSensorPin2 = A1;

void setup() {
  // Initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
  Serial.println("UV Sensor Test");
  Serial.println("--------------");
}

void loop() {
  // Read the raw analog value from sensor 1
  int sensorValue1 = analogRead(uvSensorPin1);
  // Read the raw analog value from sensor 2
  int sensorValue2 = analogRead(uvSensorPin2);

  // Convert the analog reading (which goes from 0 - 1023) to a voltage (0 - 5V)
  // The Arduino's ADC has a 10-bit resolution (2^10 = 1024)
  // We divide by 1024.0 to get a floating point result.
  float voltage1 = sensorValue1 * (5.0 / 1023.0);
  float voltage2 = sensorValue2 * (5.0 / 1023.0);

  // The output voltage from the SD12D sensor is typically proportional to the UV index.
  // A common conversion factor is 1 UV Index unit per 0.1V.
  // This may need calibration based on the specific sensor datasheet or experimental data.
  // UV Index = Voltage / 0.1
  float uvIndex1 = voltage1 / 0.1;
  float uvIndex2 = voltage2 / 0.1;

  // Print the results to the Serial Monitor
  Serial.print("Sensor 1: ");
  Serial.print("Raw Value = ");
  Serial.print(sensorValue1);
  Serial.print(", Voltage = ");
  Serial.print(voltage1, 2); // Print voltage with 2 decimal places
  Serial.print(" V, UV Index = ");
  Serial.println(uvIndex1, 2); // Print UV index with 2 decimal places

  Serial.print("Sensor 2: ");
  Serial.print("Raw Value = ");
  Serial.print(sensorValue2);
  Serial.print(", Voltage = ");
  Serial.print(voltage2, 2);
  Serial.print(" V, UV Index = ");
  Serial.println(uvIndex2, 2);

  Serial.println("--------------");

  // Wait for a second before the next reading
  delay(1000);
}
