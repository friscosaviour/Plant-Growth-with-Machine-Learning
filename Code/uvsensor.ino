/*
  Improved Arduino code for reading two SD12D UV intensity sensors.

  Connections:
  - Sensor 1 SIG pin to Arduino Analog Pin A0
  - Sensor 2 SIG pin to Arduino Analog Pin A1
  - Both sensors' GND pins to Arduino GND
  - Both sensors' VCC pins to Arduino 5V

  The code reads the analog voltage from the SIG pin of each sensor,
  converts it to a UV index, and prints the values to the Serial Monitor.
  This version is more modular and easier to read.
*/

// Define the analog pins connected to the SIG pins of the UV sensors
const int UV_SENSOR_PIN_1 = A0;
const int UV_SENSOR_PIN_2 = A1;

// Define constants for the conversion calculations.
// These are calculated once and stored in memory.
const float ADC_VOLTAGE_FACTOR = 5.0 / 1023.0;
const float UV_INDEX_CONVERSION_FACTOR = 0.1; // 1 UV Index unit per 0.1V

void setup() {
  // Initialize serial communication at 9600 bits per second
  Serial.begin(9600);
  Serial.println("Arduino Dual UV Sensor Reader");
  Serial.println("------------------------------");
}

void loop() {
  // Read and print data for the first UV sensor
  readAndPrintSensorData(UV_SENSOR_PIN_1, "Sensor 1");

  // Read and print data for the second UV sensor
  readAndPrintSensorData(UV_SENSOR_PIN_2, "Sensor 2");
 
  Serial.println("------------------------------");

  // Wait for a second before the next reading
  delay(1000);
}

/**
 * @brief Reads the analog value from a specified pin, converts it to voltage and UV index,
 *        and prints the results to the Serial Monitor.
 *
 * @param sensorPin The analog pin connected to the UV sensor's signal pin.
 * @param sensorName A string to identify the sensor in the output (e.g., "Sensor 1").
 */
void readAndPrintSensorData(int sensorPin, const char* sensorName) {
  // Read the raw analog value from the sensor
  int rawValue = analogRead(sensorPin);

  // Convert the analog reading (0-1023) to a voltage (0-5V)
  float voltage = rawValue * ADC_VOLTAGE_FACTOR;

  // Convert the voltage to a UV Index
  // UV Index = Voltage / 0.1 (based on typical sensor datasheet)
  float uvIndex = voltage / UV_INDEX_CONVERSION_FACTOR;

  // Print the results in a single, formatted line
  Serial.print(sensorName);
  Serial.print(": Raw Value = ");
  Serial.print(rawValue);
  Serial.print(", Voltage = ");
  Serial.print(voltage, 2); // Print voltage with 2 decimal places
  Serial.print(" V, UV Index = ");
  Serial.println(uvIndex, 2); // Print UV index with 2 decimal places
}
