import csv
import time
import random # Replace with actual sensor reading libraries (e.g., Adafruit_DHT, specific ADC libraries)
import datetime
import os
import numpy as np

# --- Configuration ---
CSV_FILE = 'plant_data.csv'
LOG_INTERVAL_SECONDS = 60 * 5 # Log every 5 minutes
SENSOR_RANGES = {
    'soil_moisture': {'min': 0, 'max': 1023, 'ideal_min': 300, 'ideal_max': 700}, # Example ADC range
    'light_intensity': {'min': 0, 'max': 4095, 'ideal_min': 1000, 'ideal_max': 3000}, # Example ADC/Sensor range
    'temperature': {'min': 0, 'max': 50, 'ideal_min': 18, 'ideal_max': 26}, # Celsius
    'humidity': {'min': 0, 'max': 100, 'ideal_min': 40, 'ideal_max': 60} # Percentage
}
CSV_HEADER = ['timestamp', 'soil_moisture', 'light_intensity', 'temperature', 'humidity']

# --- Sensor Reading Functions (Replace with actual hardware interaction) ---
def read_soil_moisture():
    # TODO: Implement actual sensor reading
    return random.uniform(SENSOR_RANGES['soil_moisture']['min'], SENSOR_RANGES['soil_moisture']['max'])

def read_light_intensity():
    # TODO: Implement actual sensor reading
    return random.uniform(SENSOR_RANGES['light_intensity']['min'], SENSOR_RANGES['light_intensity']['max'])

def read_temperature_humidity():
    # TODO: Implement actual sensor reading (e.g., Adafruit_DHT)
    temp = random.uniform(SENSOR_RANGES['temperature']['min'] - 5, SENSOR_RANGES['temperature']['max'] + 5)
    humidity = random.uniform(SENSOR_RANGES['humidity']['min'] - 10, SENSOR_RANGES['humidity']['max'] + 10)
    return max(SENSOR_RANGES['temperature']['min'], min(SENSOR_RANGES['temperature']['max'], temp)), \
           max(SENSOR_RANGES['humidity']['min'], min(SENSOR_RANGES['humidity']['max'], humidity))

# --- Normalization ---
def normalize(value, sensor_name):
    s_min = SENSOR_RANGES[sensor_name]['min']
    s_max = SENSOR_RANGES[sensor_name]['max']
    return (value - s_min) / (s_max - s_min) if (s_max - s_min) != 0 else 0

# --- Main Logging Loop ---
def log_data():
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or os.path.getsize(CSV_FILE) == 0:
            writer.writerow(CSV_HEADER) # Write header only if file is new/empty

        timestamp = datetime.datetime.now().isoformat()
        soil = read_soil_moisture()
        light = read_light_intensity()
        temp, hum = read_temperature_humidity()

        writer.writerow([timestamp, soil, light, temp, hum])
        print(f"Logged: {timestamp}, Soil={soil:.2f}, Light={light:.2f}, Temp={temp:.1f}C, Hum={hum:.1f}%")

if __name__ == "__main__":
    print(f"Starting data logging to {CSV_FILE} every {LOG_INTERVAL_SECONDS} seconds.")
    while True:
        try:
            log_data()
            time.sleep(LOG_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("Stopping data logging.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(LOG_INTERVAL_SECONDS) # Wait before retrying
