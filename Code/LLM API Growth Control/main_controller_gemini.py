import time
import datetime
import os

# Import project modules
import data_logger
import cv_health_analyzer
import gemini_decision_maker # The new Gemini decision module
# import actuator_controller

# --- Configuration ---
CONTROL_INTERVAL_SECONDS = 60 * 10 # Check sensors every 10 minutes
LOG_FILE = 'gemini_plant_log.csv'

# --- Initialization ---
print("Initializing Gemini-based Plant Control System...")
# actuator_controller.setup_gpio() # Placeholder for GPIO setup

# --- Main Loop ---
print("Starting Control Loop...")
while True:
    try:
        start_time = time.time()
        current_timestamp = datetime.datetime.now().isoformat()

        # 1. Read Sensors (using placeholder functions)
        sensor_readings = {
            'soil_moisture': data_logger.read_soil_moisture(),
            'light_intensity': data_logger.read_light_intensity(),
            'temperature': data_logger.read_temperature_humidity()[0],
            'humidity': data_logger.read_temperature_humidity()[1]
        }
        print(f"Sensor Readings: {sensor_readings}")

        # 2. Get Plant Health Metric via CV
        plant_image = cv_health_analyzer.capture_image_opencv()
        health_metric = cv_health_analyzer.calculate_green_health_metric(plant_image)
        print(f"CV Health Metric: {health_metric:.4f}")

        # 3. Get Decision from Gemini API
        decision = gemini_decision_maker.get_gemini_decision(sensor_readings, health_metric)

        # 4. Execute Action
        if decision and 'action' in decision:
            action = decision['action']
            reasoning = decision['reasoning']
            print(f"Gemini chose action: {action} because: {reasoning}")

            # Map action string to actuator control
            # pump_on = action in ["WATER", "WATER_AND_LIGHTS"]
            # lights_on = action in ["LIGHTS_ON", "WATER_AND_LIGHTS"]
            # actuator_controller.control_pump(pump_on)
            # actuator_controller.control_lights(lights_on)
            print(f"Executing: Pump ON, Lights ON") # Placeholder

            # 5. Log the event
            # with open(LOG_FILE, 'a') as f:
            #     f.write(f"{current_timestamp},{sensor_readings['soil_moisture']},{sensor_readings['light_intensity']},{sensor_readings['temperature']},{sensor_readings['humidity']},{health_metric},{action},{reasoning}\n")

        else:
            print("Could not get a valid decision from Gemini. Doing nothing.")

        # 6. Wait for next interval
        elapsed_time = time.time() - start_time
        sleep_time = max(0, CONTROL_INTERVAL_SECONDS - elapsed_time)
        print(f"Loop finished in {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s.")
        time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping control loop.")
        # actuator_controller.cleanup_gpio()
        break
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        # actuator_controller.cleanup_gpio()
        time.sleep(CONTROL_INTERVAL_SECONDS)
