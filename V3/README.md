# Technical Documentation for the Local RL Plant System

This document provides a technical overview of the autonomous plant care system located in the `Local_RL_Plant_System` directory.

## Project Architecture

The system is designed with a modular, object-oriented architecture to separate concerns and facilitate maintainability. It revolves around a central control loop in `run.py` that integrates various components for sensing, analysis, decision-making, and acting.

### Core Components

1.  **`config.yaml`**:
    *   **Purpose**: A centralized configuration file to manage all system parameters without hardcoding values.
    *   **Technical Details**: Uses YAML syntax for human-readable key-value pairs. It is parsed at runtime by `run.py` and the configuration dictionary is passed to the constructors of other classes. This allows for easy modification of GPIO pins, sensor thresholds, and file paths.

2.  **`database_manager.py`**:
    *   **Purpose**: Handles all database operations.
    *   **Technical Details**: Implements a `DatabaseManager` class that uses Python's built-in `sqlite3` library.
    *   **Schema**: It creates a single table, `plant_log`, with the following columns:
        *   `id`: Primary Key (INTEGER)
        *   `timestamp`: Timestamp of the event (DATETIME)
        *   `soil_moisture`, `temperature`, `humidity`, `light_level`: Sensor readings (REAL)
        *   `cv_health_score`: Health score from `CVAnalyzer` (REAL)
        *   `action_taken`: The integer action decided by the `RLAgent` (INTEGER)
        *   `reward`: The calculated reward for the state-action pair (REAL)

3.  **`hardware.py`**:
    *   **Purpose**: Abstracts all low-level hardware interactions.
    *   **Technical Details**: The `HardwareController` class provides high-level methods (e.g., `read_soil_moisture()`, `pump_on()`). It dynamically checks for the availability of Raspberry Pi-specific libraries (`RPi.GPIO`, `adafruit_dht`, etc.). If the libraries are not found, it transparently switches to a **mock mode**, generating random sensor data and simulating actuator behavior. This allows for development and testing on non-Pi systems.
    *   **Dependencies**: `RPi.GPIO` for general pin control, `adafruit-circuitpython-dht` for the temperature/humidity sensor, and `spidev` for communicating with an MCP3008 ADC to read the analog signal from the soil moisture sensor.

4.  **`cv_analyzer.py`**:
    *   **Purpose**: Manages image capture and computer vision analysis.
    *   **Technical Details**: The `CVAnalyzer` class uses `opencv-python`.
    *   `capture_image()`: Accesses a USB camera (assumed at index 0) to capture a high-resolution image. It includes error handling for cases where the camera is not available, returning a mock green-square image instead.
    *   `calculate_health_metric()`: A simple but effective algorithm that converts the BGR image to the HSV color space and then calculates the percentage of pixels that fall within a predefined green color range. This percentage is used as the plant's "health score".

5.  **`environment.py`**:
    *   **Purpose**: Defines the training environment for the Reinforcement Learning agent.
    *   **Technical Details**: The `PlantEnv` class inherits from `gymnasium.Env`.
    *   **State Space**: A `Box` space representing `[soil_moisture, temperature, humidity, light_on, health_score]`.
    *   **Action Space**: A `Discrete` space with 4 actions: do nothing, water, lights on, water and lights on.
    *   **Simulation**: The `step()` method contains a simplified physics model of the plant's environment. For example, watering increases moisture, which then gradually decreases over time. Plant health is simulated to improve when environmental conditions are close to the ideal values defined in `config.yaml`.
    *   **Reward Function**: The reward is calculated based on the negative distance from the ideal state for each sensor reading, plus a positive reward proportional to the plant's health. This incentivizes the agent to maintain optimal conditions.

6.  **`agent.py`**:
    *   **Purpose**: Loads the trained model and performs inference.
    *   **Technical Details**: The `RLAgent` class uses the `tflite_runtime` library, which is a lightweight TensorFlow Lite interpreter suitable for edge devices. It loads the `.tflite` model, allocates tensors, and provides a `choose_action()` method that takes the current state, feeds it to the model, and returns the action with the highest predicted Q-value.

7.  **`train.py`**:
    *   **Purpose**: The script for training the RL agent.
    *   **Technical Details**:
        *   It instantiates the `PlantEnv` from `environment.py`.
        *   It defines a Deep Q-Network (DQN) model using `tensorflow.keras`. The model is a simple multi-layer perceptron (MLP) that takes the state as input and outputs Q-values for each possible action.
        *   It uses a basic DQN training loop: collect experience from the environment, store it in a replay buffer, and periodically sample from the buffer to train the network.
        *   **Model Export**: After training, it saves the model in the standard TensorFlow `SavedModel` format. It then uses the `TFLiteConverter` to convert the `SavedModel` into a quantized `.tflite` file, which is optimized for inference on the Raspberry Pi.

8.  **`run.py`**:
    *   **Purpose**: The main entry point for the application on the Raspberry Pi.
    *   **Technical Details**:
        *   **Initialization**: It starts by loading the `config.yaml` file and initializing all the manager classes (`HardwareController`, `CVAnalyzer`, `DatabaseManager`, `RLAgent`).
        *   **TUI**: It uses the `rich` library to create a terminal-based dashboard. A `Live` display is used to continuously refresh the data without clearing the screen. The layout is organized using `Table` and `Panel` objects.
        *   **Main Loop**: The core of the application is a `while True` loop that:
            1.  Reads sensor data from the `HardwareController`.
            2.  Captures an image and calculates the health score using the `CVAnalyzer`.
            3.  Constructs the current state array.
            4.  Passes the state to the `RLAgent` to get the next action.
            5.  Executes the action using the `HardwareController`.
            6.  Calculates the reward (this is a simple implementation for logging purposes).
            7.  Logs all data to the SQLite database via the `DatabaseManager`.
            8.  Updates the `rich` TUI with the latest data.
            9.  Sleeps for the configured interval.

## Usage

1.  **Installation**:
    ```bash
    pip install -r Local_RL_Plant_System/requirements.txt
    ```
2.  **Training**:
    ```bash
    python Local_RL_Plant_System/train.py
    ```
    This will generate `model/plant_model.h5` and `model/plant_model.tflite`.

3.  **Running**:
    ```bash
    python Local_RL_Plant_System/run.py
    ```
    This will start the main control loop and display the TUI.
