**Role:** You are an expert AI system specializing in embedded ML and robotics. Your task is to generate a complete, production-ready Python codebase for an autonomous plant care system that runs entirely on a local device like a Raspberry Pi.

**Project Goal:**

Rebuild and significantly improve upon a prototype system. The original system (`RLv2_Raspberry_Pi`) used a local Reinforcement Learning (RL) model to control a plant's environment. This new version must be a complete, robust, and user-friendly implementation of that concept.

**Core Functionality:**

The system will operate in a continuous loop:
1.  Read data from sensors (soil moisture, light, temperature, humidity).
2.  Capture an image of the plant and calculate a "health score" using computer vision.
3.  Feed the sensor data and health score into a pre-trained local RL model (`.tflite` format).
4.  The RL model will decide on an action (e.g., do nothing, water, turn on light, water and light).
5.  Execute the chosen action using actuators (water pump, LED light strip).
6.  Log all data and actions to a local database.
7.  Display the system status on a clean terminal-based user interface.

**Key Architectural Requirements and Improvements:**

This is not a prototype. You must generate a complete and polished application, addressing the following specific improvements over the original code:

1.  **No Placeholders or Incomplete Code:** All modules must be fully implemented. Replace all `# TODO`, commented-out code blocks, and placeholder functions with complete, working Python code.

2.  **Separate Training and Inference Scripts:**
    *   **`train.py`:** A script to define the RL environment (using OpenAI `gymnasium`) and train a Deep Q-Network (DQN) or PPO agent using `tensorflow` or `keras`. After training, this script must save the model in both the standard `SavedModel` format and a quantized `.tflite` format ready for inference.
    *   **`run.py`:** The main script for deployment on the Raspberry Pi. It loads the `.tflite` model and executes the main control loop for sensing, inference, and acting.

3.  **Robust Data Logging with a Database:**
    *   Do not use CSV files for logging. Implement a `database_manager.py` module that uses **SQLite**.
    *   Create a simple database schema to log timestamped sensor readings, the CV health score, the action chosen by the agent, and the calculated reward.

4.  **Centralized Configuration:**
    *   Do not hardcode variables. Create a `config.yaml` file to store all settings, including GPIO pin numbers for sensors and actuators, control loop intervals, model file paths, and ideal sensor value ranges.

5.  **Clean, Object-Oriented Structure:**
    *   **`HardwareController` Class:** Create a dedicated class in `hardware.py` to handle all interactions with GPIO devices. It should have clear methods like `read_soil_moisture()`, `get_temperature()`, `pump_on()`, `lights_off()`.
    *   **`CVAnalyzer` Class:** A class in `cv_analyzer.py` with methods like `capture_image()` and `calculate_health_metric()`.
    *   **`RLAgent` Class:** A class in `agent.py` responsible for loading the `.tflite` model and running inference.

6.  **User-Friendly Terminal Interface (TUI):**
    *   In `run.py`, implement a clean TUI using a library like `rich` or `textual`.
    *   The interface should be a dashboard that continuously updates (e.g., every 5 seconds) to show:
        *   Current sensor readings.
        *   The last action taken.
        *   The current CV health score.
        *   A log of the last 5 events.

**Deliverables:**

Please generate the complete directory structure and file contents for this project.

```
/Local_RL_Plant_System/
├── agent.py              # Contains the RLAgent class for loading the model and inference.
├── cv_analyzer.py        # Contains the CVAnalyzer class.
├── database_manager.py   # Manages the SQLite database.
├── environment.py        # The custom Gymnasium environment for training.
├── hardware.py           # Contains the HardwareController class.
├── run.py                # Main entry point for the Pi. Loads model, runs TUI and control loop.
├── train.py              # Script to train the RL model and save the .tflite version.
├── config.yaml           # Central configuration file.
├── requirements.txt      # Lists all necessary Python packages.
└── README.md             # A brief explanation of how to use train.py and run.py.
```
