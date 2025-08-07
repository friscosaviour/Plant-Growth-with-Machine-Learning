# Plant Growth Optimization with Reinforcement Learning and IoT

## Project Overview

This repository showcases an innovative project focused on optimizing plant growth using a combination of Reinforcement Learning (RL), Computer Vision (CV), and Internet of Things (IoT) hardware. The aim is to create an autonomous system that monitors plant health, makes intelligent decisions regarding environmental parameters, and facilitates optimal growth conditions.

This project demonstrates a strong understanding of full-stack development in the context of intelligent systems, encompassing hardware design, embedded programming, machine learning model development, and data management.

## Features

*   **Reinforcement Learning Agent:** Developed and trained an RL agent to learn optimal watering, lighting, and nutrient delivery strategies based on sensor data and plant health feedback.
*   **Computer Vision Health Analysis:** Implemented a CV module to analyze plant images, detect signs of stress or disease, and provide real-time health assessments.
*   **Raspberry Pi Integration:** Deployed the RL and CV models on a Raspberry Pi for edge computing, enabling autonomous control of plant environment parameters.
*   **Data Logging & Monitoring:** Established a robust data logging system to collect sensor data (moisture, light, temperature, humidity) and plant health metrics over time, crucial for training and evaluating the RL agent.
*   **3D Printable Hardware:** Designed and developed custom 3D models (using SolidWorks) for essential hardware components, such as moisture sensor holders and structural elements, ensuring seamless integration with the system.
*   **Modular Codebase:** Structured the code into distinct modules for easy maintenance, scalability, and future enhancements.

## Technologies Used

*   **Machine Learning:** Python, TensorFlow/Keras (for CV), OpenAI Gym (for RL environment), Custom RL algorithms (Q-Learning).
*   **Embedded Systems:** Raspberry Pi, Python
*   **Computer Vision:** OpenCV
*   **Hardware Design:** SolidWorks, STL, GCODE for 3D printing.
*   **Data Management:** CSV/text files for logging.

## Project Structure

```
Plant-Growth-with-Machine-Learning/
├── 3d Models/              # Contains all 3D design files for physical components
│   ├── GCODE files/        # GCODE files for 3D printing
│   ├── Solidworks Files/   # Original SolidWorks design files
│   └── STL files/          # Stereolithography (STL) files for 3D printing
└── Code/                   # All software source code
    ├── RL.v1/              # Initial Reinforcement Learning prototype
    │   ├── plantenvironment.py
    │   ├── QLearningagent
    │   └── train_plant_agent.py
    └── RLv2_Raspberry_Pi/  # Optimized RL and CV code for Raspberry Pi deployment
        ├── converter_script.py     # Script for model conversion
        ├── cv_health_analyzer.py   # Computer Vision module for plant health
        ├── data_logger.py          # Module for logging sensor data
        ├── main_controller.py      # Main control logic for the Raspberry Pi
        ├── rl_agent.py             # Reinforcement Learning agent implementation
        ├── run_rl_on_pi.py         # Script to run RL system on Raspberry Pi
        └── System_Architecture.txt # Documentation of the system architecture
```

## Setup and Usage

*(Detailed instructions will be added here for setting up the hardware, deploying the software to the Raspberry Pi, and training/running the RL agent. This typically involves dependencies installation, model training, and hardware calibration.)*

## Future Enhancements

*   Integration with cloud platforms for remote monitoring and data visualization. 
*   Implementation of more advanced RL algorithms and neural network architectures.
*   Development of a user-friendly interface for system control and data analysis.
*   Expansion to support multiple plant types and environmental conditions.

## Contact

For any inquiries or collaborations, please feel free to reach out.

System Architecture Overview
 - Sensors: Soil moisture, Light, Temp, Humidity sensors connected to RPi GPIO/I2C/Analog pins. Camera connected via CSI.
 - Data Acquisition: A Python script reads sensor values at regular intervals.
 - Data Logging: Sensor readings (excluding images) are appended to a CSV file.
 - CV Module: Periodically captures images, processes them (potentially using the AI accelerator if a neural network is used) to generate a health score.
 - RL Agent:
   - Reads the latest sensor data from the CSV or directly.
   - Optionally incorporates the CV health score into its state or reward calculation. This measures the amount of vegitaton by calculating the amount of green pixels in the FOV.
   - Uses a trained policy (running inference on the AI accelerator via TFLite) to decide on an action.
   - Actuators: The RL agent's chosen action triggers the corresponding actuator (e.g., turns the water pump on/off via RPi GPIO).
   - Feedback Loop: The environment changes (plant grows, soil dries), sensors detect this, feeding new data back into the system.



## Sensors used
- [ ] Soil moisture sensor
- [ ] Humidity sensor
- [ ] Temperature sensor
- [ ] light intensity sensor
- [ ] UV light strip
- [ ] Oled screen
 
      
## Goals
 - A machine learning model can properly grow a plant by manipulating the soul moisture and light intensity, while it tracks the soil moisture, light intensity, temperature, and humidity.


