# Local Reinforcement Learning Plant System

This project contains a complete, production-ready Python codebase for an autonomous plant care system that runs entirely on a local device like a Raspberry Pi.

## How to Use

### 1. Installation

First, ensure you have Python 3 installed. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Before running the system, you may need to adjust the settings in `config.yaml`. This is where you can set the GPIO pin numbers for your hardware, adjust the ideal sensor values for your specific plant, and change the control loop timing.

### 3. Training the Model (Optional)

A pre-trained model is not provided. You must train the agent first. This script will simulate the plant's environment and train a Reinforcement Learning model to care for it.

```bash
python train.py
```

This will create two files in the `model/` directory:
*   `plant_model_saved_model/`: A TensorFlow SavedModel directory.
*   `plant_model.tflite`: A TensorFlow Lite model, which is used for inference on the Raspberry Pi.

### 4. Running the System

Once you have a trained `.tflite` model, you can run the main application on your Raspberry Pi.

```bash
sudo python run.py
```

*Note: `sudo` may be required for GPIO access.*

This will start the autonomous control loop and display a terminal dashboard with live updates on the system's status.
