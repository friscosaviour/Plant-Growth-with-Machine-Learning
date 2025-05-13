# Example: run_rl_on_pi.py (Conceptual)
import tflite_runtime.interpreter as tflite
import numpy as np
import time

# --- Load TFLite Model and Allocate Tensors ---
# Use the Edge TPU delegate if you compiled for it
use_edgetpu = True # Set to False if not using Coral TPU compiled model
model_path = "dqn_plant_model_edgetpu.tflite" if use_edgetpu else "dqn_plant_model.tflite"

try:
    if use_edgetpu:
        interpreter = tflite.Interpreter(model_path=model_path,
                                         experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    else:
        interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print(f"Loaded TFLite model: {model_path}")
except ValueError as e:
     print(f"Error loading model or delegate: {e}")
     print("Ensure the correct model file path and Edge TPU runtime library are available.")
     # Fallback or exit
     use_edgetpu = False
     try:
        model_path = "dqn_plant_model.tflite"
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"Loaded TFLite model (CPU): {model_path}")
     except Exception as e2:
         print(f"Failed to load CPU TFLite model: {e2}")
         exit() # Or handle appropriately


# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check input type (important if using integer quantization)
input_type = input_details[0]['dtype']
print(f"Model expects input type: {input_type}")


def get_action_from_model(state):
    """Gets the best action from the loaded TFLite model."""
    # Prepare input tensor (normalize state and match model's expected type/shape)
    input_state = np.array(state, dtype=np.float32) # Start with float32
    if input_type == np.int8 or input_type == np.uint8:
        # Quantize input if model expects integers
        scale, zero_point = input_details[0]['quantization']
        input_state = (input_state / scale + zero_point).astype(input_type)

    # Add batch dimension
    input_data = np.expand_dims(input_state, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor (Q-values)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output if necessary (if output is int8/uint8)
    if output_details[0]['dtype'] == np.int8 or output_details[0]['dtype'] == np.uint8:
         scale, zero_point = output_details[0]['quantization']
         output_data = (output_data.astype(np.float32) - zero_point) * scale

    # Get the action with the highest Q-value
    action = np.argmax(output_data[0])
    return action

# --- Example Usage (inside your main control loop) ---
# current_raw_state = read_all_sensors() # [soil, light, temp, hum]
# normalized_state = normalize_state(current_raw_state) # Use your normalization function
# chosen_action = get_action_from_model(normalized_state)
# print(f"Raw State: {current_raw_state}")
# print(f"Normalized State: {normalized_state}")
# print(f"Chosen Action: {chosen_action}")
# # execute_action(chosen_action) # Control pump/lights based on action index
