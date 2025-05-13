import tensorflow as tf

# Load your trained Keras model (the main policy network)
# Make sure to build the model first if you are just loading weights
# model = build_dqn_model(STATE_SIZE, ACTION_SIZE, LEARNING_RATE) # Use same parameters
# model.load_weights('dqn_plant_model_final.h5') # Load your best weights

# Placeholder - Assuming 'model' is your trained Keras model object
# model = tf.keras.models.load_model('path/to/your/saved_model_directory') # Or load from .h5

# --- TFLite Conversion ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# --- Optimization (Quantization) - Highly Recommended for Edge TPU ---
# Option 1: Default Optimizations (includes some quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Option 2: Integer Quantization (Requires representative dataset)
# This is usually BEST for Coral TPU performance
def representative_dataset_gen():
    # Provide ~100-500 samples of typical input data (normalized states)
    # Load from your CSV, preprocess/normalize
    # Example: Load N rows from CSV, normalize, yield one by one
    # data = pd.read_csv(CSV_FILE).sample(n=200) # Load sample data
    # for index, row in data.iterrows():
    #     raw_state = [row['soil_moisture'], row['light_intensity'], row['temperature'], row['humidity']]
    #     normalized = normalize_state(raw_state)
    #     yield [np.array(normalized, dtype=np.float32)]
    # Need to implement this based on your actual data loading
    num_calibration_steps = 100
    for _ in range(num_calibration_steps):
         # Generate or load representative input data (needs to match model input shape and type)
         # Example: create random data matching the expected input characteristics
         yield [tf.random.uniform(shape=(1, STATE_SIZE), minval=0.0, maxval=1.0, dtype=tf.float32)]


# Uncomment for Integer Quantization:
# converter.representative_dataset = representative_dataset_gen
# # Ensure integer input/output tensors for full integer quantization (often needed for Edge TPU)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8  # or tf.uint8 depending on model/quantization
# converter.inference_output_type = tf.int8 # or tf.uint8

# --- Convert ---
tflite_model = converter.convert()

# --- Save TFLite Model ---
tflite_model_path = 'dqn_plant_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved to {tflite_model_path}")

# --- Optional: Compile for Edge TPU ---
# If you have the Edge TPU Compiler installed (usually on your development machine):
# run in terminal: edgetpu_compiler dqn_plant_model.tflite
# This will produce 'dqn_plant_model_edgetpu.tflite', optimized for the Coral device.
