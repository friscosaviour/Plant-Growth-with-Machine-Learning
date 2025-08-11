import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tflite

class RLAgent:
    """Handles loading the TFLite model and running inference."""

    def __init__(self, model_path: str):
        """
        Initializes the RLAgent.

        Args:
            model_path: The path to the .tflite model file.
        """
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def choose_action(self, state: np.ndarray) -> int:
        """
        Chooses an action based on the current state.

        Args:
            state: The current state of the environment.

        Returns:
            The action to take.
        """
        # Ensure state is in the correct format (float32 and batched)
        input_data = np.array([state], dtype=np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get the output tensor (Q-values)
        q_values = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Choose the action with the highest Q-value
        action = np.argmax(q_values[0])
        return int(action)
