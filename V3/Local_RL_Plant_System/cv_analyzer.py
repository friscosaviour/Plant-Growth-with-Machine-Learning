import cv2
import numpy as np
import time

class CVAnalyzer:
    """Handles capturing images and analyzing plant health."""

    def __init__(self, config: dict):
        """
        Initializes the CVAnalyzer.

        Args:
            config: A dictionary containing CV analyzer settings.
        """
        self.config = config['cv_analyzer']
        self.capture_width = self.config['image_capture']['width']
        self.capture_height = self.config['image_capture']['height']
        self.lower_green = np.array(self.config['health_metric']['lower_green_hsv'])
        self.upper_green = np.array(self.config['health_metric']['upper_green_hsv'])

    def capture_image(self, file_path: str = "plant_image.jpg") -> np.ndarray:
        """
        Captures an image from the camera.

        Args:
            file_path: The path to save the captured image.

        Returns:
            The captured image as a NumPy array.
        """
        # In a real deployment, you might need to select the correct camera index
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open camera. Returning a mock image.")
            return self._create_mock_image()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

        # Allow camera to warm up
        time.sleep(2)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("ERROR: Failed to capture frame. Returning a mock image.")
            return self._create_mock_image()

        cv2.imwrite(file_path, frame)
        print(f"Image captured and saved to {file_path}")
        return frame

    def calculate_health_metric(self, image: np.ndarray) -> float:
        """
        Calculates a health score based on the greenness of the plant.

        Args:
            image: The input image of the plant.

        Returns:
            A health score, calculated as the percentage of green pixels.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask for green colors
        mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)

        # Calculate the percentage of green pixels
        total_pixels = image.shape[0] * image.shape[1]
        green_pixels = cv2.countNonZero(mask)

        if total_pixels == 0:
            return 0.0

        health_score = (green_pixels / total_pixels) * 100
        print(f"Calculated CV Health Score: {health_score:.2f}%")
        return health_score

    def _create_mock_image(self) -> np.ndarray:
        """Creates a placeholder image for testing without a camera."""
        mock_image = np.zeros((self.capture_height, self.capture_width, 3), dtype=np.uint8)
        # Add some green to the mock image to simulate a plant
        cv2.rectangle(mock_image, (100, 100), (self.capture_width - 100, self.capture_height - 100), (0, 255, 0), -1)
        return mock_image
