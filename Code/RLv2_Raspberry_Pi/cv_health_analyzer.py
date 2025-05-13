
import cv2
import numpy as np
import time
# Optional: Use picamera library if using the official Pi Camera module
# from picamera import PiCamera
# from picamera.array import PiRGBArray

# --- Configuration ---
# Define the lower and upper bounds for GREEN in HSV color space
# These values might need significant tuning based on your camera and lighting!
GREEN_LOWER = np.array([35, 50, 50]) # Lower Hue, Saturation, Value
GREEN_UPPER = np.array([85, 255, 255]) # Upper Hue, Saturation, Value

# Image capture resolution (keep it low for speed)
CAPTURE_WIDTH = 320
CAPTURE_HEIGHT = 240

# --- Image Capture Function ---
def capture_image_opencv(device_index=0):
    """Captures an image using OpenCV from a USB webcam."""
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Error: Could not open video device {device_index}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    time.sleep(0.5) # Allow camera to warm up and stabilize
    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    else:
        print("Error: Failed to capture frame.")
        return None

# def capture_image_picamera():
#     """Captures an image using the PiCamera module."""
#     with PiCamera() as camera:
#         camera.resolution = (CAPTURE_WIDTH, CAPTURE_HEIGHT)
#         rawCapture = PiRGBArray(camera, size=(CAPTURE_WIDTH, CAPTURE_HEIGHT))
#         time.sleep(0.5) # Camera warmup
#         camera.capture(rawCapture, format="bgr") # BGR format for OpenCV compatibility
#         image = rawCapture.array
#         return image

# --- Health Metric Calculation ---
def calculate_green_health_metric(image):
    """
    Calculates a health metric based on the percentage of green pixels.

    Args:
        image: A BGR image (numpy array) from OpenCV or PiCamera.

    Returns:
        A float between 0.0 and 1.0 representing the green percentage,
        or 0.0 if the image is invalid.
    """
    if image is None:
        return 0.0

    # 1. Optional: Simple Background Subtraction (if background is static/simple)
    #    Could improve accuracy by only considering plant pixels.
    #    Requires a more complex setup (e.g., taking a picture of the background).
    #    Or, attempt segmentation based on color/contours (more advanced).

    # 2. Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 3. Create a mask for green pixels
    green_mask = cv2.inRange(hsv_image, GREEN_LOWER, GREEN_UPPER)

    # 4. Calculate the percentage
    # Option A: Percentage of *total* image pixels that are green
    total_pixels = image.shape[0] * image.shape[1]
    green_pixels = cv2.countNonZero(green_mask)

    if total_pixels == 0:
        return 0.0

    green_percentage = green_pixels / total_pixels

    # Option B (Slightly better): Percentage of *non-black* pixels that are green
    # This helps if the plant doesn't fill the frame and the background is dark.
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, non_black_mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    # total_non_black_pixels = cv2.countNonZero(non_black_mask)
    # if total_non_black_pixels == 0: return 0.0
    # green_pixels_in_non_black = cv2.countNonZero(cv2.bitwise_and(green_mask, non_black_mask))
    # green_percentage = green_pixels_in_non_black / total_non_black_pixels

    # For debugging: Show the mask
    # cv2.imshow("Original", image)
    # cv2.imshow("Green Mask", green_mask)
    # cv2.waitKey(1) # Display for a short time

    return green_percentage

# --- Example Usage ---
if __name__ == "__main__":
    print("Attempting to capture image...")
    # Use capture_image_picamera() if using the Pi Camera module
    img = capture_image_opencv(0) # Use device index 0

    if img is not None:
        print("Image captured successfully.")
        # cv2.imwrite("captured_plant.jpg", img) # Save for inspection
        health_metric = calculate_green_health_metric(img)
        print(f"Calculated Green Health Metric: {health_metric:.4f}")

        # --- How to integrate with RL ---
        # This health_metric would be used in the RL reward function:
        # reward = calculate_reward(sensor_readings, action, health_metric, previous_health_metric)
        # It could potentially also be added as an auxiliary input to the RL state:
        # state = [norm_moist, norm_light, norm_temp, norm_hum, health_metric]
        # (Requires retraining the RL model with the new state size)
    else:
        print("Failed to capture image.")
