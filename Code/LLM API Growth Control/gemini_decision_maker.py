import os
import json
import google.generativeai as genai

# --- Configuration ---
# IMPORTANT: Set your Google API Key as an environment variable named 'GOOGLE_API_KEY'
# You can get a key from https://aistudio.google.com/app/apikey
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=API_KEY)

# Configure the model
GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "application/json",
}

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

MODEL = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=GENERATION_CONFIG,
                              safety_settings=SAFETY_SETTINGS)

def get_gemini_decision(sensor_data, health_metric):
    """
    Queries the Gemini API to get a plant care decision.

    Args:
        sensor_data (dict): A dictionary of current sensor readings.
        health_metric (float): The calculated CV health metric.

    Returns:
        dict: A dictionary containing the 'action' and 'reasoning', or None on failure.
    """
    prompt = f"""
You are an expert botanist and agricultural AI. Your goal is to optimize the growth and health of a plant based on sensor data.

**Current Plant Status:**
- Soil Moisture: {sensor_data['soil_moisture']:.2f} (Ideal range is 300-700)
- Light Intensity: {sensor_data['light_intensity']:.2f} (Ideal range is 1000-3000)
- Temperature: {sensor_data['temperature']:.1f}°C (Ideal range is 18-26°C)
- Humidity: {sensor_data['humidity']:.1f}% (Ideal range is 40-60%)
- Visual Health Metric: {health_metric:.4f} (A score from 0.0 to 1.0, higher is better, representing the plant's greenness)

**Your Task:**
Based on the data above, decide which single action to take *right now*. The available actions are:
- "DO_NOTHING": If all conditions are optimal or no action is needed.
- "WATER": If the soil moisture is too low.
- "LIGHTS_ON": If the light intensity is too low.
- "WATER_AND_LIGHTS": If both soil is dry and light is low.

**Response Format:**
Respond ONLY with a JSON object containing your chosen action and a brief, one-sentence reasoning.

Example:
{{
  "action": "WATER",
  "reasoning": "The soil moisture is below the ideal minimum threshold, indicating the need for watering."
}}
"""

    print("Querying Gemini API for decision...")
    try:
        response = MODEL.generate_content(prompt)
        # The response text should be a clean JSON string thanks to response_mime_type
        decision = json.loads(response.text)

        if "action" in decision and "reasoning" in decision:
            return decision
        else:
            print(f"Error: Gemini response is missing 'action' or 'reasoning'. Response: {decision}")
            return None

    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from Gemini response. Raw response: '{response.text}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while querying Gemini: {e}")
        return None

if __name__ == '__main__':
    # Example usage for testing this module directly
    mock_sensor_data = {
        'soil_moisture': 750, # Too high
        'light_intensity': 800, # Too low
        'temperature': 23.1,
        'humidity': 51.8
    }
    mock_health = 0.91
    decision = get_gemini_decision(mock_sensor_data, mock_health)

    if decision:
        print(f"\nGemini Decision: {decision['action']}")
        print(f"Reasoning: {decision['reasoning']}")
    else:
        print("\nFailed to get a decision from Gemini.")
