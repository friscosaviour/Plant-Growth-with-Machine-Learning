

import requests
import json

# Configuration for the Ollama server
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# IMPORTANT: Make sure you have a model downloaded, e.g., `ollama pull phi3` or `ollama pull llama3`
OLLAMA_MODEL = "phi3" # Change this to your preferred model

def get_llm_decision(sensor_data, health_metric):
    """
    Queries the local LLM to get a plant care decision.

    Args:
        sensor_data (dict): A dictionary of current sensor readings.
        health_metric (float): The calculated CV health metric.

    Returns:
        dict: A dictionary containing the 'action' and 'reasoning', or None on failure.
    """
    prompt = f"""
You are an expert botanist and agricultural AI. Your goal is to optimize the growth and health of a plant based on sensor data.

**Current Plant Status:**
- Soil Moisture: {sensor_data['soil_moisture']:.2f} (Ideal is 300-700)
- Light Intensity: {sensor_data['light_intensity']:.2f} (Ideal is 1000-3000)
- Temperature: {sensor_data['temperature']:.1f}°C (Ideal is 18-26°C)
- Humidity: {sensor_data['humidity']:.1f}% (Ideal is 40-60%)
- Visual Health Metric: {health_metric:.4f} (A score from 0.0 to 1.0, higher is better, representing greenness)

**Your Task:**
Based on the data above, decide which single action to take *right now*. The available actions are:
- "DO_NOTHING": If all conditions are optimal or no action is needed.
- "WATER": If the soil moisture is too low.
- "LIGHTS_ON": If the light intensity is too low.
- "WATER_AND_LIGHTS": If both soil is dry and light is low.

**Response Format:**
Respond ONLY with a JSON object containing your chosen action and a brief reasoning. Do not add any other text or explanations.

Example:
{{
  "action": "WATER",
  "reasoning": "The soil moisture is below the ideal minimum threshold."
}}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False, # We want the full response at once
        "format": "json" # Request JSON output
    }

    print("Querying LLM for decision...")
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes

        # The actual JSON response from the LLM is in the 'response' key
        response_data = response.json()
        llm_output_str = response_data.get("response", "{}")

        # Parse the JSON string provided by the LLM
        decision = json.loads(llm_output_str)

        if "action" in decision and "reasoning" in decision:
            return decision
        else:
            print(f"Error: LLM response is missing 'action' or 'reasoning'. Response: {decision}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama API: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from LLM response. Raw response: '{llm_output_str}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == '__main__':
    # Example usage for testing this module directly
    mock_sensor_data = {
        'soil_moisture': 250,
        'light_intensity': 800,
        'temperature': 22.5,
        'humidity': 55.1
    }
    mock_health = 0.85
    decision = get_llm_decision(mock_sensor_data, mock_health)

    if decision:
        print(f"\nLLM Decision: {decision['action']}")
        print(f"Reasoning: {decision['reasoning']}")
    else:
        print("\nFailed to get a decision from the LLM.")


