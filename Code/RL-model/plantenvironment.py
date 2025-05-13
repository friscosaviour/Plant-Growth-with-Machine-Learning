# PlantEnvironment.py
import numpy as np
import pandas as pd
import gym
from gym import spaces

class PlantGrowthEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, csv_path,
                 target_soil_moisture=(0.4, 0.6), # Target 40-60%
                 target_humidity=(0.5, 0.7),      # Target 50-70%
                 target_temperature=(20, 28),     # Target 20-28 C
                 target_light=(0.6, 0.9),         # Target 60-90% (normalized)
                 max_steps_per_episode=200):

        super(PlantGrowthEnv, self).__init__()

        self.data = pd.read_csv(csv_path)
        if not all(col in self.data.columns for col in ['soil_moisture', 'humidity', 'temperature', 'light_intensity']):
            raise ValueError("CSV must contain 'soil_moisture', 'humidity', 'temperature', 'light_intensity' columns")

        # Normalize data to be roughly between 0 and 1 for easier binning, if not already
        # This is a placeholder; you'd ideally normalize based on sensor min/max
        # For simplicity, we'll assume CSV data is already somewhat scaled or we'll bin raw values
        self.sensor_mins = self.data.min()
        self.sensor_maxs = self.data.max()

        # Define action space: 0 = Do Nothing, 1 = Water
        self.action_space = spaces.Discrete(2)

        # Define observation space (binned sensor values)
        # Let's use 5 bins for each sensor for simplicity
        self.num_bins = 5
        # The observation space will be a MultiDiscrete space where each element
        # is the bin index for that sensor.
        # E.g., [bin_soil_moisture, bin_humidity, bin_temp, bin_light]
        self.observation_space = spaces.MultiDiscrete([self.num_bins] * 4)

        # Environment parameters
        self.target_soil_moisture = target_soil_moisture
        self.target_humidity = target_humidity
        self.target_temperature = target_temperature
        self.target_light = target_light
        self.max_steps_per_episode = max_steps_per_episode

        # Internal state representation (continuous, before discretization)
        # We'll use the first row of the CSV for initial non-soil_moisture values
        self.current_soil_moisture = self.data['soil_moisture'].iloc[0] # Initial, can be random
        self.current_humidity = self.data['humidity'].iloc[0]
        self.current_temperature = self.data['temperature'].iloc[0]
        self.current_light_intensity = self.data['light_intensity'].iloc[0]

        self.current_step = 0
        self.csv_row_index = 0 # To cycle through CSV for external factors

        # Create bins for discretization
        self._create_bins()

    def _create_bins(self):
        self.bins = {}
        # Use actual min/max from CSV for binning, or define reasonable fixed ranges
        # For simplicity, using CSV min/max here. Better to use expected sensor ranges.
        self.bins['soil_moisture'] = np.linspace(self.sensor_mins['soil_moisture'], self.sensor_maxs['soil_moisture'], self.num_bins + 1)[1:-1]
        self.bins['humidity'] = np.linspace(self.sensor_mins['humidity'], self.sensor_maxs['humidity'], self.num_bins + 1)[1:-1]
        self.bins['temperature'] = np.linspace(self.sensor_mins['temperature'], self.sensor_maxs['temperature'], self.num_bins + 1)[1:-1]
        self.bins['light_intensity'] = np.linspace(self.sensor_mins['light_intensity'], self.sensor_maxs['light_intensity'], self.num_bins + 1)[1:-1]
        # print("Bins created:", self.bins)


    def _discretize_state(self, continuous_state):
        soil_m, hum, temp, light = continuous_state
        # np.digitize returns the index of the bin (1-based), so subtract 1 for 0-based
        # Clip to ensure values are within [0, num_bins-1]
        state_discrete = [
            np.clip(np.digitize(soil_m, self.bins['soil_moisture']), 0, self.num_bins - 1),
            np.clip(np.digitize(hum, self.bins['humidity']), 0, self.num_bins - 1),
            np.clip(np.digitize(temp, self.bins['temperature']), 0, self.num_bins - 1),
            np.clip(np.digitize(light, self.bins['light_intensity']), 0, self.num_bins - 1)
        ]
        return tuple(state_discrete) # Q-table needs hashable states

    def _get_continuous_observation(self):
        return np.array([
            self.current_soil_moisture,
            self.current_humidity,
            self.current_temperature,
            self.current_light_intensity
        ])

    def reset(self):
        self.current_step = 0
        # Initialize soil moisture somewhat randomly or from CSV start
        self.current_soil_moisture = np.random.uniform(
            self.sensor_mins['soil_moisture'],
            self.sensor_maxs['soil_moisture'] * 0.7 # Start a bit dry
        )
        # self.current_soil_moisture = self.data['soil_moisture'].iloc[self.csv_row_index] # Alternative

        # Get other factors from the current CSV row
        current_csv_data = self.data.iloc[self.csv_row_index]
        self.current_humidity = current_csv_data['humidity']
        self.current_temperature = current_csv_data['temperature']
        self.current_light_intensity = current_csv_data['light_intensity']

        self.csv_row_index = (self.csv_row_index + 1) % len(self.data) # Cycle through CSV

        return self._discretize_state(self._get_continuous_observation())

    def step(self, action):
        self.current_step += 1
        done = False
        reward = 0

        # --- 1. Apply action ---
        if action == 1:  # Water
            self.current_soil_moisture += np.random.uniform(0.1, 0.2) # Water effect
            reward -= 0.1 # Cost of watering
        self.current_soil_moisture = np.clip(self.current_soil_moisture, 0, 1.0) # Assuming 0-1 scale after normalization

        # --- 2. Simulate environment dynamics ---
        # Soil dries out naturally (simplified)
        drying_factor = 0.01 + 0.02 * (self.current_temperature / 30) # Dries faster if hot
        self.current_soil_moisture -= np.random.uniform(drying_factor * 0.5, drying_factor * 1.5)
        self.current_soil_moisture = np.clip(self.current_soil_moisture, 0, 1.0)

        # Update external factors (humidity, temp, light) from the next CSV row
        # These are NOT directly controlled by the agent's actions in this simple model
        current_csv_data = self.data.iloc[self.csv_row_index]
        self.current_humidity = current_csv_data['humidity']
        self.current_temperature = current_csv_data['temperature']
        self.current_light_intensity = current_csv_data['light_intensity']
        self.csv_row_index = (self.csv_row_index + 1) % len(self.data)


        # --- 3. Calculate reward ---
        # Reward for soil moisture in target range
        if self.target_soil_moisture[0] <= self.current_soil_moisture <= self.target_soil_moisture[1]:
            reward += 1.0
        elif self.current_soil_moisture < self.target_soil_moisture[0] * 0.8: # Very dry
            reward -= 1.0
            # done = True # Optionally end episode if too dry
        elif self.current_soil_moisture > self.target_soil_moisture[1] * 1.2: # Very wet
            reward -= 1.0
            # done = True # Optionally end episode if too wet
        else:
            reward -= 0.2 # Slightly outside range

        # Bonus/penalty for other conditions being good/bad (even if not directly controlled)
        if not (self.target_humidity[0] <= self.current_humidity <= self.target_humidity[1]):
            reward -= 0.25
        if not (self.target_temperature[0] <= self.current_temperature <= self.target_temperature[1]):
            reward -= 0.25
        if not (self.target_light[0] <= self.current_light_intensity <= self.target_light[1]):
            reward -= 0.25


        # --- 4. Check if episode is done ---
        if self.current_step >= self.max_steps_per_episode:
            done = True

        # --- 5. Get new observation ---
        obs_continuous = self._get_continuous_observation()
        obs_discrete = self._discretize_state(obs_continuous)

        info = {'continuous_state': obs_continuous} # For debugging or richer info

        return obs_discrete, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"  Soil Moisture: {self.current_soil_moisture:.2f}")
            print(f"  Humidity: {self.current_humidity:.2f}")
            print(f"  Temperature: {self.current_temperature:.2f}")
            print(f"  Light Intensity: {self.current_light_intensity:.2f}")
            print(f"  Discretized State: {self._discretize_state(self._get_continuous_observation())}")

    def close(self):
        pass