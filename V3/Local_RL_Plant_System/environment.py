import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PlantEnv(gym.Env):
    """Custom Gymnasium environment for training a plant care RL agent."""

    def __init__(self, config):
        super(PlantEnv, self).__init__()

        self.config = config['environment']

        # Define action space: 0: nothing, 1: water, 2: light on, 3: water and light on
        self.action_space = spaces.Discrete(4)

        # Define observation space: [soil_moisture, temp, humidity, light_on, health]
        # Using practical ranges for the simulation
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([1023, 40, 100, 1, 100]),
            dtype=np.float32
        )

        # Initial state
        self.state = self._get_initial_state()
        self.day_step = 0

    def _get_initial_state(self):
        """Returns a random initial state for the environment."""
        return np.array([
            np.random.uniform(300, 700), # soil moisture
            np.random.uniform(18, 25),   # temperature
            np.random.uniform(50, 70),   # humidity
            0,                           # light is initially off
            np.random.uniform(20, 60)    # initial health
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self._get_initial_state()
        self.day_step = 0
        return self.state, {}

    def step(self, action):
        self.day_step += 1

        # Unpack state
        soil_moisture, temp, humidity, light_on, health = self.state

        # Simulate action effects
        if action == 1 or action == 3: # Water
            soil_moisture += np.random.uniform(150, 250)
            soil_moisture = min(soil_moisture, 1023)

        if action == 2 or action == 3: # Light on
            light_on = 1
        else: # Light off
            light_on = 0

        # Simulate natural environmental changes over one step
        # Soil dries out
        soil_moisture -= np.random.uniform(30, 60)
        soil_moisture = max(soil_moisture, 0)

        # Temperature and humidity fluctuate slightly
        temp += np.random.uniform(-0.5, 0.5)
        humidity += np.random.uniform(-2, 2)

        # Simulate health change based on conditions
        health_change = self._calculate_health_change(soil_moisture, temp, humidity, light_on)
        health += health_change
        health = np.clip(health, 0, 100)

        # Update state
        self.state = np.array([soil_moisture, temp, humidity, light_on, health])

        # Calculate reward
        reward = self._calculate_reward(self.state)

        # Check if done (e.g., after a simulated month)
        done = self.day_step > 30 or health <= 0

        return self.state, reward, done, False, {}

    def _calculate_health_change(self, soil, temp, hum, light):
        """Simulates how the plant's health changes based on the environment."""
        health_change = 0
        # Health improves if conditions are near ideal
        if self.config['ideal_soil_moisture'] * 0.8 < soil < self.config['ideal_soil_moisture'] * 1.2:
            health_change += 0.5
        else:
            health_change -= 0.5

        if self.config['ideal_temperature'] * 0.9 < temp < self.config['ideal_temperature'] * 1.1:
            health_change += 0.2
        else:
            health_change -= 0.2

        if light == 1:
            health_change += 0.3 # Assume light is generally good

        return health_change

    def _calculate_reward(self, state):
        """Calculates the reward for the current state."""
        soil, temp, hum, light, health = state

        # Reward for being close to ideal conditions
        soil_reward = -abs(soil - self.config['ideal_soil_moisture']) / 100.0
        temp_reward = -abs(temp - self.config['ideal_temperature'])
        hum_reward = -abs(hum - self.config['ideal_humidity']) / 10.0

        # Reward for health
        health_reward = health / 10.0

        # Penalty for dead plant
        if health <= 0:
            return -100

        return soil_reward + temp_reward + hum_reward + health_reward

    def render(self, mode='human'):
        print(f"Step: {self.day_step}")
        print(f"State: {self.state}")
        print(f"Reward: {self._calculate_reward(self.state)}")
