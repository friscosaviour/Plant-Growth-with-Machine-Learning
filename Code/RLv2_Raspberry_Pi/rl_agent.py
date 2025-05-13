import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random
from collections import deque

# --- Agent Configuration ---
STATE_SIZE = 4 # moisture, light, temp, humidity
ACTION_SIZE = 4 # Defined previously
LEARNING_RATE = 0.001
GAMMA = 0.95 # Discount factor for future rewards
EPSILON_START = 1.0 # Exploration rate start
EPSILON_END = 0.01 # Exploration rate end
EPSILON_DECAY_STEPS = 10000 # How fast to decay epsilon
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 100 # Steps between updating the target network

# --- Normalization (Use same logic as data_logger.py) ---
# Placeholder for sensor ranges - load from config or define here
SENSOR_RANGES = {
    'soil_moisture': {'min': 0, 'max': 1023},
    'light_intensity': {'min': 0, 'max': 4095},
    'temperature': {'min': 0, 'max': 50},
    'humidity': {'min': 0, 'max': 100}
}

def normalize_state(raw_state):
    """Normalizes raw sensor readings [soil, light, temp, hum]"""
    n_soil = (raw_state[0] - SENSOR_RANGES['soil_moisture']['min']) / (SENSOR_RANGES['soil_moisture']['max'] - SENSOR_RANGES['soil_moisture']['min'])
    n_light = (raw_state[1] - SENSOR_RANGES['light_intensity']['min']) / (SENSOR_RANGES['light_intensity']['max'] - SENSOR_RANGES['light_intensity']['min'])
    n_temp = (raw_state[2] - SENSOR_RANGES['temperature']['min']) / (SENSOR_RANGES['temperature']['max'] - SENSOR_RANGES['temperature']['min'])
    n_hum = (raw_state[3] - SENSOR_RANGES['humidity']['min']) / (SENSOR_RANGES['humidity']['max'] - SENSOR_RANGES['humidity']['min'])
    # Clip values to ensure they are within [0, 1] after normalization
    return np.clip([n_soil, n_light, n_temp, n_hum], 0.0, 1.0)

# --- DQN Model ---
def build_dqn_model(state_size, action_size, learning_rate):
    """Builds a simple Keras DQN model."""
    model = tf.keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(32, activation='relu'), # Keep layers small for Pi
        layers.Dense(32, activation='relu'),
        layers.Dense(action_size, activation='linear') # Output Q-values for each action
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse') # Mean Squared Error for Q-learning
    return model

# --- DQNAgent Class (Simplified) ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        # Calculate decay factor to reach min epsilon in specified steps
        self.epsilon_decay = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS
        self.learning_rate = LEARNING_RATE

        # Main model (gets trained)
        self.model = build_dqn_model(state_size, action_size, self.learning_rate)
        # Target model (predicts target Q values for stability)
        self.target_model = build_dqn_model(state_size, action_size, self.learning_rate)
        self.update_target_model() # Initialize target model weights
        self.train_step_counter = 0

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())
        print("Target model updated.")

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Chooses an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        # Exploit: predict Q-values and choose the best action
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0) # Add batch dimension
        act_values = self.model.predict(state_tensor)
        return np.argmax(act_values[0]) # Returns action index

    def replay(self, batch_size):
        """Trains the model using randomly sampled experiences from the buffer."""
        if len(self.memory) < batch_size:
            return # Not enough memory yet

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Predict Q-values for the next states using the target network
        target_q_next = self.target_model.predict(next_states)
        # Calculate the target Q-value using the Bellman equation: Q_target = R + gamma * max_a'(Q_target(s', a'))
        # If the episode is done, the target Q-value is just the reward
        target_q = rewards + self.gamma * np.amax(target_q_next, axis=1) * (1 - dones)

        # Predict current Q-values using the main network
        current_q = self.model.predict(states)
        # Update the Q-value for the action actually taken
        for i in range(batch_size):
            current_q[i][actions[i]] = target_q[i]

        # Train the main model
        self.model.fit(states, current_q, epochs=1, verbose=0, batch_size=batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon) # Ensure it doesn't go below min

        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % TARGET_UPDATE_FREQ == 0:
            self.update_target_model()

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model() # Ensure target model is also updated

    def save(self, name):
        self.model.save_weights(name)

# --- Training Loop (Conceptual - Needs Environment Interaction) ---
# This part would typically run *offline* first, possibly using logged data
# or a simulator, before deploying the trained model to the Pi.
# Online training on the Pi is possible but needs careful resource management.

# agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
# episodes = 1000
# for e in range(episodes):
#     # Reset environment/get initial state (e.g., from CSV or simulator)
#     # state = normalize_state(get_initial_sensor_readings())
#     # state = np.reshape(state, [1, STATE_SIZE])
#     # total_reward = 0
#     # for time_step in range(MAX_STEPS_PER_EPISODE):
#         # action = agent.act(state)
#         # # Execute action (pump, lights) -> Get next state, reward, done
#         # # next_state_raw, reward, done = environment.step(action)
#         # # next_state = normalize_state(next_state_raw)
#         # # next_state = np.reshape(next_state, [1, STATE_SIZE])
#         # # agent.remember(state, action, reward, next_state, done)
#         # # state = next_state
#         # # total_reward += reward
#         # # agent.replay(BATCH_SIZE) # Train the agent
#         # # if done: break
#     # print(f"Episode {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
#     # if e % 50 == 0: # Save periodically
#     #     agent.save(f"dqn_plant_model_{e}.h5")
