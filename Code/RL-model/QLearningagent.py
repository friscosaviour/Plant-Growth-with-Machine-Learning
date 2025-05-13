# QLearningAgent.py
import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_space_size, observation_space_shape,
                 learning_rate=0.1, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=10000):

        self.action_space_size = action_space_size
        # observation_space_shape is a list like [num_bins_sensor1, num_bins_sensor2, ...]
        # For Q-table, states are tuples of bin indices.
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.current_step_for_epsilon = 0

    def choose_action(self, state_tuple):
        # Epsilon-greedy strategy
        self.current_step_for_epsilon +=1
        # Linear decay for epsilon
        self.epsilon = max(self.epsilon_end,
                           self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.current_step_for_epsilon / self.epsilon_decay_steps))

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space_size))  # Explore
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            # If all Q-values are zero for this state, pick randomly to encourage exploration
            if np.sum(self.q_table[state_tuple]) == 0:
                 return random.choice(range(self.action_space_size))
            return np.argmax(self.q_table[state_tuple])


    def learn(self, state_tuple, action, reward, next_state_tuple, done):
        # Q-Learning update rule
        # Q(s,a) = Q(s,a) + lr * [reward + gamma * max_a'(Q(s',a')) - Q(s,a)]
        # If done, the future reward (max_a'(Q(s',a'))) is 0
        current_q = self.q_table[state_tuple][action]
        max_future_q = np.max(self.q_table[next_state_tuple]) if not done else 0
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state_tuple][action] = new_q

    def save_q_table(self, filepath="q_table_plant.npy"):
        # Convert defaultdict to a regular dict for saving
        # This simple save might not be ideal for very large q_tables
        # Consider pickle or other serialization if it gets too big
        np.save(filepath, dict(self.q_table))
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath="q_table_plant.npy"):
        try:
            loaded_q_table_dict = np.load(filepath, allow_pickle=True).item()
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))
            self.q_table.update(loaded_q_table_dict)
            print(f"Q-table loaded from {filepath}")
        except FileNotFoundError:
            print(f"No Q-table found at {filepath}, starting fresh.")
        except Exception as e:
            print(f"Error loading Q-table: {e}. Starting fresh.")