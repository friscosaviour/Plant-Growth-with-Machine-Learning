import tensorflow as tf
import numpy as np
import yaml
import os
from collections import deque
import random

from environment import PlantEnv

def create_dqn_model(input_shape, num_actions):
    """Creates a Deep Q-Network model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_agent(config):
    """Trains the DQN agent."""
    env = PlantEnv(config)
    num_actions = env.action_space.n
    input_shape = env.observation_space.shape

    model = create_dqn_model(input_shape, num_actions)
    target_model = create_dqn_model(input_shape, num_actions)
    target_model.set_weights(model.get_weights())

    replay_buffer = deque(maxlen=2000)
    batch_size = 64
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    num_episodes = 500

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, *input_shape])
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(state, verbose=0)
                action = np.argmax(q_values[0])

            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, *input_shape])
            total_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            if len(replay_buffer) > batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                for s, a, r, ns, d in minibatch:
                    target = r
                    if not d:
                        target = r + gamma * np.amax(target_model.predict(ns, verbose=0)[0])
                    target_f = model.predict(s, verbose=0)
                    target_f[0][a] = target
                    model.fit(s, target_f, epochs=1, verbose=0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())
            print(f"Episode: {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Save the trained model
    model_dir = os.path.dirname(config['files']['model_path'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    saved_model_path = os.path.join(model_dir, "plant_model_saved_model")
    model.save(saved_model_path)
    print(f"Model saved in SavedModel format at {saved_model_path}")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_model_path = config['files']['model_path']
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted and saved in TFLite format at {tflite_model_path}")

if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    train_agent(config)
