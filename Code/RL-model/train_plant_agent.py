# train_plant_agent.py
import numpy as np
import pandas as pd
from PlantEnvironment import PlantGrowthEnv # Make sure this file is in the same directory
from QLearningAgent import QLearningAgent   # Make sure this file is in the same directory
import matplotlib.pyplot as plt

def create_dummy_csv(filepath="dummy_plant_data.csv", num_rows=1000):
    """Creates a dummy CSV file for testing."""
    data = {
        'soil_moisture': np.random.uniform(0.1, 0.9, num_rows), # %
        'humidity': np.random.uniform(0.3, 0.8, num_rows),      # %
        'temperature': np.random.uniform(15, 35, num_rows),     # Celsius
        'light_intensity': np.random.uniform(0.2, 1.0, num_rows) # Normalized 0-1
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Dummy CSV created at {filepath}")
    return filepath

if __name__ == "__main__":
    # --- Configuration ---
    # Create a dummy CSV if you don't have one
    # CSV_PATH = "your_plant_data.csv" # Replace with your actual CSV path
    CSV_PATH = create_dummy_csv("plant_growth_sim_data.csv", num_rows=500)

    NUM_EPISODES = 2000 # Number of training episodes
    MAX_STEPS_PER_EPISODE = 200 # Max steps in one episode

    # Define target ranges (adjust these to what's optimal for your plant)
    # These values should generally be within the 0-1 range if you normalize,
    # or in their raw sensor units if you don't normalize before binning.
    # The example env uses 0-1 for soil moisture from its simulation logic.
    # For others, it takes from CSV. Ensure consistency.
    # Let's assume CSV values are:
    # soil_moisture: 0-1 (after any potential pre-processing in your CSV)
    # humidity: 0-1 (relative humidity percentage / 100)
    # temperature: Celsius (the env's bins handle the raw values)
    # light_intensity: 0-1 (normalized light reading)

    TARGET_SOIL_MOISTURE = (0.40, 0.60) # e.g. 40%-60%
    TARGET_HUMIDITY = (0.50, 0.70)      # e.g. 50%-70%
    TARGET_TEMPERATURE = (20, 28)       # e.g. 20-28°C
    TARGET_LIGHT = (0.6, 0.9)           # e.g. 60-90% of max light

    # --- Initialize Environment and Agent ---
    env = PlantGrowthEnv(
        csv_path=CSV_PATH,
        target_soil_moisture=TARGET_SOIL_MOISTURE,
        target_humidity=TARGET_HUMIDITY,
        target_temperature=TARGET_TEMPERATURE,
        target_light=TARGET_LIGHT,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE
    )

    agent = QLearningAgent(
        action_space_size=env.action_space.n,
        observation_space_shape=env.observation_space.nvec, # For MultiDiscrete
        learning_rate=0.1,
        discount_factor=0.95, # একটু কম ডিসকাউন্ট কারণ আমরা তাৎক্ষণিক রিওয়ার্ড চাই
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=NUM_EPISODES * MAX_STEPS_PER_EPISODE * 0.5 # Decay over half of total steps
    )
    # agent.load_q_table() # Uncomment to load a previously saved Q-table

    # --- Training Loop ---
    episode_rewards = []
    successful_episodes = 0

    for episode in range(NUM_EPISODES):
        state_tuple = env.reset()
        total_reward = 0
        done = False

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state_tuple)
            next_state_tuple, reward, done, info = env.step(action)
            agent.learn(state_tuple, action, reward, next_state_tuple, done)

            state_tuple = next_state_tuple
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)
        if total_reward > 0: # Arbitrary definition of "successful"
            successful_episodes +=1

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {episode + 1}/{NUM_EPISODES}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            # env.render() # Optionally render last step of an episode

    print("Training finished.")
    agent.save_q_table() # Save the learned Q-table

    # --- Plotting Results ---
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Total Reward per Episode')
    # Moving average
    window_size = 50
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(episode_rewards)), moving_avg, label=f'{window_size}-Episode Moving Average', color='red')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("RL Agent Training Rewards for Plant Growth")
    plt.legend()
    plt.grid(True)
    plt.savefig("plant_growth_rl_rewards.png")
    plt.show()

    print(f"Total successful episodes (reward > 0): {successful_episodes}/{NUM_EPISODES}")
    # You can inspect the Q-table size:
    print(f"Number of states explored in Q-table: {len(agent.q_table)}")


    # --- Example of running the trained agent (evaluation) ---
    print("\n--- Running Trained Agent (Evaluation Example) ---")
    state_tuple = env.reset()
    total_eval_reward = 0
    agent.epsilon = 0 # Turn off exploration for evaluation
    for _ in range(MAX_STEPS_PER_EPISODE):
        env.render()
        action = agent.choose_action(state_tuple) # Should be deterministic now
        print(f"Chosen Action: {'Water' if action == 1 else 'Do Nothing'}")
        next_state_tuple, reward, done, info = env.step(action)
        state_tuple = next_state_tuple
        total_eval_reward += reward
        if done:
            break
    print(f"Total evaluation reward: {total_eval_reward}")

    env.close()