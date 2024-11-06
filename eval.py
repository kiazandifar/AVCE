# eval.py

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)


# Define the custom action wrapper used in main.py
class CarRacingWrapper(gym.Wrapper):
    """
    Wrapper for the CarRacing environment that converts continuous to discrete actions.
    """

    def __init__(self, env):
        super().__init__(env)
        # Define discrete action space with 5 actions
        self.action_space = gym.spaces.Discrete(5)

    def step(self, action):
        # Ensure action is an integer for dictionary lookup
        if isinstance(action, np.ndarray):
            action = int(action.item())  # Extract the integer if it's a 1-element array
        elif isinstance(action, (list, tuple)):
            action = int(action[0])

        # Convert discrete actions to continuous actions expected by the environment
        action_map = {
            0: np.array([0.0, 0.0, 0.0], dtype=np.float32),  # No action
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Gas
            2: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Brake
            3: np.array([0.0, 0.0, -1.0], dtype=np.float32),  # Left turn
            4: np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Right turn
        }
        continuous_action = action_map[action]
        return self.env.step(continuous_action)

    def reset(self, **kwargs):
        # Handle both old and new gym API
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            return obs, info
        return result


# Function to create and wrap the environment
def create_env():
    env = gym.make('CarRacing-v3', continuous=True, render_mode="rgb_array")
    env = CarRacingWrapper(env)  # Apply the custom wrapper
    return Monitor(env)


# Load the trained model
dqn_agent = DQN.load("logs/best_model/best_model.zip")

# Initialize the evaluation environment
eval_env = create_env()


# Function to evaluate the agent
def evaluate_agent(agent, env, num_episodes=10):
    all_rewards = []
    for episode in range(num_episodes):
        state, info = env.reset()
        terminated = truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            action, _ = agent.predict(state, deterministic=True)

            # Check and adjust action if necessary
            if isinstance(action, np.ndarray):
                action = int(action.item())
            elif isinstance(action, (list, tuple)):
                action = int(action[0])

            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        all_rewards.append(episode_reward)

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    return mean_reward, std_reward


# Evaluate the trained agent
mean_reward, std_reward = evaluate_agent(dqn_agent, eval_env, num_episodes=10)
print(f"Mean reward over 10 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")


# Function to visualize the agent's performance
def visualize_agent_performance(agent, env, num_episodes=1):
    frames = []

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        terminated = truncated = False
        score = 0
        step = 0

        while not (terminated or truncated):
            frame = env.render()
            if len(frame) > 0 and frame.shape == (400, 600, 3):  # Ensure the frame is in the correct format
                frames.append(frame)

            # Get action from the trained agent
            action, _ = agent.predict(state, deterministic=True)

            # Check and adjust action if necessary
            if isinstance(action, np.ndarray):
                action = int(action.item())
            elif isinstance(action, (list, tuple)):
                action = int(action[0])

            state, reward, terminated, truncated, info = env.step(action)
            score += reward

            # Debugging statements
            logging.info(
                f"Step: {step}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            step += 1

        # Log the accumulated rewards at the end of the episode
        logging.info(f"Episode: {episode} Score: {score}")

    env.close()

    # Check if frames were captured
    if not frames:
        print("No frames captured. Visualization aborted.")
        return

    # Save frames as a video file
    height, width, layers = frames[0].shape
    video_name = 'agent_performance.mp4'
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in frames:
        # Convert RGB to BGR (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"Video saved as {video_name}")

    # Display the video
    fig, ax = plt.subplots()
    plt.axis('off')

    def update_frame(i):
        ax.imshow(frames[i])
        return [ax]

    ani = animation.FuncAnimation(fig, update_frame, frames=len(frames), blit=True, interval=50)
    plt.show()


# Visualize the agent's performance
visualize_agent_performance(dqn_agent, eval_env, num_episodes=1)

# Load and plot evaluation results from 'evaluations.npz'
data = np.load('logs/results/evaluations.npz')
print("Keys in the npz file:", data.files)

# Extract timesteps and results
timesteps = data['timesteps']
results = data['results']

# Calculate mean and standard deviation of rewards at each checkpoint
mean_rewards = [np.mean(r) for r in results]
std_rewards = [np.std(r) for r in results]

# Plot rewards at each evaluation checkpoint
plt.figure(figsize=(10, 5))
for i, reward in enumerate(results):
    plt.plot(range(len(reward)), reward,
             label=f'Timestep {timesteps[i]} (Mean: {mean_rewards[i]:.2f}, Std: {std_rewards[i]:.2f})')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Rewards at Each Evaluation Checkpoint')
plt.legend()
plt.show()

# Plot mean reward over time
plt.figure(figsize=(10, 5))
plt.plot(timesteps, mean_rewards, label='Mean Reward')
plt.fill_between(timesteps, np.array(mean_rewards) - np.array(std_rewards),
                 np.array(mean_rewards) + np.array(std_rewards), alpha=0.2, label='Standard Deviation')
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.title('Mean Reward Over Time')
plt.legend()
plt.show()
