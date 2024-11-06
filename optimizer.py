# optimizer.py

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)


# Define the environment creation function
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


# Initialize the training and evaluation environments
env = create_env()
eval_env = create_env()

# Hyperparameter tuning
best_mean_reward = -np.inf
best_hyperparams = None

# List of hyperparameters to try
learning_rates = [1e-4, 5e-4, 1e-3]
gammas = [0.98, 0.99, 0.995]
batch_sizes = [32, 64, 128]

# Device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter tuning loop
for lr in learning_rates:
    for gamma in gammas:
        for batch_size in batch_sizes:
            logging.info(f"Training with learning_rate={lr}, gamma={gamma}, batch_size={batch_size}")

            # Initialize DQN agent with current hyperparameters
            agent = DQN(
                "CnnPolicy",
                env,
                learning_rate=lr,
                gamma=gamma,
                buffer_size=100000,
                learning_starts=1000,
                target_update_interval=1000,
                train_freq=4,
                gradient_steps=1,  # Fixed value for gradient steps; batch size controls batch updates
                batch_size=batch_size,
                verbose=0,
                device=device
            )

            # Train the agent for a reduced number of timesteps for initial evaluation
            agent.learn(total_timesteps=50000)

            # Evaluate the agent
            mean_reward, std_reward = evaluate_policy(agent, eval_env, n_eval_episodes=10)
            logging.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

            # Update best hyperparameters if current mean reward is higher
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_hyperparams = (lr, gamma, batch_size)

                # Save the best agent
                agent.save("best_dqn_car_racing")
                logging.info(f"New best agent saved with mean reward {best_mean_reward:.2f}")

            # Clean up agent and free GPU memory
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Output the best hyperparameters
logging.info(
    f"Best hyperparameters: learning_rate={best_hyperparams[0]}, gamma={best_hyperparams[1]}, batch_size={best_hyperparams[2]}")
logging.info(f"Best mean reward: {best_mean_reward:.2f}")

# Close environments
env.close()
eval_env.close()