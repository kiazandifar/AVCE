from dataclasses import dataclass
import torch
import numpy as np
# Import your training components
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import logging


# Custom Environment Wrapper (if not already defined elsewhere)
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


# Configuration class for training hyperparameters
@dataclass
class TrainingConfig:
    learning_rate: float = 0.001  # Best learning rate found
    gamma: float = 0.99  # Best gamma found
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.1
    total_timesteps: int = 150000  # Increased timesteps for final training
    buffer_size: int = 100000
    learning_starts: int = 1000
    target_update_interval: int = 1000
    train_freq: int = 4
    gradient_steps: int = 1
    batch_size: int = 64  # Best batch size found
    eval_freq: int = 10000
    checkpoint_freq: int = 10000
    log_dir: str = './logs'


# Trainer class to manage training logic
class CarRacingTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.env = self._create_env()
        self.eval_env = self._create_env()
        self.agent = self._create_agent()

    def _create_env(self):
        env = gym.make('CarRacing-v3', continuous=True, render_mode="rgb_array")
        return Monitor(CarRacingWrapper(env))

    def _create_agent(self):
        return DQN(
            "CnnPolicy",
            self.env,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            target_update_interval=self.config.target_update_interval,
            train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,
            batch_size=self.config.batch_size,
            verbose=1,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def train(self):
        logging.info("Starting training...")
        self.agent.learn(
            total_timesteps=self.config.total_timesteps,
            log_interval=10
        )
        logging.info("Training complete.")
        self.agent.save(f"{self.config.log_dir}/final_model")
        logging.info(f"Model saved to {self.config.log_dir}/final_model")


# Main function
def main():
    # Instantiate TrainingConfig with the best hyperparameters
    config = TrainingConfig(
        learning_rate=0.001,  # Best learning rate found
        gamma=0.99,  # Best gamma found
        batch_size=64,  # Best batch size found
        total_timesteps=150000  # Adjusted for final training
    )

    trainer = CarRacingTrainer(config)
    try:
        trainer.train()
    finally:
        trainer.env.close()
        trainer.eval_env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
