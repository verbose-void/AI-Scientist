import torch
from torch import nn


import os
import cv2
import numpy as np
import csv

import torch
import torch.nn as nn
import torch
import numpy as np
import gymnasium
from gymnasium.vector.utils import batch_space
from vizdoom import gymnasium_wrapper
import json
import time

# from gymnasium.envs.registration import register


# register(
#     id="VizdoomOblige-v0",
#     entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
#     kwargs={"scenario_file": "oblige.cfg"},
# )


DISPLAY_SIZE = (1280, 720)


class VizDoomVectorized:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.envs = [gymnasium.make("VizdoomCorridor-v0") for _ in range(num_envs)]
        self.dones = [False] * num_envs

        # Pre-allocate observation and reward tensors
        first_obs_space = self.envs[0].observation_space['screen']
        self.obs_shape = first_obs_space.shape
        self.observations = torch.zeros((num_envs, *self.obs_shape), dtype=torch.uint8)
        self.rewards = torch.zeros(num_envs, dtype=torch.float32)
        self.dones_tensor = torch.zeros(num_envs, dtype=torch.bool)

    def reset(self):
        for i in range(self.num_envs):
            obs, _ = self.envs[i].reset()
            self.observations[i] = torch.tensor(obs["screen"], dtype=torch.uint8)  # Fill the pre-allocated tensor
            self.dones[i] = False
        return self.observations

    def step(self, actions):
        """Steps all environments in parallel and fills pre-allocated tensors for observations, rewards, and dones.
           If an environment is done, it will automatically reset.
        """
        for i in range(self.num_envs):
            if self.dones[i]:
                # Reset the environment if it was done in the last step
                obs, _ = self.envs[i].reset()
                self.observations[i] = torch.tensor(obs["screen"], dtype=torch.uint8)  # Fill the pre-allocated tensor
                self.rewards[i] = 0  # No reward on reset
                self.dones_tensor[i] = False
                self.dones[i] = False
            else:
                obs, reward, terminated, truncated, _ = self.envs[i].step(actions[i])
                self.observations[i] = torch.tensor(obs["screen"], dtype=torch.uint8)  # Fill the pre-allocated tensor
                self.rewards[i] = reward
                done = terminated or truncated
                self.dones_tensor[i] = done
                self.dones[i] = done

        return self.observations, self.rewards, self.dones_tensor

    def close(self):
        for env in self.envs:
            env.close()

class DoomInteractor:
    """This thing manages the state of the environment and uses the agent
    to infer and step on the environment. This way is a bit easier
    because we can have environment mp while relying on the agent's
    internal vectorization, making gradients easier to accumulate.
    """

    def __init__(self, num_envs: int, watch: bool = False):
        self.num_envs = num_envs
        self.env = VizDoomVectorized(num_envs)  # Using the vectorized environment
        self.action_space = batch_space(self.env.envs[0].action_space, self.num_envs)
        self.watch = watch  # If True, OpenCV window will display frames from env 0

        # OpenCV window for visualization
        if self.watch:
            cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("screen", *DISPLAY_SIZE)

    def reset(self):
        return self.env.reset()

    def step(self, actions=None):
        if actions is None:
            actions = np.array([self.env.envs[i].action_space.sample() for i in range(self.num_envs)])

        # Step the environments with the sampled actions
        observations, rewards, dones = self.env.step(actions)

        # Show the screen from the 0th environment if watch is enabled
        if self.watch:
            # Convert tensor to numpy array for OpenCV display
            screen = observations[0].cpu().numpy()
            screen = cv2.resize(screen, DISPLAY_SIZE)

            cv2.imshow("screen", screen)
            cv2.waitKey(1)  # Display for 1 ms

        # Return the results
        return observations, rewards, dones

    def close(self):
        if self.watch:
            cv2.destroyAllWindows()  # Close the OpenCV window
        self.env.close()



class Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Doom action space is Discrete(8), so we want to output a distribution over 8 actions
        hidden_channels = 64
        embedding_size = 64

        self.hidden_channels = hidden_channels
        self.embedding_size = embedding_size

        # output should be a vector of 8 (our means)

        # obs_shape = (3, 180, 320)  # oblige
        obs_shape = (3, 240, 320)
        
        # 1. Observation Embedding: Convolutions + AdaptiveAvgPool + Flatten
        self.obs_embedding = nn.Sequential(
            torch.nn.LayerNorm(obs_shape),
            nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((1, 1)),
            # just simple averaging across all channels
            # nn.AvgPool2d(kernel_size=3, stride=2),
        )

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_channels, out_features=embedding_size),
            nn.Sigmoid(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
            nn.Sigmoid(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
            nn.Sigmoid(),
        )

        # Initialize hidden state to None; it will be dynamically set later
        self.hidden_state = None
        
        # 2. Embedding Blender: Combine the observation embedding and hidden state
        self.embedding_blender = nn.Sequential(
            nn.Linear(in_features=embedding_size * 2, out_features=embedding_size),
            nn.Sigmoid(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
            nn.Sigmoid(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
            nn.Sigmoid(),
        )

        # 3. Action Head: Map blended embedding to action logits
        self.action_head = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=8),
            nn.Sigmoid()
        )

    def reset(self, reset_mask: torch.Tensor):
        """Resets hidden states for the agent based on the reset mask."""
        batch_size = reset_mask.size(0)
        # Initialize hidden state to zeros where the reset mask is 1
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(batch_size, self.embedding_size, device=reset_mask.device)

        # Reset hidden states for entries where reset_mask is True (done flags)
        self.hidden_state[reset_mask == 1] = 0

    def forward(self, observations: torch.Tensor):
        # Reorder observations to (batch, channels, height, width) from (batch, height, width, channels)
        observations = observations.float().permute(0, 3, 1, 2)
        
        # Get batch size to handle hidden state initialization if needed
        batch_size = observations.size(0)

        # Initialize hidden state if it's the first forward pass
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.embedding_size, device=observations.device)

        # 1. Get the observation embedding
        obs_embedding = self.obs_embedding(observations)
        # print(obs_embedding.shape, "obs emb shape after conv")
        # average across all channels
        obs_embedding = obs_embedding.mean(dim=(2, 3))
        # print(obs_embedding.shape, "obs emb shape after avg")
        obs_embedding = self.embedding_head(obs_embedding)

        # Detach the hidden state from the computation graph (to avoid gradient tracking)
        hidden_state = self.hidden_state.detach()

        # 2. Concatenate the observation embedding with the hidden state
        combined_embedding = torch.cat((obs_embedding, hidden_state), dim=1)

        # 3. Blend embeddings
        blended_embedding = self.embedding_blender(combined_embedding)

        # Update the hidden state for the next timestep without storing gradients
        # Ensure we do not modify inplace - create a new tensor
        self.hidden_state = blended_embedding.detach().clone()

        # 4. Compute action logits
        action_logits = self.action_head(blended_embedding)

        # 5. Return the action distribution
        dist = self.get_distribution(action_logits)

        # HACK: maybe we need a more general way to do this, but store
        # the previous action in the hidden state
        actions = dist.sample()
        self.hidden_state[:, -1] = actions

        return actions, dist

    def get_distribution(self, means: torch.Tensor) -> torch.distributions.Categorical:
        """Returns a categorical distribution over the action space."""
        dist = torch.distributions.Categorical(probs=means)
        return dist

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def timestamp_name():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agent and environment
    agent = Agent().to(device)
    VSTEPS = 10_000
    NUM_ENVS = 48
    GRID_SIZE = int(np.ceil(np.sqrt(NUM_ENVS)))  # Dynamically determine the grid size
    LR = 1e-4
    NORM_WITH_REWARD_COUNTER = False
    WATCH = False  # Show live video frames

    interactor = DoomInteractor(NUM_ENVS, watch=WATCH)
    observations = interactor.reset()

    # Initialize metrics and counters
    cumulative_rewards = torch.zeros((NUM_ENVS,))
    step_counters = torch.zeros((NUM_ENVS,), dtype=torch.float32)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    best_episode_cumulative_reward = -float("inf")
    start_time = time.time()

    # Open CSV file for logging
    csv_file_path = os.path.join(out_dir, "instantaneous_rewards.csv")
    with open(csv_file_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Step", "Average Reward"])

        # Training loop
        for step_i in range(VSTEPS):
            optimizer.zero_grad()

            # Forward pass and environment step
            actions, dist = agent.forward(observations.float().to(device))
            entropy = dist.entropy()
            log_probs = dist.log_prob(actions)
            observations, rewards, dones = interactor.step(actions.cpu().numpy())

            # Update cumulative rewards and step counters
            cumulative_rewards += rewards
            step_counters += 1
            step_counters *= 1 - dones.float()

            # update best_episode_cumulative_reward
            best_episode_cumulative_reward = max(best_episode_cumulative_reward, cumulative_rewards.max().item())

            # Reset environments and agent hidden states as necessary
            agent.reset(dones)

            norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)

            # Calculate average reward for logging
            avg_reward = rewards.mean().item()

            # Log average reward to CSV
            csv_writer.writerow([step_i, avg_reward])

            # Loss calculation and optimization
            loss = (-log_probs * norm_rewards.to(device)).mean()
            loss.backward()
            optimizer.step()

            # Console logging every 100 steps
            if step_i % 100 == 0:
                print(f"Step {step_i} | Loss: {loss.item():.4f} | Entropy: {entropy.mean().item():.4f} | Avg Reward: {avg_reward:.4f}")
    
    # End of training metrics
    total_time = time.time() - start_time
    experiment_info = {
        "best_episode_cumulative_reward": {"means": best_episode_cumulative_reward},
        "total_time_seconds": {"means": total_time},
    }
    
    # Save final metrics to a JSON file
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(experiment_info, f, indent=4)

    print("Training complete.")
    print(f"Total time: {experiment_info['total_time_seconds']['means']} seconds")
    print(f"Best episode cumulative reward: {experiment_info['best_episode_cumulative_reward']['means']}")