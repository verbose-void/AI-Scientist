import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import time
import gymnasium as gym

from vizdoom import gymnasium_wrapper

# Hyperparameters
learning_rate = 1e-4
max_steps = 10_000
num_envs = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, obs_shape, num_actions, device=device):
        super(Agent, self).__init__()
        self.device = device  # Set the device attribute
        self.conv_layers = nn.Sequential(
            nn.LayerNorm(obs_shape),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.float().permute(0, 3, 1, 2).to(self.device)  # Ensure channel-first format and move to device
        x = self.conv_layers(x)
        x = x.mean(dim=(2, 3))  # Average pooling across spatial dimensions
        return self.fc(x)

def track_rewards(out_dir, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # envs = [VizDoomCustom() for _ in range(num_envs)]
    envs = [gym.make("VizdoomCorridor-v0") for _ in range(num_envs)]
    action_space = envs[0].action_space.n
    agent = Agent((3, 240, 320), action_space).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    cumulative_rewards = np.zeros(num_envs)
    total_rewards = []

    obs = np.array([env.reset()[0]["screen"] for env in envs])

    for step in range(max_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
        action_probs = agent(obs_tensor)
        actions = [torch.multinomial(action_probs[i], 1).item() for i in range(num_envs)]

        next_obs, rewards, dones = [], [], []
        for i, env in enumerate(envs):
            obs, reward, terminated, truncated, _ = env.step(actions[i])
            cumulative_rewards[i] += reward
            if terminated or truncated:
                total_rewards.append(cumulative_rewards[i])
                cumulative_rewards[i] = 0
                obs = env.reset()[0]["screen"]
            next_obs.append(obs)
            rewards.append(reward)
            dones.append(terminated or truncated)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=agent.device)
        norm_rewards = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        optimizer.zero_grad()
        loss = -(torch.log(action_probs.gather(1, torch.tensor(actions, device=agent.device).unsqueeze(-1))) * norm_rewards).mean()
        loss.backward()
        optimizer.step()

        obs = np.array(next_obs)

        if step % 100 == 0 or step == max_steps - 1:
            print(f"Step {step}/{max_steps}, Mean Reward: {np.mean(total_rewards)}")

    [env.close() for env in envs]
    np.save(os.path.join(out_dir, f"rewards_seed_{seed}.npy"), np.array(total_rewards))
    return np.mean(total_rewards)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Doom environment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    seed_rewards = []
    start_time = time.time()

    for seed in [0]:  # Expand seed list as needed
        mean_reward = track_rewards(args.out_dir, seed)
        seed_rewards.append(mean_reward)

    total_time = time.time() - start_time
    experiment_info = {
        "average_reward": {"means": np.mean(seed_rewards)},
        "total_time_seconds": {"means": total_time}
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(experiment_info, f, indent=4)

    print("Experiment complete.")
    print(f"Average reward: {experiment_info['average_reward']['means']}")
    print(f"Total time: {experiment_info['total_time_seconds']['means']} seconds")
