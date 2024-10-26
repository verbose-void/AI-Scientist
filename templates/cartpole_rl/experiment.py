import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import json
import time

# Hyperparameters
learning_rate = 0.01
gamma = 0.99
episodes = 1000
max_timesteps = 500
seeds = [0]

# Policy network (linear policy)
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim)

    def forward(self, x):
        logits = self.fc(x)
        return Categorical(logits=logits)

def discount_rewards(rewards, gamma):
    discounted_rewards = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    return torch.tensor(discounted_rewards)

def reinforce(out_dir, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    policy = Policy(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    rewards_all_episodes = []
    steps_to_convergence = episodes  # Default in case convergence isn't reached

    for episode in range(episodes):
        state, _ = env.reset()
        rewards, log_probs = [], []

        for t in range(max_timesteps):
            state = torch.tensor(np.array(state), dtype=torch.float32)
            dist = policy(state)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            state, reward, done, _, _ = env.step(action.item())

            rewards.append(reward)
            if done:
                break

        # Store total reward for this episode
        rewards_all_episodes.append(sum(rewards))

        # Compute discounted rewards
        discounted_rewards = discount_rewards(rewards, gamma)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Compute loss and optimize
        loss = -torch.stack(log_probs) * discounted_rewards
        loss = loss.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check convergence
        if sum(rewards) >= 500:
            steps_to_convergence = episode + 1
            print(f"Solved for seed {seed} in {steps_to_convergence} episodes!")
            break

    # Save rewards data and return convergence steps
    np.save(os.path.join(out_dir, f"rewards_seed_{seed}.npy"), np.array(rewards_all_episodes))
    env.close()
    return steps_to_convergence

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Track steps to convergence and time for each seed
    convergence_steps = []
    start_time = time.time()

    for seed in seeds:
        steps = reinforce(args.out_dir, seed)
        convergence_steps.append(steps)

    total_time = time.time() - start_time

    # Calculate means and save final_info.json
    experiment_info = {
        "steps_to_convergence": {"means": np.mean(convergence_steps)},
        "total_time_seconds": {"means": total_time}
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(experiment_info, f, indent=4)

    print("All experiments completed.")
    print(f"Total time: {experiment_info['total_time_seconds']['means']} seconds")
    print(f"Mean steps to convergence: {experiment_info['steps_to_convergence']['means']}")
