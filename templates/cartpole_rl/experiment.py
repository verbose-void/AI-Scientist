import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Hyperparameters
learning_rate = 0.01
gamma = 0.99
episodes = 1000
max_timesteps = 500

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

def reinforce():
    # Create environment and policy
    env = gym.make("CartPole-v1")
    policy = Policy(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

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

        # Compute discounted rewards
        discounted_rewards = discount_rewards(rewards, gamma)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Compute loss and optimize
        loss = -torch.stack(log_probs) * discounted_rewards
        loss = loss.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        print(f"Episode {episode + 1}\tTotal Reward: {sum(rewards)}")
        if sum(rewards) >= 500:
            print(f"Solved in {episode + 1} episodes!")
            break

    env.close()

if __name__ == "__main__":
    reinforce()
