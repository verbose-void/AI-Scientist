import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_rewards(out_dir):
    # Get all .npy files in the directory
    reward_files = glob.glob(os.path.join(out_dir, "rewards_seed_*.npy"))

    plt.figure(figsize=(10, 6))

    # Loop through each file and plot rewards
    for file in reward_files:
        rewards = np.load(file)
        plt.plot(rewards, label=f"Seed {file.split('_')[-1].split('.')[0]}")

    # Plot settings
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward over Time for Different Seeds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig("rewards.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot rewards over time for all seeds.")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Directory containing reward files")
    args = parser.parse_args()

    plot_rewards(args.out_dir)
