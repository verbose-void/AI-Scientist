import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_rewards(out_dir):
    # Load data from CSV file
    csv_file_path = os.path.join(out_dir, "instantaneous_rewards.csv")
    data = pd.read_csv(csv_file_path)

    plt.figure(figsize=(10, 6))

    # Plot the average reward per step
    plt.plot(data["Step"], data["Average Reward"], label="Average Reward per Step")

    # Plot settings
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Step over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show or save the plot
    # plt.show()
    plt.savefig("rewards_plot.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot average rewards over time from CSV file.")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Directory containing the CSV file")
    args = parser.parse_args()

    plot_rewards(args.out_dir)