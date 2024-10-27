import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_rewards(out_dir):
    # Load data from CSV file
    csv_file_path = os.path.join(out_dir, "instantaneous_rewards.csv")
    data = pd.read_csv(csv_file_path)

    # Calculate a rolling mean and standard deviation
    window_size = 80  # Adjust this window size based on the data variance
    data["Rolling Mean"] = data["Average Reward"].rolling(window=window_size).mean()
    data["Rolling Std"] = data["Average Reward"].rolling(window=window_size).std()

    plt.figure(figsize=(10, 4))  # Adjusted figure size for a thinner plot

    # Plot the rolling mean with a shaded region for variance
    plt.plot(data["Step"], data["Rolling Mean"], color="blue", linewidth=1.5)
    plt.fill_between(
        data["Step"],
        data["Rolling Mean"] - data["Rolling Std"],
        data["Rolling Mean"] + data["Rolling Std"],
        color="blue",
        alpha=0.2,
    )

    # Plot settings
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Step over Time")
    plt.grid(True)
    plt.tight_layout()

    # Show or save the plot
    plt.savefig("rewards_plot.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot average rewards over time from CSV file.")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Directory containing the CSV file")
    args = parser.parse_args()

    plot_rewards(args.out_dir)
