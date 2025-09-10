import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_loss_from_csv(log_folder, output_file, resolution='4k'):
    """
    Plots the training loss from a progress.csv file and saves it as a PNG.

    :param log_folder: Path to the directory containing the progress.csv file.
    :param output_file: Path to save the output PNG file.
    :param resolution: Image resolution ('4k' or '1080p').
    """
    csv_path = os.path.join(log_folder, "progress.csv")

    if not os.path.exists(csv_path):
        print(f"Error: Could not find progress.csv in {log_folder}")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"Error: {csv_path} is empty. Training may have just started or failed.")
        return

    # Check if 'train/loss' column exists
    if 'train/loss' not in df.columns:
        print(f"Error: 'train/loss' column not found in {csv_path}. Available columns: {list(df.columns)}")
        return

    # Set plot resolution
    dpi_map = {'4k': 300, '1080p': 150}
    image_dpi = dpi_map.get(resolution.lower(), 300)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot the training loss
    ax.plot(df['time/total_timesteps'], df['train/loss'], label='Training Loss', color='orangered', linewidth=2)

    # --- Labels and Title ---
    ax.set_title('PPO Training Loss for Ms. Pac-Man', fontsize=20, pad=20)
    ax.set_xlabel('Timesteps', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    
    # --- Legend and Grid ---
    ax.legend(fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Aesthetics ---
    fig.tight_layout()

    # Save the figure
    plt.savefig(output_file, dpi=image_dpi)
    print(f"Graph saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss from a Stable Baselines3 log file.")
    parser.add_argument("-i", "--log-folder", required=True, type=str, help="Path to the log directory containing progress.csv.")
    parser.add_argument("-o", "--output-file", default="pacman_loss_graph.png", type=str, help="Path to save the output PNG file.")
    parser.add_argument("-r", "--resolution", default="4k", type=str, choices=['4k', '1080p'], help="Output image resolution.")
    
    args = parser.parse_args()

    plot_loss_from_csv(args.log_folder, args.output_file, args.resolution)
