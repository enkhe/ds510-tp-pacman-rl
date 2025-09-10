import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from tensorboard.backend.event_processing import event_accumulator
from datetime import datetime

def plot_individual_metrics(log_dir, output_dir, resolution):
    """
    Create and save individual plots for each metric from a TensorBoard event file.
    """
    try:
        # Find the TensorBoard event file
        event_file = None
        for root, _, files in os.walk(log_dir):
            for file in files:
                if "events.out.tfevents" in file:
                    event_file = os.path.join(root, file)
                    break
            if event_file:
                break

        if not event_file:
            print(f"No TensorBoard event file found in {log_dir}.")
            return

        print(f"Reading data from: {event_file}")
        ea = event_accumulator.EventAccumulator(event_file,
                                                size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving graphs to: {output_dir}")

        # Get current timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Set resolution
        dpi = 300 if resolution == '4k' else 150

        # Extract and plot each scalar metric
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            df = pd.DataFrame([(e.step, e.value) for e in events], columns=['Step', 'Value'])
            
            if df.empty:
                print(f"Skipping empty metric: {tag}")
                continue

            plt.figure(figsize=(12, 7))
            plt.plot(df['Step'], df['Value'], label=tag)
            
            # Formatting the plot
            plt.title(f"Metric: {tag.replace('/', ' / ').title()}", fontsize=16)
            plt.xlabel("Steps", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Generate a clean filename
            clean_tag = tag.replace('/', '_')
            filename = f"{clean_tag}_{timestamp}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Save the figure
            plt.savefig(output_path, dpi=dpi)
            print(f"Saved plot: {output_path}")
            plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate individual plots for each metric from a TensorBoard log file.')
    parser.add_argument('-i', '--log-dir', type=str, required=True, help='Path to the log directory containing the TensorBoard event file.')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='Directory to save the output plot images.')
    parser.add_argument('-r', '--resolution', type=str, default='4k', choices=['hd', '4k'], help='Resolution of the output images.')
    
    args = parser.parse_args()
    
    plot_individual_metrics(args.log_dir, args.output_dir, args.resolution)
