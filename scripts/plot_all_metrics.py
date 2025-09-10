import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import json

def plot_all_metrics(log_dir, output_file, resolution):
    """
    Plot metrics from monitor.csv and progress.csv files in a log directory.
    """
    try:
        # --- Plotting from monitor.csv ---
        monitor_files = [f for f in os.listdir(log_dir) if f.endswith('.monitor.csv')]
        if not monitor_files:
            print("No monitor.csv files found in the directory.")
            return

        # Concatenate all monitor files
        all_monitor_data = []
        for monitor_file in monitor_files:
            path = os.path.join(log_dir, monitor_file)
            with open(path, 'r') as f:
                # Skip the first line which is a JSON header
                f.readline()
                df = pd.read_csv(f)
                all_monitor_data.append(df)
        
        monitor_df = pd.concat(all_monitor_data, ignore_index=True)
        monitor_df['t'] = pd.to_datetime(monitor_df['t'], unit='s')


        # --- Plotting from progress.csv ---
        progress_file = os.path.join(log_dir, 'progress.csv')
        if not os.path.exists(progress_file):
            print("progress.csv not found. Skipping detailed metrics.")
            progress_df = None
        else:
            progress_df = pd.read_csv(progress_file)

        # --- Create Subplots ---
        num_monitor_plots = len(monitor_df.columns)
        num_progress_plots = len(progress_df.columns) if progress_df is not None else 0
        total_plots = num_monitor_plots + num_progress_plots

        if total_plots == 0:
            print("No data to plot.")
            return

        cols = 3
        rows = (total_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), constrained_layout=True)
        fig.suptitle(f'Training Metrics for {os.path.basename(log_dir)}', fontsize=20)
        axes = axes.flatten()
        
        plot_idx = 0

        # Plot monitor data
        for column in monitor_df.columns:
            ax = axes[plot_idx]
            monitor_df[column].plot(ax=ax, legend=False)
            ax.set_title(f"Monitor: {column.replace('_', ' ').title()}")
            ax.set_xlabel("Episodes")
            ax.set_ylabel("Value")
            ax.grid(True)
            plot_idx += 1

        # Plot progress data
        if progress_df is not None:
            time_col = 'time/total_timesteps'
            if time_col not in progress_df.columns:
                print(f"'{time_col}' not in progress.csv. Using default index.")
                x_axis = progress_df.index
                x_label = "Updates"
            else:
                x_axis = progress_df[time_col]
                x_label = "Timesteps"

            for column in progress_df.columns:
                if column != time_col and pd.api.types.is_numeric_dtype(progress_df[column]):
                    ax = axes[plot_idx]
                    ax.plot(x_axis, progress_df[column])
                    ax.set_title(f"Metric: {column.replace('/', ' / ').title()}")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel("Value")
                    ax.grid(True)
                    plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        # Set resolution
        dpi = 300 if resolution == '4k' else 150

        # Save the figure
        plt.savefig(output_file, dpi=dpi)
        print(f"Plot saved to {output_file}")
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot all training metrics from a log directory.')
    parser.add_argument('-i', '--log-dir', type=str, required=True, help='Path to the log directory containing monitor.csv and progress.csv.')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to save the output plot image.')
    parser.add_argument('-r', '--resolution', type=str, default='4k', choices=['hd', '4k'], help='Resolution of the output image.')
    
    args = parser.parse_args()
    
    plot_all_metrics(args.log_dir, args.output_file, args.resolution)
