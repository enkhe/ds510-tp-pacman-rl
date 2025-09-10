import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_advanced(log_file, output_file, resolution):
    """
    Plot all columns from a training log file and save to a high-resolution image.
    """
    try:
        # Load the monitor.csv file
        df = pd.read_csv(log_file)

        # The first row is headers, but pandas might read it as data.
        # If the first row contains the column names, we can skip it.
        if df.columns[0] == '#':
            df = pd.read_csv(log_file, skiprows=1)

        # Set the 't' or 'time/total_timesteps' column as the index
        if 'time/total_timesteps' in df.columns:
            df.set_index('time/total_timesteps', inplace=True)
        elif 't' in df.columns:
            df.set_index('t', inplace=True)
        elif 'total_timesteps' in df.columns:
            df.set_index('total_timesteps', inplace=True)
        else:
            print("Could not find a suitable time/timestep column to use as index.")
            # Fallback to using the default index if no time column is found
            pass
            
        # Drop non-numeric columns if any, except for the index
        df = df.select_dtypes(include=['number'])

        if df.empty:
            print("No numeric data to plot.")
            return

        # Determine the number of subplots needed
        num_plots = len(df.columns)
        if num_plots == 0:
            print("No data columns to plot.")
            return
            
        # Create subplots
        # Let's aim for a grid that's roughly square
        cols = int(num_plots**0.5)
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True)
        fig.suptitle('Training Metrics', fontsize=20)

        # Flatten axes array for easy iteration
        axes = axes.flatten()

        for i, column in enumerate(df.columns):
            ax = axes[i]
            df[column].plot(ax=ax)
            ax.set_title(column.replace('_', ' ').title())
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Value")
            ax.grid(True)

        # Hide any unused subplots
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)

        # Set resolution
        dpi = 300 # Default high-res
        if resolution == '4k':
            dpi = 300 # A typical 4K monitor is 3840x2160. We'll use high DPI.
        elif resolution == 'hd':
            dpi = 150

        # Save the figure
        plt.savefig(output_file, dpi=dpi)
        print(f"Plot saved to {output_file}")
        plt.close()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training metrics from a log file.')
    parser.add_argument('-i', '--log-file', type=str, required=True, help='Path to the monitor.csv log file.')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to save the output plot image.')
    parser.add_argument('-r', '--resolution', type=str, default='4k', choices=['hd', '4k'], help='Resolution of the output image.')
    
    args = parser.parse_args()
    
    plot_advanced(args.log_file, args.output_file, args.resolution)
