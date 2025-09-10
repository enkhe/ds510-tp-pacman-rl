import pandas as pd
import mplfinance as mpf
import argparse
import os
import matplotlib.pyplot as plt

def plot_candlestick(log_dir, output_file, resolution):
    """
    Create candlestick charts for training metrics.
    """
    try:
        progress_file = os.path.join(log_dir, 'progress.csv')
        if not os.path.exists(progress_file):
            print(f"progress.csv not found in {log_dir}. Cannot create candlestick chart.")
            return

        df = pd.read_csv(progress_file)

        # We need Open, High, Low, Close. We can derive these from the 'loss' column
        # by resampling over a certain period (e.g., every 10 updates).
        if 'loss' not in df.columns:
            print("Loss column not found. Cannot create candlestick chart.")
            return
            
        # Ensure 'time/total_timesteps' is numeric and use it as the index
        if 'time/total_timesteps' in df.columns and pd.api.types.is_numeric_dtype(df['time/total_timesteps']):
            df.set_index('time/total_timesteps', inplace=True)
        else:
            # If not available, use a simple range index
            df.index = pd.to_numeric(df.index)

        # Resample the data to create OHLC for the 'loss'
        # Let's create intervals of 1000 timesteps for each candle
        # Resample requires a datetime-like index
        df.index = pd.to_timedelta(df.index, unit='s')
        ohlc = df['loss'].resample('1000S').ohlc()
        
        # Let's also do it for other interesting metrics if they exist
        metrics_to_plot = ['rollout/ep_rew_mean', 'train/value_loss', 'train/entropy_loss']
        
        num_plots = 1 + len([m for m in metrics_to_plot if m in df.columns])
        
        fig = mpf.figure(figsize=(15, num_plots * 7), style='yahoo')

        # Plot Loss Candlestick
        ax1 = fig.add_subplot(num_plots, 1, 1)
        mpf.plot(ohlc, type='candle', ax=ax1, title='Training Loss (Candlestick)')
        
        # Plot other metrics as line plots
        plot_idx = 2
        for metric in metrics_to_plot:
            if metric in df.columns:
                ax = fig.add_subplot(num_plots, 1, plot_idx, sharex=ax1)
                df[metric].plot(ax=ax, legend=True)
                ax.set_title(metric.replace('/', ' / ').title())
                ax.set_ylabel("Value")
                ax.grid(True)
                plot_idx += 1

        # Set resolution
        dpi = 300 if resolution == '4k' else 150

        # Save the figure
        fig.savefig(output_file, dpi=dpi)
        print(f"Candlestick plot saved to {output_file}")
        plt.close(fig)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create candlestick charts from training logs.')
    parser.add_argument('-i', '--log-dir', type=str, required=True, help='Path to the log directory containing progress.csv.')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to save the output plot image.')
    parser.add_argument('-r', '--resolution', type=str, default='4k', choices=['hd', '4k'], help='Resolution of the output image.')
    
    args = parser.parse_args()
    
    plot_candlestick(args.log_dir, args.output_file, args.resolution)
