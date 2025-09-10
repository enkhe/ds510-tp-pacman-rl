import pandas as pd
import mplfinance as mpf
import argparse
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_candlestick_from_tb(log_dir, output_file, resolution):
    """
    Create candlestick charts for training metrics from TensorBoard event files.
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

        # Extract scalar data
        scalar_tags = ea.Tags()['scalars']
        data = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            data[tag] = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value']).set_index('step')

        if not data:
            print("No scalar data found in the event file.")
            return

        df = pd.concat(data.values(), axis=1, keys=data.keys())
        df.columns = df.columns.droplevel(1) # drop the 'value' level

        # --- Create Candlestick for Loss ---
        if 'train/loss' not in df.columns:
            print("Loss data ('train/loss') not found.")
            return
            
        # Resample to create OHLC. We'll resample by number of steps.
        # Let's create a candle for every 10 data points (steps are not always uniform)
        loss_data = df['train/loss'].dropna()
        resample_period = (loss_data.index.max() - loss_data.index.min()) // 50 # Aim for ~50 candles
        if resample_period == 0: resample_period = 1

        # Create groups and calculate OHLC
        groups = loss_data.index // resample_period
        ohlc = loss_data.groupby(groups).agg(['first', 'max', 'min', 'last'])
        ohlc.columns = ['Open', 'High', 'Low', 'Close']
        ohlc.index = pd.to_datetime(ohlc.index, unit='s')


        # --- Plotting ---
        metrics_to_plot = ['rollout/ep_rew_mean', 'train/value_loss', 'train/entropy_loss']
        
        num_plots = 1 + len([m for m in metrics_to_plot if m in df.columns])
        
        fig = mpf.figure(figsize=(20, num_plots * 8), style='charles')

        # Plot Loss Candlestick
        ax1 = fig.add_subplot(num_plots, 1, 1)
        ax1.set_title('Training Loss (Candlestick)', fontsize=16)
        mpf.plot(ohlc, type='candle', ax=ax1, axtitle='Training Loss (Candlestick)')
        
        # Plot other metrics as line plots
        plot_idx = 2
        for metric in metrics_to_plot:
            if metric in df.columns:
                ax = fig.add_subplot(num_plots, 1, plot_idx, sharex=ax1)
                # Convert step index to datetime for plotting if needed, or just plot against steps
                metric_data = df[metric].dropna()
                ax.plot(metric_data.index, metric_data.values, label=metric)
                ax.set_title(metric.replace('/', ' / ').title(), fontsize=14)
                ax.set_ylabel("Value")
                ax.set_xlabel("Steps")
                ax.legend()
                ax.grid(True)
                plot_idx += 1

        fig.suptitle('Training Metrics Overview', fontsize=24, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.96])


        # Set resolution
        dpi = 300 if resolution == '4k' else 150

        # Save the figure
        fig.savefig(output_file, dpi=dpi)
        print(f"Candlestick plot saved to {output_file}")
        plt.close(fig)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create candlestick charts from TensorBoard log files.')
    parser.add_argument('-i', '--log-dir', type=str, required=True, help='Path to the log directory containing the TensorBoard event file.')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Path to save the output plot image.')
    parser.add_argument('-r', '--resolution', type=str, default='4k', choices=['hd', '4k'], help='Resolution of the output image.')
    
    args = parser.parse_args()
    
    plot_candlestick_from_tb(args.log_dir, args.output_file, args.resolution)
