import argparse
import os
import subprocess
from datetime import datetime

def setup_logging(log_dir="logs"):
    """
    Sets up the logging directory.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success_log_path = os.path.join(log_dir, f"success_{timestamp}.log")
    failure_log_path = os.path.join(log_dir, f"failure_{timestamp}.log")
    
    return success_log_path, failure_log_path

def log_message(log_file, message):
    """
    Logs a message to the specified log file.
    """
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()}: {message}\n")

def render_architecture(output_dir, dpi=300):
    """
    Generates architecture graphs for the project using pydeps.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success_log, failure_log = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Let's focus on the smallest module first
    modules = {
        "train": "train.py",
    }

    for name, path in modules.items():
        dot_filename = os.path.join(output_dir, f"{name}_architecture_{timestamp}.dot")
        png_filename = os.path.join(output_dir, f"{name}_architecture_{timestamp}.png")
        
        pydeps_command = [
            "pydeps",
            path,
            "-T", "dot",
            "-o", dot_filename,
            "--cluster",
            "--log", "debug",
        ]
        
        dot_command = [
            "dot",
            "-Tpng",
            f"-Gdpi={dpi}",
            dot_filename,
            "-o",
            png_filename
        ]

        try:
            print(f"Generating .dot file for {name}...")
            pydeps_process = subprocess.run(pydeps_command, check=True, timeout=600, capture_output=True, text=True)
            print(pydeps_process.stdout)
            print(pydeps_process.stderr)

            if os.path.exists(dot_filename):
                print(f"Generating .png file for {name} from {dot_filename}...")
                subprocess.run(dot_command, check=True)
            else:
                log_message(failure_log, f"Failed to generate .dot file for {name}.")
                print(f"Failed to generate .dot file for {name}.")
                continue

            # Validation step
            if os.path.exists(png_filename) and os.path.getsize(png_filename) > 0:
                log_message(success_log, f"Successfully generated graph for {name} at {png_filename}")
                print(f"Successfully generated graph for {name} at {png_filename}")
            else:
                log_message(failure_log, f"Failed to generate graph for {name} at {png_filename}. File not created or is empty.")
                print(f"Failed to generate graph for {name} at {png_filename}. File not created or is empty.")

        except subprocess.TimeoutExpired as e:
            log_message(failure_log, f"Timeout expired while generating graph for {name}.")
            print(f"Timeout expired while generating graph for {name}.")
            if e.stdout:
                print(f"Pydeps stdout: {e.stdout}")
            if e.stderr:
                print(f"Pydeps stderr: {e.stderr}")
        except subprocess.CalledProcessError as e:
            log_message(failure_log, f"Failed to generate graph for {name}: {e}")
            print(f"Failed to generate graph for {name}: {e}")
            print(f"Pydeps stdout: {e.stdout}")
            print(f"Pydeps stderr: {e.stderr}")
        except FileNotFoundError as e:
            log_message(failure_log, f"Error: Command not found. Make sure pydeps and graphviz are installed and in your PATH. Details: {e}")
            print(f"Error: Command not found. Make sure pydeps and graphviz are installed and in your PATH. Details: {e}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate architecture graphs for the project.")
    parser.add_argument("--output-dir", type=str, default="architecture", help="The directory to save the architecture graphs.")
    # DPI is not directly supported by pydeps in the same way, but we can keep it for future use or other tools.
    parser.add_argument("--dpi", type=int, default=300, help="The resolution of the output images in dots per inch (for tools that support it).")
    args = parser.parse_args()
    
    render_architecture(args.output_dir, args.dpi)
