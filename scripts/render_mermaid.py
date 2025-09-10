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
    success_log_path = os.path.join(log_dir, f"mermaid_success_{timestamp}.log")
    failure_log_path = os.path.join(log_dir, f"mermaid_failure_{timestamp}.log")
    
    return success_log_path, failure_log_path

def log_message(log_file, message):
    """
    Logs a message to the specified log file.
    """
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()}: {message}\n")

def convert_dot_to_mermaid(dot_content):
    """
    Converts graphviz .dot file content to mermaid graph syntax.
    """
    mermaid_lines = ["graph TD;"]
    lines = dot_content.splitlines()

    for line in lines:
        line = line.strip()
        if "->" in line:
            # It's an edge
            parts = line.split("->")
            if len(parts) == 2:
                source = parts[0].strip().replace('"', '')
                target = parts[1].strip().replace(';', '').replace('"', '')
                mermaid_lines.append(f"    {source} --> {target};")
        elif line and not line.startswith("digraph") and not line.startswith("}") and not "label=" in line and not "node" in line:
            # It's a node declaration
            node_name = line.replace('"', '').replace(';', '')
            mermaid_lines.append(f"    {node_name};")
            
    return "\n".join(mermaid_lines)

def render_mermaid_diagram(output_dir):
    """
    Generates architecture graphs for the project in Mermaid format.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success_log, failure_log = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    modules = {
        "train": "train.py",
        # "enjoy": "enjoy.py",
        # "rl_zoo3": "rl_zoo3"
    }

    for name, path in modules.items():
        dot_filename = os.path.join(output_dir, f"{name}_architecture_{timestamp}.dot")
        mermaid_filename = os.path.join(output_dir, f"{name}_architecture_{timestamp}.mmd")
        
        pydeps_command = [
            "pydeps",
            path,
            "-T", "dot",
            "-o", dot_filename,
            "--cluster"
        ]

        try:
            print(f"Generating .dot file for {name}...")
            subprocess.run(pydeps_command, check=True, shell=False, timeout=300)
            
            if os.path.exists(dot_filename):
                print(f"Converting {dot_filename} to Mermaid format...")
                with open(dot_filename, 'r') as f:
                    dot_content = f.read()
                
                mermaid_content = convert_dot_to_mermaid(dot_content)
                
                with open(mermaid_filename, 'w') as f:
                    f.write(mermaid_content)

                # Validation step
                if os.path.exists(mermaid_filename) and os.path.getsize(mermaid_filename) > 0:
                    log_message(success_log, f"Successfully generated Mermaid diagram for {name} at {mermaid_filename}")
                    print(f"Successfully generated Mermaid diagram for {name} at {mermaid_filename}")
                else:
                    log_message(failure_log, f"Failed to generate Mermaid diagram for {name} at {mermaid_filename}.")
                    print(f"Failed to generate Mermaid diagram for {name} at {mermaid_filename}.")
            else:
                log_message(failure_log, f"Failed to generate .dot file for {name}.")
                print(f"Failed to generate .dot file for {name}.")
                continue

        except subprocess.TimeoutExpired:
            log_message(failure_log, f"Timeout expired while generating graph for {name}.")
            print(f"Timeout expired while generating graph for {name}.")
        except subprocess.CalledProcessError as e:
            log_message(failure_log, f"Failed to generate graph for {name}: {e}")
            print(f"Failed to generate graph for {name}: {e}")
        except FileNotFoundError as e:
            log_message(failure_log, f"Error: Command not found. Make sure pydeps is installed and in your PATH. Details: {e}")
            print(f"Error: Command not found. Make sure pydeps is installed and in your PATH. Details: {e}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate architecture graphs for the project in Mermaid format.")
    parser.add_argument("--output-dir", type=str, default="architecture", help="The directory to save the architecture graphs.")
    args = parser.parse_args()
    
    render_mermaid_diagram(args.output_dir)
