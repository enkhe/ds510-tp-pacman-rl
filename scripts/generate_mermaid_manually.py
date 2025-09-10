import argparse
import os
import re
from datetime import datetime

def setup_logging(log_dir="logs"):
    """Sets up the logging directory."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success_log_path = os.path.join(log_dir, f"manual_mermaid_success_{timestamp}.log")
    failure_log_path = os.path.join(log_dir, f"manual_mermaid_failure_{timestamp}.log")
    return success_log_path, failure_log_path

def log_message(log_file, message):
    """Logs a message to the specified log file."""
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()}: {message}\n")

def find_imports(file_path):
    """Finds all imports in a given Python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Regex to find imports, e.g., 'import x', 'from x import y'
            import_statements = re.findall(r"^(?:from\s+([\w\.]+)|import\s+([\w\.]+))", content, re.MULTILINE)
            for imp in import_statements:
                # imp is a tuple, e.g., ('rl_zoo3.utils', '') or ('', 'os')
                module = imp[0] or imp[1]
                # Take the root of the module
                root_module = module.split('.')[0]
                imports.add(root_module)
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return imports

def generate_mermaid_for_file(file_path, output_dir, success_log, failure_log):
    """Generates a Mermaid diagram for a single Python file."""
    module_name = os.path.basename(file_path).replace('.py', '')
    imports = find_imports(file_path)
    
    if not imports:
        log_message(failure_log, f"No imports found for {module_name}")
        return

    mermaid_content = ["graph TD;"]
    mermaid_content.append(f"    subgraph {module_name};")
    mermaid_content.append(f"        {module_name}_main[({module_name})];")
    mermaid_content.append("    end;")

    for imp in imports:
        # Avoid self-reference
        if imp != module_name and imp not in ['__future__']:
             mermaid_content.append(f"    {module_name}_main --> {imp};")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mermaid_filename = os.path.join(output_dir, f"{module_name}_manual_architecture_{timestamp}.mmd")

    try:
        with open(mermaid_filename, 'w') as f:
            f.write("\n".join(mermaid_content))
        
        if os.path.exists(mermaid_filename) and os.path.getsize(mermaid_filename) > 0:
            log_message(success_log, f"Successfully generated Mermaid diagram for {module_name} at {mermaid_filename}")
            print(f"Successfully generated Mermaid diagram for {module_name} at {mermaid_filename}")
        else:
            log_message(failure_log, f"Failed to generate or write Mermaid diagram for {module_name}.")
            print(f"Failed to generate or write Mermaid diagram for {module_name}.")

    except Exception as e:
        log_message(failure_log, f"Error writing Mermaid file for {module_name}: {e}")
        print(f"Error writing Mermaid file for {module_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate architecture graphs in Mermaid format by manually parsing imports.")
    parser.add_argument("--output-dir", type=str, default="architecture", help="The directory to save the architecture graphs.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    success_log, failure_log = setup_logging()

    files_to_process = ["train.py", "enjoy.py"]
    for file in files_to_process:
        generate_mermaid_for_file(file, args.output_dir, success_log, failure_log)

if __name__ == "__main__":
    main()
