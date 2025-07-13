import subprocess
import platform

import shlex
import logging

# Define allowed commands (example)
ALLOWED_COMMANDS = {'ls', 'pwd', 'whoami', 'date', 'echo'}

def execute_command(command: str) -> str:
    # Log all command attempts
    logging.info(f"Command execution attempt: {command}")
    
    # Basic input validation
    if not command or len(command.strip()) == 0:
        return "Error: Empty command"
    
    # Parse command safely
    try:
        cmd_parts = shlex.split(command)
        if not cmd_parts:
            return "Error: Invalid command format"
        
        base_command = cmd_parts[0]
        if base_command not in ALLOWED_COMMANDS:
            return f"Error: Command '{base_command}' not allowed"
    except ValueError as e:
        return f"Error: Invalid command syntax: {e}"
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=30)
        else:
            result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=30)
        # Separate stdout and stderr for better security
        if result.returncode != 0:
            return f"Command failed (exit code {result.returncode}): {result.stderr.strip()}"
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Command execution error: {e}")
        return f"Error executing command: {e}"