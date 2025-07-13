import subprocess
import platform

def execute_command(command: str) -> str:
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, executable='/bin/bash')
        output = result.stdout + result.stderr
        return output.strip()
    except Exception as e:
        return f"Error executing command: {e}" 