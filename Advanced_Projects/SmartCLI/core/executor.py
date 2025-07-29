import subprocess
import shlex
import os

class CommandExecutor:
    """Executes commands with safety checks and user approval."""
    def __init__(self):
        # List of dangerous patterns to block
        self.blocklist = [
            'rm -rf', ':(){:|:&};:', 'mkfs', 'dd if=', 'shutdown', 'reboot', 'init 0', 'halt',
            'poweroff', '>:(', '>/dev/sda', '>/dev/nvme', '>/dev/hda', '>/dev/null', 'forkbomb',
            'wget http', 'curl http', 'nc -l', 'netcat', 'nmap', 'chmod 777 /', 'chown root',
        ]
        self.safe_dirs = [os.path.expanduser('~'), '/tmp', '/var/tmp']

    def is_safe(self, command: str) -> bool:
        """Check if the command is safe to execute."""
        # Block dangerous patterns
        for pattern in self.blocklist:
            if pattern in command:
                return False
        # Restrict delete operations to safe directories
        if '-delete' in command or 'rm ' in command:
            for safe_dir in self.safe_dirs:
                if safe_dir in command:
                    return True
            return False
        # Block any command that tries to operate on / or system dirs
        if any(x in command for x in ['/etc', '/bin', '/usr', '/lib', '/boot', '/root', '/dev']):
            return False
        return True

    def execute(self, command: str, preview: bool = True) -> int:
        """Preview and execute the command if approved and safe."""
        if not self.is_safe(command):
            print("[!] Command blocked for safety: ", command)
            return -1
        if preview:
            print(f"Parsed Command:\n> {command}")
            confirm = input("\u2705 Proceed? [Y/n] ").strip().lower()
            if confirm not in ('y', 'yes', ''):
                print("[i] Command execution cancelled.")
                return 0
        try:
            result = subprocess.run(command, shell=True, check=True)
            return result.returncode
        except Exception as e:
            print(f"[!] Error executing command: {e}")
            return -1 