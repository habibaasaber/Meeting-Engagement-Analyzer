import subprocess

def run_git_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_git_command("git status")
