import subprocess

def run_git_command(command):
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Add all changes except scratch directory
    if run_git_command("git add ."):
        # Commit changes
        commit_message = "Refactor Meeting Engagement Analyzer: modularize project, enhance ML models, and improve GUI"
        if run_git_command(f'git commit -m "{commit_message}"'):
            # Push to origin main
            run_git_command("git push origin main")
        else:
            print("Commit failed. Maybe no changes to commit?")
    else:
        print("Add failed.")
