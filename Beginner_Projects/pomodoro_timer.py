import time
from datetime import datetime

# Constants
FOCUS_DURATION = 25 * 60     # 25 minutes
SHORT_BREAK = 5 * 60         # 5 minutes
LONG_BREAK = 15 * 60         # 15 minutes
SESSIONS_BEFORE_LONG_BREAK = 4
LOG_FILE = "pomodoro_log.txt"

def log_session(message):
    with open(LOG_FILE, "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] {message}\n")

def countdown(seconds, label):
    print(f"\n⏱️ {label} started for {seconds // 60} minutes.")
    while seconds:
        mins, secs = divmod(seconds, 60)
        time_str = f"{mins:02d}:{secs:02d}"
        print(f"\r{label}: {time_str}", end="")
        time.sleep(1)
        seconds -= 1
    print(f"\n✅ {label} completed!")
    log_session(f"{label} completed.")

def pomodoro_timer():
    pomodoros_completed = 0

    while True:
        countdown(FOCUS_DURATION, "Focus Session")
        pomodoros_completed += 1
        print(f"🎯 Pomodoros completed: {pomodoros_completed}")
        log_session(f"Pomodoros completed so far: {pomodoros_completed}")

        if pomodoros_completed % SESSIONS_BEFORE_LONG_BREAK == 0:
            countdown(LONG_BREAK, "Long Break")
        else:
            countdown(SHORT_BREAK, "Short Break")

        # Ask user if they want to continue
        cont = input("Start next session? (y/n): ").strip().lower()
        if cont != 'y':
            log_session("Pomodoro session ended by user.\n")
            print("👋 Goodbye! Stay productive!")
            break

if __name__ == "__main__":
    print("🍅 Welcome to the CLI Pomodoro Timer")
    pomodoro_timer()
