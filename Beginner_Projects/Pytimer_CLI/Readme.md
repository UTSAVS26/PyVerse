⏳ PyTimer – Command Line Countdown Timer
A beginner-friendly Python script that runs a countdown timer right in your terminal.
Great for productivity workflows like the Pomodoro technique, short study breaks, or any time-based task.

📌 Features
✅ Simple command-line interface (CLI)
⏱ Countdown display in MM:SS format
🔔 Alerts when time is up
🔁 Default timer: 60 seconds
🍅 Pomodoro mode: 25 minutes
☕ Short break: 5 minutes
🛌 Long break: 15 minutes
⛔ Handles Ctrl+C gracefully (KeyboardInterrupt)

🚀 How to Run
Step 1: Open terminal in this folder

cd Beginner_Projects/Pytimer_CLI/

Step 2: Run with one of the following commands:
▶️ Custom countdown (in seconds)

python pytimer.py --seconds 30
Runs a 30-second countdown.

⏱ Default countdown (60 seconds)

python pytimer.py
Runs a 60-second timer if no arguments are provided.

🍅 Start a Pomodoro session (25 minutes)

python pytimer.py --pomodoro
Useful for focused work or study sessions.

☕ Take a short break (5 minutes)

python pytimer.py --short-break
Take a quick reset between pomodoros.

🛌 Take a long break (15 minutes)

python pytimer.py --long-break
A longer rest after several pomodoro sessions.

⛔ Stop the timer early
Press Ctrl + C in the terminal to cancel mid-session.

Example:
⏳ Time Left: 00:10
...
🚨 Time's up!
If interrupted:
⛔ Timer interrupted by user.

🧰 Built With
Python 3
argparse
time module

🙋‍♀ Author
Saniya Kousar
Contributor – GSSoC’25
GitHub: @saniyashk1542

📄 License
This script is part of the PyVerse open-source project.
Usage and contribution follow the same license as the main repository.
