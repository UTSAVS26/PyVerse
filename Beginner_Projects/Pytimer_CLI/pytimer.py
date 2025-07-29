import time
import argparse

def countdown_timer(seconds):
    try:
        while seconds:
            mins, secs = divmod(seconds, 60)
            timer = f"{mins:02d}:{secs:02d}"
            print(f"\r⏳ Time Left: {timer}", end="", flush=True)
            time.sleep(1)
            seconds -= 1
        print("\n🚨 Time's up!\a")  # Terminal beep
        if seconds < 30:
            print("⏰ Quick session complete!")
        elif seconds <= 300:
            print("✅ Short break over. Back to work!")
        else:
            print("🎉 Great job finishing your session!")
    except KeyboardInterrupt:
        print("\n⛔ Timer interrupted.")

def run():
    parser = argparse.ArgumentParser(description="Start a countdown timer from CLI.")
    parser.add_argument('--seconds', type=int, default=0, help="Time in seconds")
    parser.add_argument('--minutes', type=int, default=0, help="Time in minutes")
    parser.add_argument('--pomodoro', action='store_true', help="Start a 25-minute Pomodoro session")
    parser.add_argument('--short-break', action='store_true', help="Start a 5-minute short break")
    parser.add_argument('--long-break', action='store_true', help="Start a 15-minute long break")

    args = parser.parse_args()

    # Predefined modes
    if args.pomodoro:
        total_seconds = 25 * 60
    elif args.short_break:
        total_seconds = 5 * 60
    elif args.long_break:
        total_seconds = 15 * 60
    else:
        total_seconds = args.seconds + args.minutes * 60

    if total_seconds == 0:
        print("⚠ Please provide a valid time (use --seconds, --minutes, or a preset flag).")
        return

    countdown_timer(total_seconds)

if _name_ == "_main_":
    run()
