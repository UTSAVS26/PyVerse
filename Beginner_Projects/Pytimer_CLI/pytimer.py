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
        print("\n🚨 Time's up!")
    except KeyboardInterrupt:
        print("\n⛔ Timer interrupted.")

def run():
    parser = argparse.ArgumentParser(description="Start a countdown timer from CLI.")
    parser.add_argument('--seconds', type=int, help="Time in seconds", default=60)
    args = parser.parse_args()
    countdown_timer(args.seconds)
