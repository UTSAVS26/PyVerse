import argparse
import sys

# Import dashboard (PyQt5 or Streamlit)
from gui.dashboard import run_dashboard

def main():
    parser = argparse.ArgumentParser(description="Dynamic Hardware Resource Monitor with Prediction")
    parser.add_argument('--mode', choices=['qt', 'web'], default='qt', help='Choose GUI mode: qt (PyQt5) or web (Streamlit)')
    args = parser.parse_args()

    if args.mode == 'qt':
        run_dashboard(mode='qt')
    elif args.mode == 'web':
        run_dashboard(mode='web')
    else:
        print("Invalid mode. Use --mode=qt or --mode=web.")
        sys.exit(1)

if __name__ == "__main__":
    main()
