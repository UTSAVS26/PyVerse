import sys

def run_gui():
    print("Running GUI...")  # Replace with actual GUI launch code

def run_tui():
    from game import Game
    color = input('Select color : black or white  ')
    game = Game(color=color)
    game.play()

if __name__ == "__main__":
    if "-t" in sys.argv[1:]:
        run_tui()
    else:
        run_gui()