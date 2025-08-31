import random
import argparse

tips = [
    "🐍 Use list comprehensions instead of loops.",
    "💡 Always use virtual environments.",
    "🔥 Practice daily to master Python.",
    "🚀 Use meaningful variable names.",
    "📌 Comment your code wisely.",
    "🔧 Debug step by step, not by guessing.",
    "🧪 Write test cases for your code.",
    "🧠 Learn algorithms and data structures.",
    "📚 Read Python Enhancement Proposals (PEPs).",
    "✅ Always format your code using Black or Prettier."
]

def show_tip():
    print("\n✨ PYVERSE TIP ✨")
    print(random.choice(tips))
    print("⚡ Keep Learning, Keep Coding!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyVerse CLI Tool")
    subparsers = parser.add_subparsers(dest="command")

    tip_parser = subparsers.add_parser("tip", help="Show a random Python tip or quote")

    args = parser.parse_args()

    if args.command == "tip":
        show_tip()
    else:
        parser.print_help()
