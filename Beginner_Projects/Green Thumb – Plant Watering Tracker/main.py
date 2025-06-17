import json
import argparse
from datetime import datetime, timedelta, date
from colorama import init, Fore, Style
import os

# Initialize colorama
init(autoreset=True)

DATA_FILE = "plants.json"

def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(plants):
    with open(DATA_FILE, "w") as f:
        json.dump(plants, f, indent=4)

def add_plant(name, frequency, last_watered):
    plants = load_data()
    plant = {
        "name": name,
        "frequency": frequency,
        "last_watered": last_watered
    }
    plants.append(plant)
    save_data(plants)
    print(Fore.GREEN + f"Plant '{name}' added successfully.")

def list_plants():
    plants = load_data()
    if not plants:
        print(Fore.YELLOW + "No plants found.")
        return

    today = date.today()
    for plant in plants:
        last_watered_date = datetime.strptime(plant['last_watered'], "%Y-%m-%d").date()
        next_watering_date = last_watered_date + timedelta(days=plant['frequency'])

        if next_watering_date < today:
            print(Fore.RED + f"{plant['name']} is overdue for watering! (Next: {next_watering_date})")
        else:
            print(Fore.GREEN + f"{plant['name']} is on track. (Next: {next_watering_date})")

def water_plant(name):
    plants = load_data()
    for plant in plants:
        if plant['name'].lower() == name.lower():
            plant['last_watered'] = str(date.today())
            save_data(plants)
            print(Fore.CYAN + f"{name} has been watered today.")
            return
    print(Fore.RED + f"No plant named '{name}' found.")

def main():
    parser = argparse.ArgumentParser(description="Green Thumb – Plant Watering Tracker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new plant")
    add_parser.add_argument("name", help="Name of the plant")
    add_parser.add_argument("frequency", type=int, help="Watering frequency in days")
    add_parser.add_argument("--last_watered", default=str(date.today()), help="Last watered date (YYYY-MM-DD)")

    # List command
    subparsers.add_parser("list", help="List all plants with next watering dates")

    # Water command
    water_parser = subparsers.add_parser("water", help="Mark a plant as watered")
    water_parser.add_argument("name", help="Name of the plant to mark as watered")

    args = parser.parse_args()

    if args.command == "add":
        add_plant(args.name, args.frequency, args.last_watered)
    elif args.command == "list":
        list_plants()
    elif args.command == "water":
        water_plant(args.name)

if __name__ == "__main__":
    main()
