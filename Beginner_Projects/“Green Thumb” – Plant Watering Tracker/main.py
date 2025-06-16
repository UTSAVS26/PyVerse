import json
from datetime import datetime, timedelta
import os

DATA_FILE = "plants.json"

def load_plants():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_plants(plants):
    with open(DATA_FILE, "w") as f:
        json.dump(plants, f, indent=4)

def add_plant():
    name = input("🌱 Enter plant name: ")
    freq = int(input("💧 Water every how many days? "))
    today = datetime.today().strftime("%Y-%m-%d")
    plant = {
        "name": name,
        "frequency": freq,
        "last_watered": today
    }
    plants = load_plants()
    plants.append(plant)
    save_plants(plants)
    print(f"✅ '{name}' added to your garden!")

def list_plants():
    plants = load_plants()
    if not plants:
        print("No plants added yet!")
        return
    print("\n🪴 Your Plants:")
    today = datetime.today()
    for p in plants:
        last = datetime.strptime(p["last_watered"], "%Y-%m-%d")
        next_due = last + timedelta(days=p["frequency"])
        status = "✅ On track" if next_due > today else "⚠️ Needs water!"
        print(f"- {p['name']} | Water every {p['frequency']} days | Last: {p['last_watered']} | Next: {next_due.date()} | {status}")

def water_plant():
    name = input("Enter plant name to mark as watered: ")
    plants = load_plants()
    found = False
    for p in plants:
        if p["name"].lower() == name.lower():
            p["last_watered"] = datetime.today().strftime("%Y-%m-%d")
            found = True
            break
    if found:
        save_plants(plants)
        print(f"💧 {name} marked as watered today!")
    else:
        print("❌ Plant not found.")

def main():
    while True:
        print("\n===== Green Thumb =====")
        print("1. Add Plant")
        print("2. List Plants")
        print("3. Mark Plant as Watered")
        print("4. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            add_plant()
        elif choice == "2":
            list_plants()
        elif choice == "3":
            water_plant()
        elif choice == "4":
            print("Goodbye! 🌿")
            break
        else:
            print("❗ Invalid choice.")

if __name__ == "__main__":
    main()
