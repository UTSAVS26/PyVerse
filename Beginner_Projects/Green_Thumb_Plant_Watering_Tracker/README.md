# 🌿 Green Thumb – Plant Watering Tracker (Basic Version)

**Green Thumb** is a beginner-friendly Python CLI application that helps you track and manage your indoor plant watering schedule.

---

## 🚀 Features

- ✅ Add new plants with custom watering intervals
- 📅 Track last watered date and next watering due
- 💧 Mark plants as watered to update their schedule
- 🧠 Simple JSON-based data storage

---

## 📂 Project Structure

Green_Thumb/\
├── main.py [Main CLI application]\
├── plants.json [Plant data storage (initially empty)]\
└── README.md [Project documentation]

---

## ⚙️ How It Works

Each plant entry includes:
- **Name**
- **Watering frequency** (in days)
- **Last watered date** (auto-set to today on addition)

When you list your plants, the program calculates the **next watering date** and shows if a plant is overdue.

---

## 🧑‍💻 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AbhishekSharma-9/PyVerse.git
cd PyVerse/Beginner_Projects/Green_Thumb
```

### 2. Run the App

```bash
python main.py
```
---

## 🖥️ CLI Menu

===== Green Thumb =====
1. Add Plant
2. List Plants
3. Mark Plant as Watered
4. Exit

---

## 📊 Sample Usage

🌱 Enter plant name: Snake Plant
💧 Water every how many days? 5
✅ 'Snake Plant' added to your garden!

🪴 Your Plants:
- Snake Plant | Water every 5 days | Last: 2025-06-16 | Next: 2025-06-21 | ✅ On track

---

## 🧠 Concepts Covered

+ File I/O with JSON

+ datetime calculations

+ User input via CLI

+ Basic data structures (lists, dictionaries)

---

## 📌 Dependencies

This project uses only built-in Python libraries — no extra installation required:

+ json

+ datetime

+ os

---

**🤝 Contributing**

This project is part of the PyVerse open-source repository.

Feel free to improve the project by:

+ Adding features (e.g., CSV export, overdue reminders)

+ Improving UX or visuals (colored CLI output)

+ Migrating to SQLite (for intermediate versions)

---

Happy Plant Parenting! 🌿💧
