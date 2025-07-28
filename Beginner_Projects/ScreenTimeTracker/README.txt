# Project Title
NeoTracker: Cyber-Futuristic Screen Time Tracker

## AIM
To track and visualize daily screen time using a glowing heatmap interface for productivity insights.

## DATASET LINK
Automatically generated via user input and saved at: logs/log.csv

## MY NOTEBOOK LINK
Standalone Python script — no cloud-based notebook used

## DESCRIPTION
NeoTracker addresses a rising need: understanding how much time we actually spend staring at our screens. In an age of digital distractions and remote work, it's easy to lose track of time — and motivation. This project was built to help reflect and stay accountable.

- Requirement: A local, lightweight, visually compelling tracker to monitor and analyze daily screen time.
- Necessity: Helps users build healthy screen habits, monitor work-life balance, and achieve focus goals.
- Usefulness: The streak-based calendar provides motivation similar to GitHub commits and LeetCode practice streaks.
- Approach: Started by designing the core timer logic with CSV logging, followed by GUI prototyping in Tkinter and then heatmap generation.
- Additional Resources:
  - Articles on digital wellness & dopamine loops
  - GitHub streak calendar CSS inspiration
  - Font selection from Google Fonts — Orbitron & Share Tech Mono

## EXPLANATION

### DETAILS OF THE DIFFERENT FEATURES
- Start/Stop Timer: A simple button-activated session tracker recording time in seconds.
- Session Logging: Every entry is timestamped and stored in log.csv.
- Live Heatmap Preview: Shows calendar-style streak visualization inside the GUI using seaborn.
- Stats Viewer: Displays cumulative screen time and daily averages.
- Stylized UI: Dark mode layout with neon fonts and glowing button effects inspired by cyberpunk design themes.

### WHAT I HAVE DONE
1. Defined the time tracker function and CSV write logic.
2. Created GUI in Tkinter and added custom fonts and themes.
3. Developed calendar-based heatmap visual with seaborn and matplotlib.
4. Connected logging to visualization pipeline.
5. Embedded live heatmap image into GUI.
6. Tested logging accuracy and calendar streak rendering.

### PROJECT TRADE-OFFS AND SOLUTIONS
1. Trade-off 1: Real-time heatmap rendering slows UI refresh rate.
   - Solution: Shifted image regeneration to batch mode rather than every second.
2. Trade-off 2: High customization vs. simplicity of UI code.
   - Solution: Modularized GUI components to keep logic readable and flexible.

### LIBRARIES NEEDED
- pandas
- matplotlib
- seaborn
- pillow
- tkinter

### SCREENSHOTS
Include the following:
- UI layout with heatmap and buttons
- EDA showing day-wise screen time intensity
- Tree diagram of project flow from logging → stats → visualization

### MODELS USED AND THEIR ACCURACIES
(Not ML-based, but if stats computation models applied)
| Metric             | Value   |
|--------------------|---------|
| Daily Avg Time     | 3.5 hrs |
| Max Time (Single Day) | 7 hrs |
| Streak Days        | 14 days |

### MODELS COMPARISON GRAPHS
Include:
- Line chart of daily time tracked
- Heatmap screenshots
- Streak comparison visuals (e.g., week vs. month)

## CONCLUSION

### WHAT YOU HAVE LEARNED
- GUI development using Tkinter and asset styling
- Dynamic image generation and embedding
- Balancing aesthetic design with functional UX
- CSV handling and time-based data grouping

### USE CASES OF THIS MODEL
1. Digital Detox Programs — Monitor habits and set screen limits.
2. Productivity Gamification — Encourage users to keep streaks for focus time.

### HOW TO INTEGRATE THIS MODEL IN REAL WORLD
1. Hook the session logger to system-wide time tracking tools.
2. Serve visualization via a web app using Flask or Streamlit.
3. Sync CSV logs to cloud (e.g., Google Drive) and deploy notification services.

### FEATURES PLANNED BUT NOT IMPLEMENTED
- Voice-activated session start/end
- Weekly goal setting & reminders
- Mobile app conversion with touch gestures

###  NAME
Ariyan
www.linkedin.com/in/ariyan-pal-590b0b321
