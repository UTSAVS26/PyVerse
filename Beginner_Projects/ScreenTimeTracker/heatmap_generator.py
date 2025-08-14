import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Load session logs
def load_data(log_path="logs/log.csv"):
    if not os.path.exists(log_path):
        print("No log file found.")
        return pd.DataFrame(columns=["Date", "Duration (seconds)"])

    df = pd.read_csv(log_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Duration (seconds)"] = pd.to_numeric(df["Duration (seconds)"], errors="coerce")
    return df

# Group by day
def group_daily_totals(df):
    daily = df.groupby(df["Date"].dt.date)["Duration (seconds)"].sum()
    return pd.DataFrame({
        "Date": pd.to_datetime(daily.index),
        "TotalSeconds": daily.values
    })

# Create heatmap grid
def generate_heatmap(data):
    data["Day"] = data["Date"].dt.day
    data["Month"] = data["Date"].dt.month

    pivot = data.pivot("Month", "Day", "TotalSeconds").fillna(0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="Blues", linewidths=0.5, linecolor="#101522", cbar=True)
    plt.title("📆 Screen Time Heatmap", fontsize=16)
    plt.xlabel("Day of Month")
    plt.ylabel("Month")

    os.makedirs("assets", exist_ok=True)
    chart_path = f"assets/streak_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"✅ Heatmap saved to {chart_path}")

# Main
def run_heatmap():
    df = load_data()
    if df.empty:
        print("No data to show.")
        return
    streak_data = group_daily_totals(df)
    generate_heatmap(streak_data)

if __name__ == "__main__":
    run_heatmap()