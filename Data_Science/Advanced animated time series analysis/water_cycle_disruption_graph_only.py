import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import interp1d
from scipy.stats import linregress

def interpolate_data(data, param, points_per_segment=20):
    years = data['Year']
    values = data[param]
    interpolator = interp1d(years, values, kind='quadratic')
    smooth_years = np.linspace(years.min(), years.max(), points_per_segment * (len(years) - 1))
    smooth_values = interpolator(smooth_years)
    return smooth_values, smooth_years

def display_current_data(data):
    print("\nCurrent Data:")
    print(data)

def display_predicted_data(data, regression_results):
    print("\nPredicted Data for Next 5 Years:")
    future_years = np.arange(data['Year'].max() + 1, data['Year'].max() + 6)
    predictions = {}
    for param in parameters:
        predictions[param] = regression_results[param]['intercept'] + regression_results[param]['slope'] * future_years
    predicted_data = pd.DataFrame(predictions, index=future_years)
    print(predicted_data)

def avg(data):
    averages = data.mean()
    print("Average Parameters:")
    print(averages)

def update1(frame):
    for param in parameters:
        if frame <= len(data) * 20 and frame % 20 == 0:
            scatters[param].set_offsets(np.c_[data['Year'][:int(frame / 20) + 1], data[param][:int(frame / 20) + 1]])
        elif (frame >= len(data) * 20) and frame < len(data) * 40:
            lines[param].set_data(interpolated_years[param][:frame + 1 - len(data) * 20], interpolated_data[param][:frame + 1 - len(data) * 20])
        elif frame >= len(data) * 40:
            reg_lines[param].set_data(future_years[:frame+1-len(data) * 40], regression_lines[param][:frame+1-len(data) * 40])

    return list(scatters.values()) + list(lines.values()) + list(reg_lines.values())

def plot_graph(data, interpolated_data, interpolated_years, regression_lines, regression_results, future_years):
    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, param in enumerate(parameters):
        color = colors[idx]
        scatters[param] = ax.scatter([], [], label=param, color=color)
        lines[param] = ax.plot([], [], '--', lw=1, alpha=0.7, color=color)[0]
        reg_lines[param] = ax.plot([], [], '-', lw=1.5, color=color)[0]

    ax.set_xlim(data['Year'].min() - 1, data['Year'].max() + 5)
    ax.set_ylim(data[parameters].min().min() - 5, data[parameters].max().max() + 5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Values')
    ax.set_title('Trends in Parameters Over the Years (with Linear Regression)')
    ax.legend()
    ax.grid()

    ani = FuncAnimation(
        fig, update1,
        frames=(len(data) * 40)+(len(data)+5)*10,
        blit=True, interval=interval, repeat=False
    )

    writer = PillowWriter(fps=20)
    ani.save("animated_plot.gif", writer=writer)
    plt.show()


def provide_insights(data, regression_results):
    for param in parameters:
        slope = regression_results[param]['slope']
        r_value = regression_results[param]['r_value']

        trend = "increasing" if slope > 0 else "decreasing"
        trend_strength = "strong" if abs(slope) > 0.5 else "mild"
        correlation_strength = "strong" if abs(r_value) > 0.7 else "weak"

        print(f"\nParameter: {param}")
        print(f"- Trend: {trend} ({trend_strength} slope of {slope:.2f})")
        print(f"- Correlation with Year: {correlation_strength} (R-value: {r_value:.2f})")

        # Contextual insight
        if trend == "increasing" and param == "Urbanization Rate":
            print("  Insight: Urbanization is growing, which might impact other environmental factors such as deforestation, air quality, and water demand.")
        elif trend == "decreasing" and param == "Yearly Rainfall(in cm)":
            print("  Insight: Rainfall is decreasing, potentially indicating a shift in climate patterns or the onset of drought conditions in the region.")
        elif trend == "increasing" and param == "Temperature":
            print("  Insight: Rising temperatures could be a sign of global warming, potentially leading to heatwaves, reduced crop yields, and melting polar ice caps.")
        elif trend == "increasing" and param == "Evaporation Rate":
            print("  Insight: Higher evaporation rates may be driven by increased temperatures, potentially affecting water availability and soil moisture levels.")
        elif trend == "decreasing" and param == "Runoff Rate":
            print("  Insight: A decline in runoff rates might suggest reduced rainfall, increased water absorption by soil, or higher water retention due to urban planning.")
        elif trend == "increasing" and param == "Precipitation Rate":
            print("  Insight: Increased precipitation may lead to more frequent flooding events, affecting infrastructure and ecosystems.")
        elif trend == "decreasing" and param == "Urbanization Rate":
            print("  Insight: A slowdown in urbanization might reflect economic changes, shifts in population growth, or government policies aimed at decentralization.")
        elif trend == "increasing" and param == "Runoff Rate":
            print("  Insight: Rising runoff rates could indicate increased urbanization or reduced vegetation cover, leading to higher risks of soil erosion.")
        elif trend == "increasing" and param == "Yearly Rainfall(in cm)":
            print("  Insight: Growing rainfall levels could be indicative of a wet climate phase or changes in regional atmospheric circulation patterns.")
        elif trend == "decreasing" and param == "Evaporation Rate":
            print("  Insight: Reduced evaporation rates might point to cooler temperatures or higher humidity, potentially benefiting agriculture.")

        mean = data[param].mean()
        std_dev = data[param].std()
        outliers = data[(data[param] > mean + 2 * std_dev) | (data[param] < mean - 2 * std_dev)]
        if not outliers.empty:
            print(f"  Outliers detected in {param}:")
            print(outliers[['Year', param]])


# === Setup ===

data = pd.read_csv('data.csv')
data['Yearly Rainfall(in cm)'] = data['Yearly Rainfall'] / 10

parameters = ['Yearly Rainfall(in cm)', 'Evaporation Rate', 'Precipitation Rate', 'Runoff Rate', 'Temperature', 'Urbanization Rate']
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

speed = 500
interval = int(speed / (50 * len(data)))

# Global storage
scatters = {}
lines = {}
reg_lines = {}

# Interpolated values
interpolated_data = {}
interpolated_years = {}
for param in parameters:
    interpolated_data[param], interpolated_years[param] = interpolate_data(data, param)

# Linear regression for prediction
regression_results = {}
for param in parameters:
    slope, intercept, r_value, p_value, std_err = linregress(data['Year'], data[param])
    regression_results[param] = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
    }

future_years = np.linspace(data['Year'].min(), data['Year'].max() + 5, (len(data)+5)*5)
regression_lines = {
    param: regression_results[param]['intercept'] + regression_results[param]['slope'] * future_years
    for param in parameters
}

# === Run ===
plot_graph(data, interpolated_data, interpolated_years, regression_lines, regression_results, future_years)
