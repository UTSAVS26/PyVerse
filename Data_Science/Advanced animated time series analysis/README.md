
# 📈 Animated Climate Parameter Trends Visualization

This project visualizes environmental parameters over time using animated plots created with Matplotlib. It uses interpolation and regression techniques to create a smooth, insightful animation showcasing trends in rainfall, temperature, urbanization, and more.

## 🚀 Features

- 📊 Animated plot generation with interpolation (quadratic) for smooth transitions
- 🔮 Linear regression-based future prediction for 5 years
- 🌐 Multi-parameter visualization with legends, grids, and labels
- 🎥 Exports final animation as a GIF

## 🛠 Parameters Tracked

- Yearly Rainfall (in cm)
- Evaporation Rate
- Precipitation Rate
- Runoff Rate
- Temperature
- Urbanization Rate

## 📂 File Structure

```
.
├── data.csv                  # Input data file
├── animated_plot.gif         # Output animated visualization
├── climate_visualizer.py     # Main Python script
└── README.md                 # Project overview
```

## 🧪 How It Works

1. The script reads historical environmental data from a CSV file.
2. It performs quadratic interpolation on each parameter for smooth visualization.
3. A linear regression is used to project future values for the next 5 years.
4. Matplotlib's `FuncAnimation` is used to generate a dynamic animated graph.
5. The animation is exported as a `.gif` using `PillowWriter`.

## 📦 Requirements

Install the required dependencies using:

```bash
pip install matplotlib pandas numpy scipy
```

## 🖼 Output Example

The final animated output (`animated_plot.gif`) will show how each parameter evolved over the years, including predicted future values.

## 📌 Notes

- Ensure `data.csv` includes the required parameters and is correctly formatted.
- Customize interpolation type or plotting speed in the script settings.

## 👤 Author

GitHub: [SK8-infi](https://github.com/SK8-infi)
