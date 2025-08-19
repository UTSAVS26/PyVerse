# AI Habit Tracker - Usage Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- All dependencies installed (run `pip install -r requirements.txt`)

### Running the Applications

#### 1. GUI Application (Recommended for beginners)
```bash
python run_gui.py
```
- **Features**: User-friendly interface with buttons and forms
- **Best for**: Daily habit logging and quick insights
- **Screenshots**: Available in the `screenshots/` directory

#### 2. Streamlit Web Dashboard (Recommended for analytics)
```bash
streamlit run src/streamlit_app.py
```
- **Features**: Interactive web dashboard with charts and analytics
- **Best for**: Deep analysis and data visualization
- **Access**: Opens in your default web browser

#### 3. Command Line Interface (Recommended for power users)
```bash
python run_cli.py [command] [options]
```
- **Features**: Fast command-line operations
- **Best for**: Quick data entry and batch operations

## 📊 Data Structure

Each habit entry includes:
- **Date**: The day of the entry
- **Sleep Hours**: Hours of sleep (0-24)
- **Exercise Minutes**: Minutes of exercise (0-1440)
- **Screen Time Hours**: Hours spent on screens (0-24)
- **Water Glasses**: Number of glasses of water (0-20)
- **Work/Study Hours**: Hours spent working/studying (0-24)
- **Mood Rating**: 1-5 scale (1=terrible, 5=excellent)
- **Productivity Rating**: 1-5 scale (1=very low, 5=very high)

## 🎯 CLI Commands

### Add a new habit entry
```bash
python run_cli.py add --date 2025-01-15 --sleep 8 --exercise 30 --screen 4 --water 8 --work 6 --mood 4 --productivity 4
```

### View entries
```bash
# View all entries
python run_cli.py view

# View entries for a specific date range
python run_cli.py view --date 2025-01-01 --output 2025-01-31
```

### Get AI insights
```bash
python run_cli.py insights
```

### View statistics
```bash
python run_cli.py stats
```

### Export data
```bash
# Export to CSV
python run_cli.py export --output habits_data.csv

# Export to JSON
python run_cli.py export --output habits_data.json
```

### Clear all data
```bash
python run_cli.py clear
```

### Interactive mode
```bash
python run_cli.py interactive
```

## 🔍 AI Features

### Pattern Detection
The AI analyzes your data to find:
- **Sleep Patterns**: Optimal sleep duration for your productivity
- **Exercise Impact**: How exercise affects your mood and productivity
- **Screen Time Effects**: Correlation between screen time and well-being
- **Weekly Patterns**: Day-of-week trends in your habits
- **Streaks**: Consecutive days of good habits

### Predictions
- **Productivity Prediction**: Forecast your productivity for upcoming days
- **Low-Performance Alerts**: Identify days when you might struggle
- **Optimal Habit Recommendations**: Personalized suggestions

### Insights
- **Correlation Analysis**: Hidden relationships between habits
- **Trend Analysis**: Long-term changes in your behavior
- **Actionable Recommendations**: Specific steps to improve

## 📈 Sample Data

To get started quickly, you can add sample data:

```bash
# Add a week of sample data
python run_cli.py add --date 2025-01-01 --sleep 7.5 --exercise 45 --screen 5 --water 8 --work 7 --mood 4 --productivity 4
python run_cli.py add --date 2025-01-02 --sleep 8.0 --exercise 30 --screen 6 --water 6 --work 8 --mood 3 --productivity 3
python run_cli.py add --date 2025-01-03 --sleep 6.5 --exercise 60 --screen 3 --water 10 --work 6 --mood 5 --productivity 5
python run_cli.py add --date 2025-01-04 --sleep 8.5 --exercise 0 --screen 8 --water 4 --work 4 --mood 2 --productivity 2
python run_cli.py add --date 2025-01-05 --sleep 7.0 --exercise 90 --screen 2 --water 12 --work 5 --mood 5 --productivity 5
```

## 🎨 Visualization Features

### Available Charts
1. **Dashboard**: Comprehensive overview with multiple charts
2. **Correlation Heatmap**: Shows relationships between all variables
3. **Trend Analysis**: Long-term patterns over time
4. **Weekly Summary**: Day-of-week patterns
5. **Sleep Analysis**: Sleep quality vs. productivity
6. **Exercise Impact**: Exercise correlation with mood/productivity

### Saving Charts
Charts can be saved as HTML files for sharing or embedding:
```python
# In the GUI or Streamlit app, use the "Save Charts" feature
# Charts will be saved to the 'charts/' directory
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **Database Errors**: Check that the `data/` directory exists and is writable
3. **GUI Not Opening**: Ensure you have tkinter installed (usually included with Python)
4. **Streamlit Not Working**: Install streamlit with `pip install streamlit`

### Data Backup
Your data is stored in `data/habits.db`. You can backup this file to preserve your data.

### Reset Application
To start fresh:
```bash
python run_cli.py clear
```

## 📱 Tips for Best Results

1. **Consistency**: Log your habits daily for better pattern detection
2. **Honesty**: Be truthful about your ratings for accurate insights
3. **Patience**: The AI needs at least 7-10 days of data for meaningful patterns
4. **Regular Review**: Check insights weekly to track your progress
5. **Experiment**: Try the AI's recommendations and see how they affect your data

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run the test suite: `python -m pytest tests/`
3. Check the logs in the console output
4. Ensure all dependencies are installed correctly

## 🎯 Next Steps

After getting familiar with the basic features:
1. Explore the AI insights to understand your patterns
2. Try the productivity predictions
3. Experiment with the recommendations
4. Export your data for external analysis
5. Share your progress with friends or coaches
