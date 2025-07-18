# Stock Market Analysis of Top Tech Companies (2014-2018)

![Stock Analysis Visualization]

A Python-based analysis of historical stock prices for Amazon, Apple, Microsoft, and Google from 2014 to 2018, providing insights into market trends and company performance.

## 📌 Project Overview

This project analyzes real-world stock price datasets of leading tech companies, offering:
- Data cleaning and preparation
- Exploratory analysis with visualizations
- Trend identification and comparative analysis
- Insights for traders and investors

## ✨ Key Features

- **Automated data downloading** directly from web sources
- **Comprehensive visualizations** including line charts, bar graphs, and trend distributions
- **Comparative analysis** across multiple companies
- **Statistical insights** including correlation matrices and trend percentages
- **Modular codebase** for easy extension and maintenance

## 📊 Analysis Highlights

- Amazon's dominant growth trajectory
- Apple's consistent performance
- Microsoft's successful cloud transformation
- Google's steady market position
- Comparative performance metrics
- Intraday profit/loss patterns

## 🛠️ Technologies Used

- Python 3.x
- Pandas (Data manipulation)
- Matplotlib/Seaborn (Visualization)
- Requests (Data downloading)
- NumPy (Numerical calculations)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-analysis.git
cd stock-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Edit the `DATA_URLS` dictionary in `data_loader.py` to include your dataset URLs:

```python
DATA_URLS = {
    'apple': '/kaggle/input/stock-prices-for/AAPL_data.csv',     # DECLARING DIRECTORY 
    'amazon': '/kaggle/input/stock-prices-for/AMZN_data.csv',    # DECLARING DIRECTORY
    'google': '/kaggle/input/stock-prices-for/GOOG_data.csv',    # DECLARING DIRECTORY
    'microsoft': '/kaggle/input/stock-prices-for/MSFT_data.csv'  # DECLARING DIRECTORY
}
```

### Usage

Run the main analysis script:

```bash
python main.py
```

This will:
- Download the latest stock data
- Generate visualizations
- Display statistical analysis
- Output correlation matrices

## 📂 Project Structure

```
stock-analysis/
├── data_loader.py      # Data downloading and processing
├── visualizer.py       # Visualization functions
├── analyzer.py         # Analysis functions
├── main.py             # Main script
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

## 📈 Key Findings

- Amazon showed the highest growth, reflecting its e-commerce and AWS dominance
- Apple maintained steady growth through flagship products
- Microsoft transformed successfully under Nadella's cloud-focused strategy
- Google (Alphabet) capitalized on search and digital advertising
- Strong correlations between tech stock movements

## 📝 Further Exploration Opportunities

- Add technical indicators (SMA, RSI, Bollinger Bands)
- Implement predictive modeling
- Add sentiment analysis from news sources
- Extend to real-time data analysis
- Portfolio optimization simulations

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any:
- Bug fixes
- Additional analyses
- Visualization improvements
- Documentation enhancements

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.