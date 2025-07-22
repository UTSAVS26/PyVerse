import matplotlib.pyplot as plt
import seaborn as sns

def setup_plots():
    """Configure plot styling"""
    sns.set_style('darkgrid')
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (14, 7)

def plot_prices(df, company_name):
    """Plot opening and closing prices"""
    if df is None:
        return
        
    plt.figure()
    plt.plot(df['date'], df['close'], label='Closing', linewidth=2)
    plt.plot(df['date'], df['open'], label='Opening', linestyle='--')
    plt.title(f'{company_name} Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_profit_loss(df, company_name):
    """Plot intraday profit/loss"""
    if df is None:
        return
        
    plt.figure()
    colors = ['green' if x > 0 else 'red' for x in df['intraday_profit']]
    plt.bar(df['date'], df['intraday_profit'], color=colors)
    plt.title(f'{company_name} Daily Profit/Loss')
    plt.xlabel('Date')
    plt.ylabel('Amount ($)')
    plt.tight_layout()
    plt.show()

def plot_trend_distribution(df, company_name):
    """Plot trend distribution"""
    if df is None:
        return
        
    plt.figure(figsize=(8, 5))
    df['trend'].value_counts().plot(
        kind='bar', 
        color=['green', 'red', 'blue'],
        rot=0
    )
    plt.title(f'{company_name} Price Trends')
    plt.xlabel('Trend Direction')
    plt.ylabel('Days Count')
    plt.tight_layout()
    plt.show()

def compare_companies(data, metric='close'):
    """Compare all companies on a metric"""
    plt.figure()
    for name, df in data.items():
        if df is not None:
            plt.plot(df['date'], df[metric], label=name, linewidth=2)
    
    plt.title(f'Stock Comparison: {metric.capitalize()}')
    plt.xlabel('Date')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.show()