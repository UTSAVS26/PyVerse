from data_loader import download_stock_data, get_basic_stats
from visualizer import setup_plots, plot_prices, plot_profit_loss, plot_trend_distribution, compare_companies
from analyzer import analyze_trends, correlation_analysis
import pandas as pd

def display_stats(data):
    """Display basic statistics for all companies"""
    stats = {}
    for name, df in data.items():
        if df is not None:
            stats[name] = get_basic_stats(df)
    
    return pd.DataFrame(stats).T

def main():
    # Setup visualization
    setup_plots()
    
    # Download and process data
    print("Downloading stock data...")
    stock_data = download_stock_data()
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(display_stats(stock_data))
    
    # Analyze and visualize each company
    for name, df in stock_data.items():
        if df is not None:
            print(f"\nAnalyzing {name}...")
            plot_prices(df, name)
            plot_profit_loss(df, name)
            plot_trend_distribution(df, name)
    
    # Comparative analysis
    print("\nComparative Analysis:")
    compare_companies(stock_data, 'close')
    compare_companies(stock_data, 'volume')
    
    # Advanced analysis
    print("\nTrend Analysis:")
    print(analyze_trends(stock_data))
    
    print("\nPrice Correlation:")
    print(correlation_analysis(stock_data))

if __name__ == "__main__":
    main()