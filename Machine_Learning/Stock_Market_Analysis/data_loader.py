import pandas as pd
import numpy as np
import requests
from io import StringIO

# Dataset URLs
DATA_URLS = {
    'apple': '/kaggle/input/stock-prices-for/AAPL_data.csv',     # DECLARING DIRECTORY 
    'amazon': '/kaggle/input/stock-prices-for/AMZN_data.csv',    # DECLARING DIRECTORY
    'google': '/kaggle/input/stock-prices-for/GOOG_data.csv',    # DECLARING DIRECTORY
    'microsoft': '/kaggle/input/stock-prices-for/MSFT_data.csv'  # DECLARING DIRECTORY
}

def download_stock_data():
    """Download and process all stock data from URLs"""
    data = {}
    
    for name, url in DATA_URLS.items():
        try:
            # Download CSV data
            response = requests.get(url)
            response.raise_for_status()
            
            # Read into DataFrame
            df = pd.read_csv(StringIO(response.text))
            
            # Process data
            df['date'] = pd.to_datetime(df['date'])
            df['intraday_profit'] = df['close'] - df['open']
            df['trend'] = np.where(
                df['open'] < df['close'], 'UP',
                np.where(df['open'] > df['close'], 'DOWN', 'FLAT')
            )
            
            data[name] = df
            print(f"Successfully downloaded {name} data")
            
        except Exception as e:
            print(f"Error downloading {name} data: {str(e)}")
            data[name] = None
    
    return data

def get_basic_stats(df):
    """Calculate basic statistics for a dataframe"""
    if df is None:
        return None
        
    return {
        'start_date': df['date'].min().strftime('%Y-%m-%d'),
        'end_date': df['date'].max().strftime('%Y-%m-%d'),
        'average_close': round(df['close'].mean(), 2),
        'max_close': round(df['close'].max(), 2),
        'min_close': round(df['close'].min(), 2),
        'up_days': (df['trend'] == 'UP').sum(),
        'down_days': (df['trend'] == 'DOWN').sum(),
        'total_days': len(df)
    }