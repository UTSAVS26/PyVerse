import pandas as pd
import numpy as np

def analyze_trends(data):
    """Analyze trends across all companies"""
    results = {}
    for name, df in data.items():
        if df is not None:
            results[name] = {
                'up_trend_pct': round((df['trend'] == 'UP').mean() * 100, 1),
                'down_trend_pct': round((df['trend'] == 'DOWN').mean() * 100, 1),
                'avg_profit': round(df[df['intraday_profit'] > 0]['intraday_profit'].mean(), 2),
                'avg_loss': round(df[df['intraday_profit'] < 0]['intraday_profit'].mean(), 2)
            }
    return pd.DataFrame(results).T

def correlation_analysis(data):
    """Calculate correlation between companies"""
    closes = pd.DataFrame()
    for name, df in data.items():
        if df is not None:
            closes[name] = df['close']
    
    if not closes.empty:
        return closes.corr()
    return None