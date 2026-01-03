import pandas as pd
import numpy as np
import os


"""
    This file is where the preprocessing of the data takes place.
    Here the core technical indicators are defined as computed - Moving Averages, Relative Strength Index and Rolling Volatility computed
        
"""
    

def calculate_indicators(df, tickers):
    """Calculates the indicators for each asset.
        Moving Averages: To identitfy trends by smoothing out price data over specific periods
        Relative Strength Index (RSI): To measure the speed and change of price movements, indicating overbought or oversold conditions
        Rolling Volatility: To assess the degree of variation in asset prices over a specified period, indicating market risk and stability
    Args:
    """
    processed = pd.DataFrame(index=df.index)
    
    for ticker in tickers:
        # Extract Adj Close for specific ticker
        price = df['Adj Close'][ticker]
        
        # pct Returns - change pct in price from previous day
        processed[f"{ticker}_Return"] = price.pct_change()
        
        # SMA (50 & 200 days)
        processed[f"{ticker}_SMA_50"] = price.rolling(window=50).mean()
        processed[f"{ticker}_SMA_200"] = price.rolling(window=200).mean()
        
        # RSI (14 days)
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        processed[f"{ticker}_RSI_14"] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26)
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        processed[f"{ticker}_MACD"] = ema12 - ema26
        
        # Rolling Volatility (30 days)
        processed[f"{ticker}_Vol_30"] = processed[f"{ticker}_Return"].rolling(window=30).std()

    return processed

def process_data():
    # 1. Load Data
    # raw_prices has multi-level headers: Level 0 (Price Type), Level 1 (Ticker)
    prices = pd.read_csv("data/raw_prices.csv", header=[0, 1], index_col=0, parse_dates=True)
    macro = pd.read_csv("data/macro_indicators.csv", index_col=0, parse_dates=True)
    
    # 2. Extract unique tickers from columns
    tickers = prices.columns.get_level_values(1).unique()
    
    # 3. Process Technical Features
    tech_df = calculate_indicators(prices, tickers)
    
    # 4. Align Macro Data
    # Reindex macro to match daily stock dates and forward-fill sparse GDP/CPI data
    macro_daily = macro.reindex(tech_df.index).ffill()
    
    # 5. Final Merge
    master_df = pd.concat([tech_df, macro_daily], axis=1)
    
    # 6. Prevent Look-ahead Bias
    # The agent at time 't' should only see data from 't-1'
    master_df = master_df.shift(1)
    
    # 7. Final Cleanup
    master_df.dropna(inplace=True)
    master_df.to_csv("data/processed_features.csv")


if __name__ == "__main__":
    process_data()

