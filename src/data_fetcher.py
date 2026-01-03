import os
import yfinance as yf
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

    """
    DATA:
    1. Get the S&P 500 (SPY), Gold (GLD), and Crude Oil (USO) data from YFinance
    2. Get the GDP, CPI, Unemployment, Volatility data from FRED.
    3. stored as assets
    """


# Load credentials from .env
# FRED API used to extract the macroeconomic trends of financial markets
# visit https://fredaccount.stlouisfed.org/ to get your own API key
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")


def fetch_all_data():
    
    """ Fetch asset prices and macroeconomic indicators and save them to CSV files. """
    assets = ["SPY", "GLD", "USO"]
    price_df = yf.download(assets, start="2020-03-01", end="2025-12-31", auto_adjust=False)
    price_df.to_csv("data/raw_prices.csv")

    fred = Fred(api_key=FRED_API_KEY)
    macro_map = {
            "GDP": "GDP",              # Quarterly
            "CPI": "CPIAUCSL",         # Monthly
            "Unemployment": "UNRATE",  # Monthly
            "VIX": "VIXCLS"            # Daily
        } 
    macro_data = {}
    for name, fred_id in macro_map.items():
        macro_data[name] = fred.get_series(fred_id, observation_start="2020-03-01")
    macro_df = pd.DataFrame(macro_data)
    macro_df.to_csv("data/macro_indicators.csv")


if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    fetch_all_data()