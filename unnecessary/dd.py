import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Any, Optional, Union

async def fetch_historical_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> List[Dict[str, Any]]:
    """Fetch historical OHLCV data for a stock using Yahoo Finance"""
    try:
        # Convert dates to datetime objects
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Add one day to end_date to include the end date in the results
        end = end + timedelta(days=1)
        
        # Fetch data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval=interval)
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        # Convert DataFrame to list of dictionaries
        data = []
        for _, row in df.iterrows():
            data_point = {
                "date": row["Date"].isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
            data.append(data_point)
           
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {str(e)}")
        return []

async def main():
    data = await fetch_historical_data("TATAMOTORS.NS", "2023-01-01", "2024-12-31")
    print(data.head(1500))
    data.to_csv("TATAMOTORS.NS.xls", index=False)

if __name__ == "__main__":
    asyncio.run(main())
