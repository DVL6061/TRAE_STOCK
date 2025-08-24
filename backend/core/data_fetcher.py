import yfinance as yf
import pandas as pd
import numpy as np
import logging
import asyncio
from datetime import datetime, timedelta
import ta  # Technical Analysis library
from typing import List, Dict, Any, Optional, Union

# This would be replaced with actual Angel One API client
# from angel_one_api import AngelOneClient

logger = logging.getLogger(__name__)

# Placeholder for Angel One API client
class MockAngelOneClient:
    """Mock implementation of Angel One API client for development"""
    
    def __init__(self):
        self.connected = False
        
    async def connect(self):
        self.connected = True
        logger.info("Connected to mock Angel One API")
        return True
        
    async def get_quote(self, ticker: str):
        # Generate mock real-time data
        base_price = 500.0  # Base price for mock data
        if ticker == "TATAMOTORS.NS":
            base_price = 800.0
            
        # Add some random variation
        current_price = base_price * (1 + np.random.normal(0, 0.005))
        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "change": round(current_price - base_price, 2),
            "change_percent": round((current_price - base_price) / base_price * 100, 2),
            "volume": int(np.random.randint(100000, 1000000)),
            "timestamp": datetime.now().isoformat(),
            "bid": round(current_price * 0.999, 2),
            "ask": round(current_price * 1.001, 2),
            "high": round(current_price * 1.01, 2),
            "low": round(current_price * 0.99, 2),
            "open": round(base_price, 2),
        }

# Initialize clients
# angel_one_client = AngelOneClient()  # This would be the actual client
angel_one_client = MockAngelOneClient()  # Using mock for now

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
            
        return data
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
        raise Exception(f"Failed to fetch historical data: {str(e)}")

async def fetch_realtime_data(ticker: str) -> Dict[str, Any]:
    """Fetch real-time stock data using Angel One API"""
    try:
        # Ensure connection to Angel One API
        if not angel_one_client.connected:
            await angel_one_client.connect()
            
        # Get real-time quote
        quote = await angel_one_client.get_quote(ticker)
        return quote
    except Exception as e:
        logger.error(f"Error fetching real-time data for {ticker}: {str(e)}")
        raise Exception(f"Failed to fetch real-time data: {str(e)}")

async def fetch_technical_indicators(
    ticker: str,
    indicators: List[str],
    period: int = 14,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Calculate technical indicators for a stock"""
    try:
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            # Default to 1 year of data for calculating indicators
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
        # Fetch historical data
        historical_data = await fetch_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval="1d"  # Daily data for indicators
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # Initialize result dictionary
        result = {indicator: [] for indicator in indicators}
        
        # Calculate requested indicators
        for indicator in indicators:
            if indicator.lower() == "rsi":
                # Relative Strength Index
                df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=period).rsi()
                for date, value in df["rsi"].items():
                    if not pd.isna(value):
                        result["rsi"].append({
                            "date": date.isoformat(),
                            "value": float(value)
                        })
                        
            elif indicator.lower() == "macd":
                # Moving Average Convergence Divergence
                macd = ta.trend.MACD(
                    df["close"],
                    window_slow=26,
                    window_fast=12,
                    window_sign=9
                )
                df["macd"] = macd.macd()
                df["macd_signal"] = macd.macd_signal()
                df["macd_diff"] = macd.macd_diff()
                
                for date, macd_val, signal, diff in zip(
                    df.index, df["macd"], df["macd_signal"], df["macd_diff"]
                ):
                    if not pd.isna(macd_val) and not pd.isna(signal) and not pd.isna(diff):
                        result["macd"].append({
                            "date": date.isoformat(),
                            "macd": float(macd_val),
                            "signal": float(signal),
                            "histogram": float(diff)
                        })
                        
            elif indicator.lower() == "sma":
                # Simple Moving Average
                df["sma"] = ta.trend.SMAIndicator(df["close"], window=period).sma_indicator()
                for date, value in df["sma"].items():
                    if not pd.isna(value):
                        result["sma"].append({
                            "date": date.isoformat(),
                            "value": float(value)
                        })
                        
            elif indicator.lower() == "ema":
                # Exponential Moving Average
                df["ema"] = ta.trend.EMAIndicator(df["close"], window=period).ema_indicator()
                for date, value in df["ema"].items():
                    if not pd.isna(value):
                        result["ema"].append({
                            "date": date.isoformat(),
                            "value": float(value)
                        })
                        
            elif indicator.lower() == "bb":
                # Bollinger Bands
                bollinger = ta.volatility.BollingerBands(
                    df["close"],
                    window=period,
                    window_dev=2
                )
                df["bb_upper"] = bollinger.bollinger_hband()
                df["bb_middle"] = bollinger.bollinger_mavg()
                df["bb_lower"] = bollinger.bollinger_lband()
                
                for date, upper, middle, lower in zip(
                    df.index, df["bb_upper"], df["bb_middle"], df["bb_lower"]
                ):
                    if not pd.isna(upper) and not pd.isna(middle) and not pd.isna(lower):
                        result["bb"].append({
                            "date": date.isoformat(),
                            "upper": float(upper),
                            "middle": float(middle),
                            "lower": float(lower)
                        })
                        
            elif indicator.lower() == "adx":
                # Average Directional Index
                adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=period)
                df["adx"] = adx.adx()
                df["adx_pos"] = adx.adx_pos()
                df["adx_neg"] = adx.adx_neg()
                
                for date, adx_val, pos, neg in zip(
                    df.index, df["adx"], df["adx_pos"], df["adx_neg"]
                ):
                    if not pd.isna(adx_val) and not pd.isna(pos) and not pd.isna(neg):
                        result["adx"].append({
                            "date": date.isoformat(),
                            "adx": float(adx_val),
                            "plus_di": float(pos),
                            "minus_di": float(neg)
                        })
                        
            elif indicator.lower() == "stoch":
                # Stochastic Oscillator
                stoch = ta.momentum.StochasticOscillator(
                    df["high"],
                    df["low"],
                    df["close"],
                    window=period,
                    smooth_window=3
                )
                df["stoch_k"] = stoch.stoch()
                df["stoch_d"] = stoch.stoch_signal()
                
                for date, k, d in zip(df.index, df["stoch_k"], df["stoch_d"]):
                    if not pd.isna(k) and not pd.isna(d):
                        result["stoch"].append({
                            "date": date.isoformat(),
                            "k": float(k),
                            "d": float(d)
                        })
            
        return result
    except Exception as e:
        logger.error(f"Error calculating indicators for {ticker}: {str(e)}")
        raise Exception(f"Failed to calculate indicators: {str(e)}")