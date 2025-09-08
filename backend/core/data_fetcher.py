import yfinance as yf
import pandas as pd
import numpy as np
import logging
import asyncio
from datetime import datetime, timedelta
import ta  # Technical Analysis library
from typing import List, Dict, Any, Optional, Union
import requests
import pyotp
import json
import os
from dotenv import load_dotenv

load_dotenv()

ANGEL_ONE_API_KEY = os.getenv('ANGEL_ONE_API_KEY')
ANGEL_ONE_CLIENT_ID = os.getenv('ANGEL_ONE_CLIENT_ID') 
ANGEL_ONE_PASSWORD = os.getenv('ANGEL_ONE_PASSWORD')    
ANGEL_ONE_TOTP = os.getenv('ANGEL_ONE_TOTP_SECRET')
ANGEL_ONE_API_URL = os.getenv('ANGEL_ONE_API_URL')

logger = logging.getLogger(__name__)

class AngelOneClient:
    """Real Angel One SmartAPI client implementation"""
    
    def __init__(self, api_key: str, client_id: str, password: str, totp_token: str = None):
        self.api_key = api_key
        self.client_id = client_id
        self.password = password
        self.totp_token = totp_token
        self.base_url = "https://apiconnect.angelone.in"
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None
        self.connected = False
        self.session = requests.Session()
        
    async def connect(self):
        """Authenticate and establish connection to Angel One API"""
        try:
            # Generate TOTP if token is provided
            totp = None
            if self.totp_token:
                totp = pyotp.TOTP(self.totp_token).now()
            
            # Login payload
            login_data = {
                "clientcode": self.client_id,
                "password": self.password
            }
            
            if totp:
                login_data["totp"] = totp
            
            # Login request
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            response = self.session.post(
                f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword",
                headers=headers,
                json=login_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    self.auth_token = data['data']['jwtToken']
                    self.refresh_token = data['data']['refreshToken']
                    self.feed_token = data['data']['feedToken']
                    self.connected = True
                    logger.info("Successfully connected to Angel One API")
                    return True
                else:
                    logger.error(f"Angel One login failed: {data.get('message', 'Unknown error')}")
                    return False
            else:
                logger.error(f"Angel One API connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Angel One API: {str(e)}")
            return False
    
    async def get_quote(self, ticker: str):
        """Get real-time quote for a stock"""
        try:
            if not self.connected:
                await self.connect()
            
            # Convert ticker format (e.g., TATAMOTORS.NS to NSE:TATAMOTORS-EQ)
            symbol_token = await self._get_symbol_token(ticker)
            if not symbol_token:
                raise Exception(f"Could not find symbol token for {ticker}")
            
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "00:00:00:00:00:00",
                "X-PrivateKey": self.api_key
            }
            
            # Get LTP (Last Traded Price)
            ltp_data = {
                "exchange": "NSE",
                "tradingsymbol": ticker.replace(".NS", "-EQ"),
                "symboltoken": symbol_token
            }
            
            response = self.session.post(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/getLTP",
                headers=headers,
                json=ltp_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    quote_data = data['data']
                    return {
                        "ticker": ticker,
                        "price": float(quote_data.get('ltp', 0)),
                        "change": float(quote_data.get('change', 0)),
                        "change_percent": float(quote_data.get('pChange', 0)),
                        "volume": int(quote_data.get('volume', 0)),
                        "timestamp": datetime.now().isoformat(),
                        "bid": float(quote_data.get('bid', 0)),
                        "ask": float(quote_data.get('ask', 0)),
                        "high": float(quote_data.get('high', 0)),
                        "low": float(quote_data.get('low', 0)),
                        "open": float(quote_data.get('open', 0)),
                    }
                else:
                    logger.error(f"Angel One quote fetch failed: {data.get('message', 'Unknown error')}")
                    raise Exception(f"Failed to fetch quote: {data.get('message', 'Unknown error')}")
            else:
                logger.error(f"Angel One quote API failed: {response.status_code}")
                raise Exception(f"API request failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching quote for {ticker}: {str(e)}")
            # Fallback to Yahoo Finance for development
            return await self._fallback_yahoo_quote(ticker)
    
    async def _get_symbol_token(self, ticker: str) -> Optional[str]:
        """Get symbol token for a ticker (simplified implementation)"""
        # This is a simplified mapping - in production, you'd fetch from Angel One's instrument list
        symbol_mapping = {
            "TATAMOTORS.NS": "884",
            "RELIANCE.NS": "2885",
            "INFY.NS": "1594",
            "TCS.NS": "11536",
            "HDFCBANK.NS": "1333",
            "ICICIBANK.NS": "4963",
            "SBIN.NS": "3045",
            "WIPRO.NS": "3787"
        }
        return symbol_mapping.get(ticker)
    
    async def _fallback_yahoo_quote(self, ticker: str):
        """Fallback to Yahoo Finance if Angel One fails"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                current_price = float(latest['Close'])
                open_price = float(latest['Open'])
                
                return {
                    "ticker": ticker,
                    "price": current_price,
                    "change": current_price - open_price,
                    "change_percent": ((current_price - open_price) / open_price) * 100,
                    "volume": int(latest['Volume']),
                    "timestamp": datetime.now().isoformat(),
                    "bid": current_price * 0.999,
                    "ask": current_price * 1.001,
                    "high": float(latest['High']),
                    "low": float(latest['Low']),
                    "open": open_price,
                }
        except Exception as e:
            logger.error(f"Yahoo Finance fallback failed for {ticker}: {str(e)}")
            raise Exception(f"Both Angel One and Yahoo Finance failed for {ticker}")

# Placeholder for Angel One API client (for backward compatibility during development)
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
# Try to use real Angel One client if credentials are available
if ANGEL_ONE_API_KEY and ANGEL_ONE_CLIENT_ID and ANGEL_ONE_PASSWORD:
    angel_one_client = AngelOneClient(
        api_key=ANGEL_ONE_API_KEY,
        client_id=ANGEL_ONE_CLIENT_ID,
        password=ANGEL_ONE_PASSWORD
    )
    logger.info("Initialized real Angel One API client")
else:
    angel_one_client = MockAngelOneClient()  # Using mock for development
    logger.info("Using mock Angel One API client - set environment variables for real API")

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