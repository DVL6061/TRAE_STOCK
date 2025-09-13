import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import os
import re
import talib
from talib import abstract

logger = logging.getLogger(__name__)

def validate_ticker(ticker: str) -> str:
    """Validate and format stock ticker"""
    # Remove any whitespace
    ticker = ticker.strip()
    
    # Check if ticker is empty
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    
    # For Indian stocks, ensure .NS or .BO suffix for NSE/BSE
    if re.match(r'^[A-Z0-9]+$', ticker) and not ticker.endswith((".NS", ".BO")):
        # Default to NSE
        ticker = f"{ticker}.NS"
    
    return ticker

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object"""
    try:
        return pd.to_datetime(date_str)
    except Exception as e:
        logger.error(f"Error parsing date '{date_str}': {str(e)}")
        raise ValueError(f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD")

def validate_date_range(start_date: str, end_date: str) -> tuple:
    """Validate date range and return datetime objects"""
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    # Check if start date is before end date
    if start > end:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
    
    # Check if end date is in the future
    if end > datetime.now():
        logger.warning(f"End date ({end_date}) is in the future, setting to current date")
        end = datetime.now()
    
    return start, end

def calculate_returns(prices: List[float]) -> List[float]:
    """Calculate returns from a list of prices"""
    if len(prices) < 2:
        return []
    
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] == 0:
            # Avoid division by zero
            returns.append(0)
        else:
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
    
    return returns

def calculate_cumulative_returns(returns: List[float]) -> List[float]:
    """Calculate cumulative returns from a list of returns"""
    cumulative = [1.0]
    for r in returns:
        cumulative.append(cumulative[-1] * (1 + r))
    
    # Convert to percentage change from initial value
    cumulative = [(c - 1) * 100 for c in cumulative]
    
    return cumulative

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.05) -> float:
    """Calculate Sharpe ratio from a list of returns"""
    if len(returns) < 2:
        return 0
    
    # Convert annual risk-free rate to match returns frequency (assuming daily returns)
    daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
    
    excess_returns = [r - daily_risk_free for r in returns]
    avg_excess_return = sum(excess_returns) / len(excess_returns)
    std_dev = np.std(excess_returns)
    
    if std_dev == 0:
        return 0
    
    # Annualize Sharpe ratio (assuming daily returns)
    sharpe = avg_excess_return / std_dev * np.sqrt(252)
    
    return sharpe

def calculate_max_drawdown(cumulative_returns: List[float]) -> float:
    """Calculate maximum drawdown from a list of cumulative returns"""
    if len(cumulative_returns) < 2:
        return 0
    
    # Convert percentage returns to values
    values = [(1 + r/100) for r in cumulative_returns]
    
    # Calculate running maximum
    running_max = [values[0]]
    for v in values[1:]:
        running_max.append(max(running_max[-1], v))
    
    # Calculate drawdowns
    drawdowns = [(v / rm - 1) * 100 for v, rm in zip(values, running_max)]
    
    # Find maximum drawdown
    max_drawdown = min(drawdowns)
    
    return max_drawdown

def format_currency(value: float, currency: str = "INR") -> str:
    """Format value as currency string"""
    if currency == "INR":
        # Format as Indian Rupees
        if value >= 10000000:  # 1 crore
            return f"₹{value/10000000:.2f} Cr"
        elif value >= 100000:  # 1 lakh
            return f"₹{value/100000:.2f} L"
        else:
            return f"₹{value:.2f}"
    else:
        # Default format
        return f"{currency} {value:.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage string"""
    return f"{value:.2f}%"

def save_to_json(data: Any, filepath: str) -> bool:
    """Save data to JSON file"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {str(e)}")
        return False

def load_from_json(filepath: str) -> Any:
    """Load data from JSON file"""
    try:
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        return None

def get_timeframe_days(timeframe: str) -> int:
    """Convert timeframe string to number of days"""
    # Map common timeframes to days
    timeframe_map = {
        "1d": 1,
        "3d": 3,
        "5d": 5,
        "1w": 7,
        "2w": 14,
        "1m": 30,
        "3m": 90,
        "6m": 180,
        "1y": 365
    }
    
    # Check if timeframe is in map
    if timeframe in timeframe_map:
        return timeframe_map[timeframe]
    
    # Try to parse timeframe string
    match = re.match(r'^(\d+)([dwmy])$', timeframe.lower())
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        
        if unit == 'd':
            return value
        elif unit == 'w':
            return value * 7
        elif unit == 'm':
            return value * 30
        elif unit == 'y':
            return value * 365
    
    # Default to 1 day if timeframe is not recognized
    logger.warning(f"Unrecognized timeframe: {timeframe}, defaulting to 1 day")
    return 1

def translate_text(text: str, target_language: str) -> str:
    """Translate text to target language"""
    # This is a placeholder for actual translation functionality
    # In a real implementation, this would use a translation API or library
    
    # For now, we'll just return the original text
    logger.warning(f"Translation to {target_language} not implemented, returning original text")
    return text

def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Get list of trading days between start and end dates"""
    # This is a simplified implementation that excludes weekends
    # In a real implementation, this would also exclude holidays
    
    trading_days = []
    current_date = start_date
    
    while current_date <= end_date:
        # Skip weekends (5 = Saturday, 6 = Sunday)
        if current_date.weekday() < 5:
            trading_days.append(current_date)
        
        current_date += timedelta(days=1)
    
    return trading_days

def is_market_open() -> bool:
    """Check if the market is currently open"""
    # This is a simplified implementation for Indian markets
    # In a real implementation, this would check actual market hours and holidays
    
    now = datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check if it's within market hours (9:15 AM to 3:30 PM IST)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close

# Technical Indicators Implementation using TA-Lib

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return talib.SMA(data.values, timeperiod=period)

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return talib.EMA(data.values, timeperiod=period)

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    return talib.RSI(data.values, timeperiod=period)

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    macd, macd_signal, macd_histogram = talib.MACD(data.values, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    return {
        'macd': pd.Series(macd, index=data.index),
        'macd_signal': pd.Series(macd_signal, index=data.index),
        'macd_histogram': pd.Series(macd_histogram, index=data.index)
    }

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    bb_upper, bb_middle, bb_lower = talib.BBANDS(data.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
    return {
        'bb_upper': pd.Series(bb_upper, index=data.index),
        'bb_middle': pd.Series(bb_middle, index=data.index),
        'bb_lower': pd.Series(bb_lower, index=data.index)
    }

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator"""
    slowk, slowd = talib.STOCH(high.values, low.values, close.values, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
    return {
        'stoch_k': pd.Series(slowk, index=close.index),
        'stoch_d': pd.Series(slowd, index=close.index)
    }

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    return talib.ATR(high.values, low.values, close.values, timeperiod=period)

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index"""
    return talib.ADX(high.values, low.values, close.values, timeperiod=period)

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Commodity Channel Index"""
    return talib.CCI(high.values, low.values, close.values, timeperiod=period)

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R"""
    return talib.WILLR(high.values, low.values, close.values, timeperiod=period)

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume"""
    return talib.OBV(close.values, volume.values)

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

def calculate_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for a given OHLCV DataFrame.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
    Returns:
        DataFrame with all technical indicators added
    """
    result_df = df.copy()
    
    try:
        # Moving Averages
        result_df['sma_5'] = calculate_sma(df['close'], 5)
        result_df['sma_10'] = calculate_sma(df['close'], 10)
        result_df['sma_20'] = calculate_sma(df['close'], 20)
        result_df['sma_50'] = calculate_sma(df['close'], 50)
        result_df['sma_200'] = calculate_sma(df['close'], 200)
        
        result_df['ema_12'] = calculate_ema(df['close'], 12)
        result_df['ema_26'] = calculate_ema(df['close'], 26)
        result_df['ema_50'] = calculate_ema(df['close'], 50)
        
        # RSI
        result_df['rsi'] = calculate_rsi(df['close'])
        
        # MACD
        macd_data = calculate_macd(df['close'])
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['macd_signal']
        result_df['macd_histogram'] = macd_data['macd_histogram']
        
        # Bollinger Bands
        bb_data = calculate_bollinger_bands(df['close'])
        result_df['bb_upper'] = bb_data['bb_upper']
        result_df['bb_middle'] = bb_data['bb_middle']
        result_df['bb_lower'] = bb_data['bb_lower']
        
        # Stochastic
        stoch_data = calculate_stochastic(df['high'], df['low'], df['close'])
        result_df['stoch_k'] = stoch_data['stoch_k']
        result_df['stoch_d'] = stoch_data['stoch_d']
        
        # ATR
        result_df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
        
        # ADX
        result_df['adx'] = calculate_adx(df['high'], df['low'], df['close'])
        
        # CCI
        result_df['cci'] = calculate_cci(df['high'], df['low'], df['close'])
        
        # Williams %R
        result_df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'])
        
        # OBV
        result_df['obv'] = calculate_obv(df['close'], df['volume'])
        
        # VWAP
        result_df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Additional derived indicators
        result_df['price_sma20_ratio'] = df['close'] / result_df['sma_20']
        result_df['volume_sma20'] = calculate_sma(df['volume'], 20)
        result_df['volume_ratio'] = df['volume'] / result_df['volume_sma20']
        
        logger.info("Successfully calculated all technical indicators")
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        raise
    
    return result_df

def get_trading_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate trading signals based on technical indicators.
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        Dictionary containing trading signals and their strengths
    """
    signals = {
        'buy_signals': [],
        'sell_signals': [],
        'neutral_signals': [],
        'overall_signal': 'HOLD',
        'signal_strength': 0.0
    }
    
    try:
        latest = df.iloc[-1]
        
        # RSI Signals
        if latest['rsi'] < 30:
            signals['buy_signals'].append('RSI Oversold')
        elif latest['rsi'] > 70:
            signals['sell_signals'].append('RSI Overbought')
        else:
            signals['neutral_signals'].append('RSI Neutral')
        
        # MACD Signals
        if latest['macd'] > latest['macd_signal']:
            signals['buy_signals'].append('MACD Bullish')
        else:
            signals['sell_signals'].append('MACD Bearish')
        
        # Bollinger Bands Signals
        if latest['close'] < latest['bb_lower']:
            signals['buy_signals'].append('BB Oversold')
        elif latest['close'] > latest['bb_upper']:
            signals['sell_signals'].append('BB Overbought')
        else:
            signals['neutral_signals'].append('BB Neutral')
        
        # Moving Average Signals
        if latest['close'] > latest['sma_20'] > latest['sma_50']:
            signals['buy_signals'].append('MA Bullish Trend')
        elif latest['close'] < latest['sma_20'] < latest['sma_50']:
            signals['sell_signals'].append('MA Bearish Trend')
        
        # Calculate overall signal
        buy_count = len(signals['buy_signals'])
        sell_count = len(signals['sell_signals'])
        
        if buy_count > sell_count:
            signals['overall_signal'] = 'BUY'
            signals['signal_strength'] = min(0.9, (buy_count - sell_count) / 10)
        elif sell_count > buy_count:
            signals['overall_signal'] = 'SELL'
            signals['signal_strength'] = min(0.9, (sell_count - buy_count) / 10)
        else:
            signals['overall_signal'] = 'HOLD'
            signals['signal_strength'] = 0.5
        
    except Exception as e:
        logger.error(f"Error generating trading signals: {str(e)}")
        signals['overall_signal'] = 'HOLD'
        signals['signal_strength'] = 0.0
    
    return signals