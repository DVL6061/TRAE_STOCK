from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

class TimeFrame(str, Enum):
    """Supported timeframes for predictions."""
    SCALPING = "5m"
    INTRADAY = "1h"
    SWING = "1d"
    POSITION = "1wk"
    LONG_TERM = "1mo"

class MarketDataPoint(BaseModel):
    """Single market data point with OHLCV data."""
    timestamp: datetime
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    
    @validator('high')
    def high_must_be_highest(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('High must be greater than or equal to low')
        return v
    
    @validator('low')
    def low_must_be_lowest(cls, v, values):
        if 'high' in values and v > values['high']:
            raise ValueError('Low must be less than or equal to high')
        return v

class TechnicalIndicators(BaseModel):
    """Technical indicators for market analysis."""
    sma_5: Optional[float] = Field(None, description="5-period Simple Moving Average")
    sma_20: Optional[float] = Field(None, description="20-period Simple Moving Average")
    sma_50: Optional[float] = Field(None, description="50-period Simple Moving Average")
    sma_200: Optional[float] = Field(None, description="200-period Simple Moving Average")
    
    ema_12: Optional[float] = Field(None, description="12-period Exponential Moving Average")
    ema_26: Optional[float] = Field(None, description="26-period Exponential Moving Average")
    
    rsi: Optional[float] = Field(None, ge=0, le=100, description="Relative Strength Index")
    
    macd: Optional[float] = Field(None, description="MACD line")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram")
    
    bb_upper: Optional[float] = Field(None, description="Bollinger Bands upper band")
    bb_middle: Optional[float] = Field(None, description="Bollinger Bands middle band")
    bb_lower: Optional[float] = Field(None, description="Bollinger Bands lower band")
    
    atr: Optional[float] = Field(None, ge=0, description="Average True Range")
    adx: Optional[float] = Field(None, ge=0, le=100, description="Average Directional Index")
    
class MarketSummary(BaseModel):
    """Market summary for major indices."""
    nifty: Dict[str, Union[float, int]] = Field(..., description="NIFTY 50 data")
    sensex: Dict[str, Union[float, int]] = Field(..., description="SENSEX data")
    bank_nifty: Dict[str, Union[float, int]] = Field(..., description="BANK NIFTY data")
    
class StockData(BaseModel):
    """Complete stock data with technical indicators."""
    ticker: str = Field(..., description="Stock ticker symbol")
    market_data: MarketDataPoint
    technical_indicators: TechnicalIndicators
    
class PortfolioSummary(BaseModel):
    """Portfolio summary information."""
    total_value: float = Field(..., description="Total portfolio value")
    day_change: float = Field(..., description="Day change in value")
    day_change_percent: float = Field(..., description="Day change percentage")
    total_return: float = Field(..., description="Total return")
    total_return_percent: float = Field(..., description="Total return percentage")
    holdings_count: int = Field(..., ge=0, description="Number of holdings")
    
class WatchlistItem(BaseModel):
    """Watchlist item with basic information."""
    ticker: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    current_price: float = Field(..., gt=0, description="Current price")
    change: float = Field(..., description="Price change")
    change_percent: float = Field(..., description="Price change percentage")
    volume: int = Field(..., ge=0, description="Trading volume")
    
class MarketStatus(BaseModel):
    """Market status information."""
    is_open: bool = Field(..., description="Whether market is open")
    next_open: Optional[datetime] = Field(None, description="Next market open time")
    next_close: Optional[datetime] = Field(None, description="Next market close time")
    timezone: str = Field(default="Asia/Kolkata", description="Market timezone")
    
class SectorAllocation(BaseModel):
    """Sector allocation data."""
    name: str = Field(..., description="Sector name")
    value: float = Field(..., ge=0, le=100, description="Allocation percentage")
    color: str = Field(..., description="Display color")
    
class MarketOverview(BaseModel):
    """Complete market overview."""
    market_summary: MarketSummary
    market_status: MarketStatus
    top_gainers: List[WatchlistItem]
    top_losers: List[WatchlistItem]
    most_active: List[WatchlistItem]
    sector_performance: List[SectorAllocation]