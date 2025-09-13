import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, validator
import yfinance as yf
from ..utils.helpers import (
    validate_ticker, validate_date_range, calculate_technical_indicators,
    get_trading_days
)
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class MarketDataValidator(BaseModel):
    """Pydantic model for validating market data"""
    ticker: str
    start_date: datetime
    end_date: datetime
    interval: str = '1d'
    
    @validator('ticker')
    def validate_ticker_format(cls, v):
        if not validate_ticker(v):
            raise ValueError(f'Invalid ticker format: {v}')
        return v.upper()
    
    @validator('interval')
    def validate_interval(cls, v):
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if v not in valid_intervals:
            raise ValueError(f'Invalid interval: {v}. Must be one of {valid_intervals}')
        return v
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('End date must be after start date')
        return v

class NewsDataValidator(BaseModel):
    """Pydantic model for validating news data"""
    title: str
    content: str
    published_date: datetime
    source: str
    sentiment_score: Optional[float] = None
    
    @validator('sentiment_score')
    def validate_sentiment(cls, v):
        if v is not None and not (-1 <= v <= 1):
            raise ValueError('Sentiment score must be between -1 and 1')
        return v

class DataIntegrator:
    """Main class for integrating and validating data from multiple sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def fetch_market_data(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch and validate market data from Yahoo Finance"""
        
        try:
            # Validate input parameters
            validator = MarketDataValidator(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            self.logger.info(f"Fetching market data for {validator.ticker} from {start_date} to {end_date}")
            
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(validator.ticker)
            data = stock.history(
                start=validator.start_date,
                end=validator.end_date,
                interval=validator.interval
            )
            
            if data.empty:
                raise DataValidationError(f"No data found for ticker {validator.ticker}")
            
            # Clean and validate the data
            cleaned_data = self._clean_market_data(data, validator.ticker)
            
            # Add technical indicators
            enriched_data = self._add_technical_indicators(cleaned_data)
            
            self.logger.info(f"Successfully fetched and cleaned {len(enriched_data)} records for {validator.ticker}")
            return enriched_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {ticker}: {str(e)}")
            raise DataValidationError(f"Failed to fetch market data: {str(e)}")
    
    def _clean_market_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Clean and validate market data"""
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Standardize column names
        cleaned_data.columns = [col.lower() for col in cleaned_data.columns]
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in cleaned_data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with all NaN values
        cleaned_data = cleaned_data.dropna(how='all')
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (cleaned_data['high'] < cleaned_data['low']) |
            (cleaned_data['high'] < cleaned_data['open']) |
            (cleaned_data['high'] < cleaned_data['close']) |
            (cleaned_data['low'] > cleaned_data['open']) |
            (cleaned_data['low'] > cleaned_data['close'])
        )
        
        if invalid_ohlc.any():
            self.logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships for {ticker}")
            # Remove invalid rows
            cleaned_data = cleaned_data[~invalid_ohlc]
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        negative_prices = (cleaned_data[price_columns] <= 0).any(axis=1)
        if negative_prices.any():
            self.logger.warning(f"Found {negative_prices.sum()} rows with negative prices for {ticker}")
            cleaned_data = cleaned_data[~negative_prices]
        
        # Check for negative volume
        negative_volume = cleaned_data['volume'] < 0
        if negative_volume.any():
            self.logger.warning(f"Found {negative_volume.sum()} rows with negative volume for {ticker}")
            cleaned_data = cleaned_data[~negative_volume]
        
        # Handle missing values using forward fill and backward fill
        cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers using IQR method
        cleaned_data = self._remove_price_outliers(cleaned_data, ticker)
        
        # Ensure data is sorted by date
        cleaned_data = cleaned_data.sort_index()
        
        return cleaned_data
    
    def _remove_price_outliers(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Remove price outliers using Interquartile Range (IQR) method"""
        
        # Calculate daily returns
        returns = data['close'].pct_change().dropna()
        
        # Calculate IQR for returns
        Q1 = returns.quantile(0.25)
        Q3 = returns.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds (using 3*IQR for less aggressive filtering)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Identify outliers
        outliers = (returns < lower_bound) | (returns > upper_bound)
        
        if outliers.any():
            self.logger.info(f"Removing {outliers.sum()} outlier records for {ticker}")
            # Remove outlier rows (shift by 1 because returns are calculated with pct_change)
            outlier_indices = returns[outliers].index
            data = data.drop(outlier_indices, errors='ignore')
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the market data"""
        
        try:
            # Calculate technical indicators using helper function
            indicators = calculate_technical_indicators(data)
            
            # Merge indicators with original data
            enriched_data = pd.concat([data, indicators], axis=1)
            
            return enriched_data
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            # Return original data if technical indicators fail
            return data
    
    async def validate_news_data(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean news data"""
        
        validated_news = []
        
        for item in news_items:
            try:
                # Validate using Pydantic model
                validated_item = NewsDataValidator(**item)
                validated_news.append(validated_item.dict())
                
            except Exception as e:
                self.logger.warning(f"Invalid news item skipped: {str(e)}")
                continue
        
        self.logger.info(f"Validated {len(validated_news)} out of {len(news_items)} news items")
        return validated_news
    
    def check_data_quality(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Generate data quality report"""
        
        if data.empty:
            return {"status": "error", "message": "No data available"}
        
        quality_report = {
            "ticker": ticker,
            "total_records": len(data),
            "date_range": {
                "start": data.index.min().strftime('%Y-%m-%d'),
                "end": data.index.max().strftime('%Y-%m-%d')
            },
            "missing_values": data.isnull().sum().to_dict(),
            "data_completeness": (1 - data.isnull().sum() / len(data)).to_dict(),
            "price_statistics": {
                "close_mean": round(data['close'].mean(), 2),
                "close_std": round(data['close'].std(), 2),
                "volume_mean": round(data['volume'].mean(), 0),
                "volume_std": round(data['volume'].std(), 0)
            },
            "status": "success"
        }
        
        # Check for data quality issues
        issues = []
        
        # Check for high missing value percentage
        missing_pct = data.isnull().sum() / len(data)
        high_missing = missing_pct[missing_pct > 0.1]
        if not high_missing.empty:
            issues.append(f"High missing values: {high_missing.to_dict()}")
        
        # Check for low volume days
        low_volume_days = (data['volume'] == 0).sum()
        if low_volume_days > 0:
            issues.append(f"Found {low_volume_days} days with zero volume")
        
        # Check data recency
        last_date = data.index.max()
        days_old = (datetime.now() - last_date).days
        if days_old > 7:
            issues.append(f"Data is {days_old} days old")
        
        quality_report["issues"] = issues
        quality_report["quality_score"] = max(0, 100 - len(issues) * 10)
        
        return quality_report

# Global instance
data_integrator = DataIntegrator()