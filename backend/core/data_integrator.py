import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from backend.core.data_fetcher import DataFetcher
from backend.core.technical_indicators import TechnicalIndicators
from backend.core.fundamental_analyzer import FundamentalAnalyzer
from backend.core.news_sentiment import NewsSentimentAnalyzer
from backend.utils.config import PREDICTION_WINDOWS, TRADING_TIMEFRAMES

logger = logging.getLogger(__name__)

class DataIntegrator:
    """
    Comprehensive data integration module that combines:
    - Historical OHLCV data
    - Technical indicators
    - Fundamental analysis metrics
    - News sentiment analysis
    - Real-time market data
    """
    
    def __init__(self):
        """Initialize the data integrator with all required components."""
        self.data_fetcher = DataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.news_sentiment = NewsSentimentAnalyzer()
        
        # Cache for storing processed data
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)  # Cache for 15 minutes
        
    def get_comprehensive_data(self, 
                             ticker: str, 
                             timeframe: str = '1d',
                             period: str = '1y',
                             include_sentiment: bool = True,
                             include_fundamental: bool = True) -> pd.DataFrame:
        """
        Get comprehensive dataset combining all data sources.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe (1m, 5m, 15m, 1h, 1d, etc.)
            period: Historical period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            include_sentiment: Whether to include sentiment analysis
            include_fundamental: Whether to include fundamental analysis
            
        Returns:
            DataFrame with comprehensive market data
        """
        try:
            # Check cache first
            cache_key = f"{ticker}_{timeframe}_{period}_{include_sentiment}_{include_fundamental}"
            if self._is_cache_valid(cache_key):
                logger.info(f"Returning cached data for {ticker}")
                return self.cache[cache_key]['data']
            
            logger.info(f"Fetching comprehensive data for {ticker} ({timeframe}, {period})")
            
            # 1. Get historical OHLCV data
            historical_data = self.data_fetcher.get_historical_data(
                ticker=ticker,
                timeframe=timeframe,
                period=period
            )
            
            if historical_data.empty:
                logger.error(f"No historical data available for {ticker}")
                return pd.DataFrame()
            
            # 2. Add technical indicators
            data_with_indicators = self._add_technical_indicators(historical_data)
            
            # 3. Add fundamental analysis (if requested)
            if include_fundamental:
                data_with_fundamental = self._add_fundamental_features(
                    data_with_indicators, ticker
                )
            else:
                data_with_fundamental = data_with_indicators
            
            # 4. Add sentiment analysis (if requested)
            if include_sentiment:
                final_data = self._add_sentiment_features(
                    data_with_fundamental, ticker
                )
            else:
                final_data = data_with_fundamental
            
            # 5. Add derived features
            final_data = self._add_derived_features(final_data)
            
            # 6. Clean and validate data
            final_data = self._clean_data(final_data)
            
            # Cache the result
            self.cache[cache_key] = {
                'data': final_data,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Successfully integrated data for {ticker}: {final_data.shape[0]} rows, {final_data.shape[1]} columns")
            return final_data
            
        except Exception as e:
            logger.error(f"Error integrating data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the dataset.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            df = data.copy()
            
            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                sma_data = self.technical_indicators.calculate_sma(df, period)
                ema_data = self.technical_indicators.calculate_ema(df, period)
                
                if f'sma_{period}' in sma_data.columns:
                    df[f'sma_{period}'] = sma_data[f'sma_{period}']
                if f'ema_{period}' in ema_data.columns:
                    df[f'ema_{period}'] = ema_data[f'ema_{period}']
            
            # RSI
            rsi_data = self.technical_indicators.calculate_rsi(df)
            if 'rsi' in rsi_data.columns:
                df['rsi'] = rsi_data['rsi']
            
            # MACD
            macd_data = self.technical_indicators.calculate_macd(df)
            for col in ['macd', 'macd_signal', 'macd_histogram']:
                if col in macd_data.columns:
                    df[col] = macd_data[col]
            
            # Bollinger Bands
            bb_data = self.technical_indicators.calculate_bollinger_bands(df)
            for col in ['bb_upper', 'bb_middle', 'bb_lower']:
                if col in bb_data.columns:
                    df[col] = bb_data[col]
            
            # Stochastic Oscillator
            stoch_data = self.technical_indicators.calculate_stochastic(df)
            for col in ['stoch_k', 'stoch_d']:
                if col in stoch_data.columns:
                    df[col] = stoch_data[col]
            
            # Williams %R
            williams_data = self.technical_indicators.calculate_williams_r(df)
            if 'williams_r' in williams_data.columns:
                df['williams_r'] = williams_data['williams_r']
            
            # Average True Range (ATR)
            atr_data = self.technical_indicators.calculate_atr(df)
            if 'atr' in atr_data.columns:
                df['atr'] = atr_data['atr']
            
            # Commodity Channel Index (CCI)
            cci_data = self.technical_indicators.calculate_cci(df)
            if 'cci' in cci_data.columns:
                df['cci'] = cci_data['cci']
            
            # Volume indicators
            obv_data = self.technical_indicators.calculate_obv(df)
            if 'obv' in obv_data.columns:
                df['obv'] = obv_data['obv']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _add_fundamental_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add fundamental analysis features to the dataset.
        
        Args:
            data: DataFrame with OHLCV and technical data
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with fundamental features
        """
        try:
            df = data.copy()
            
            # Get fundamental features
            fundamental_features = self.fundamental_analyzer.get_fundamental_features(ticker)
            
            if fundamental_features:
                # Add fundamental features as constant columns
                for feature_name, feature_value in fundamental_features.items():
                    df[f'fundamental_{feature_name}'] = feature_value
                
                logger.info(f"Added {len(fundamental_features)} fundamental features")
            else:
                logger.warning("No fundamental features available")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding fundamental features: {e}")
            return data
    
    def _add_sentiment_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add sentiment analysis features to the dataset.
        
        Args:
            data: DataFrame with existing features
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with sentiment features
        """
        try:
            df = data.copy()
            
            # Get recent news sentiment
            try:
                sentiment_data = self.news_sentiment.get_sentiment_analysis(ticker)
                
                if sentiment_data:
                    # Add sentiment scores
                    df['news_sentiment'] = sentiment_data.get('overall_sentiment', 0.0)
                    df['sentiment_score'] = sentiment_data.get('sentiment_score', 0.0)
                    df['sentiment_magnitude'] = sentiment_data.get('sentiment_magnitude', 0.0)
                    df['positive_ratio'] = sentiment_data.get('positive_ratio', 0.0)
                    df['negative_ratio'] = sentiment_data.get('negative_ratio', 0.0)
                    df['neutral_ratio'] = sentiment_data.get('neutral_ratio', 0.0)
                    
                    logger.info("Added sentiment analysis features")
                else:
                    # Add default sentiment values
                    df['news_sentiment'] = 0.0
                    df['sentiment_score'] = 0.0
                    df['sentiment_magnitude'] = 0.0
                    df['positive_ratio'] = 0.33
                    df['negative_ratio'] = 0.33
                    df['neutral_ratio'] = 0.34
                    
                    logger.warning("No sentiment data available, using defaults")
                    
            except Exception as e:
                logger.warning(f"Error fetching sentiment data: {e}")
                # Add default sentiment values
                df['news_sentiment'] = 0.0
                df['sentiment_score'] = 0.0
                df['sentiment_magnitude'] = 0.0
                df['positive_ratio'] = 0.33
                df['negative_ratio'] = 0.33
                df['neutral_ratio'] = 0.34
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
            return data
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features based on existing data.
        
        Args:
            data: DataFrame with existing features
            
        Returns:
            DataFrame with derived features
        """
        try:
            df = data.copy()
            
            # Price momentum features
            df['price_momentum_1d'] = df['close'].pct_change(1)
            df['price_momentum_3d'] = df['close'].pct_change(3)
            df['price_momentum_7d'] = df['close'].pct_change(7)
            df['price_momentum_14d'] = df['close'].pct_change(14)
            df['price_momentum_30d'] = df['close'].pct_change(30)
            
            # Volatility features
            for window in [5, 10, 20, 30]:
                df[f'volatility_{window}d'] = df['close'].rolling(window).std()
                df[f'volatility_ratio_{window}d'] = df[f'volatility_{window}d'] / df['close']
            
            # Volume features
            df['volume_momentum_1d'] = df['volume'].pct_change(1)
            df['volume_momentum_7d'] = df['volume'].pct_change(7)
            
            for window in [5, 10, 20]:
                df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
            
            # Price-Volume relationship
            df['price_volume_trend'] = df['price_momentum_1d'] * df['volume_momentum_1d']
            
            # VWAP (Volume Weighted Average Price)
            for window in [5, 10, 20]:
                df[f'vwap_{window}'] = (
                    (df['close'] * df['volume']).rolling(window).sum() / 
                    df['volume'].rolling(window).sum()
                )
                df[f'price_vwap_ratio_{window}'] = df['close'] / df[f'vwap_{window}']
            
            # High-Low features
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Gap features
            df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['gap_filled'] = (
                ((df['low'] <= df['close'].shift(1)) & (df['gap'] > 0)) |
                ((df['high'] >= df['close'].shift(1)) & (df['gap'] < 0))
            ).astype(int)
            
            # Moving average crossovers
            if 'sma_5' in df.columns and 'sma_20' in df.columns:
                df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
                df['sma_5_20_distance'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
            
            if 'ema_12' in df.columns and 'ema_26' in df.columns:
                df['ema_12_26_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
                df['ema_12_26_distance'] = (df['ema_12'] - df['ema_26']) / df['ema_26']
            
            # RSI-based features
            if 'rsi' in df.columns:
                df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
                df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
                df['rsi_momentum'] = df['rsi'].diff()
                df['rsi_divergence'] = (
                    (df['close'] > df['close'].shift(1)) & 
                    (df['rsi'] < df['rsi'].shift(1))
                ).astype(int)
            
            # MACD-based features
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
                df['macd_momentum'] = df['macd'].diff()
            
            # Bollinger Bands features
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                df['bb_position'] = (
                    (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                )
                df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_breakout_upper'] = (df['close'] > df['bb_upper']).astype(int)
                df['bb_breakout_lower'] = (df['close'] < df['bb_lower']).astype(int)
            
            # Time-based features
            if hasattr(df.index, 'dayofweek'):
                df['day_of_week'] = df.index.dayofweek
                df['month'] = df.index.month
                df['quarter'] = df.index.quarter
                df['is_month_end'] = df.index.is_month_end.astype(int)
                df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
                df['is_year_end'] = df.index.is_year_end.astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding derived features: {e}")
            return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the integrated dataset.
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df = data.copy()
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill missing values for fundamental features (they should be constant)
            fundamental_cols = [col for col in df.columns if col.startswith('fundamental_')]
            if fundamental_cols:
                df[fundamental_cols] = df[fundamental_cols].fillna(method='ffill')
                df[fundamental_cols] = df[fundamental_cols].fillna(method='bfill')
            
            # Forward fill missing values for sentiment features
            sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
            if sentiment_cols:
                df[sentiment_cols] = df[sentiment_cols].fillna(method='ffill')
                df[sentiment_cols] = df[sentiment_cols].fillna(method='bfill')
            
            # Fill remaining NaN values with 0 for derived features
            derived_feature_patterns = [
                'momentum', 'volatility', 'ratio', 'cross', 'distance', 
                'position', 'squeeze', 'breakout', 'divergence'
            ]
            
            for pattern in derived_feature_patterns:
                pattern_cols = [col for col in df.columns if pattern in col.lower()]
                if pattern_cols:
                    df[pattern_cols] = df[pattern_cols].fillna(0)
            
            # Remove columns with too many missing values (>50%)
            missing_threshold = 0.5
            missing_ratio = df.isnull().sum() / len(df)
            cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
            
            if cols_to_drop:
                logger.warning(f"Dropping columns with >50% missing values: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)
            
            # Ensure we have minimum required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns after cleaning: {required_cols}")
                return pd.DataFrame()
            
            logger.info(f"Data cleaning completed: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return data
    
    def get_prediction_features(self, 
                              ticker: str, 
                              timeframe: str = '1d',
                              lookback_period: str = '1y') -> pd.DataFrame:
        """
        Get features specifically prepared for prediction models.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            lookback_period: Historical period for training data
            
        Returns:
            DataFrame ready for ML model training/prediction
        """
        try:
            # Get comprehensive data
            data = self.get_comprehensive_data(
                ticker=ticker,
                timeframe=timeframe,
                period=lookback_period,
                include_sentiment=True,
                include_fundamental=True
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # Create target variables for different prediction horizons
            for horizon in [1, 3, 7, 14, 30]:
                data[f'target_return_{horizon}d'] = (
                    data['close'].shift(-horizon) / data['close'] - 1
                )
                data[f'target_price_{horizon}d'] = data['close'].shift(-horizon)
            
            # Remove rows where we don't have future data for targets
            data = data.dropna(subset=[col for col in data.columns if col.startswith('target_')])
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            return pd.DataFrame()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self.cache_duration
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        logger.info("Data cache cleared")
    
    def get_cache_info(self) -> Dict:
        """
        Get information about the current cache state.
        
        Returns:
            Dictionary with cache information
        """
        return {
            'cache_size': len(self.cache),
            'cache_keys': list(self.cache.keys()),
            'cache_duration_minutes': self.cache_duration.total_seconds() / 60
        }