import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.ML_models.model_factory import ModelFactory
from backend.core.data_fetcher import fetch_historical_data, calculate_technical_indicators
from backend.core.news_processor import fetch_news, analyze_news_sentiment
from backend.utils.config import TECHNICAL_INDICATORS, PREDICTION_WINDOWS
from backend.utils.helpers import validate_ticker, parse_date, validate_date_range

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'model_training.log'))
    ]
)

logger = logging.getLogger(__name__)

def prepare_training_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Prepare training data for the models.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for historical data
        end_date: End date for historical data
        
    Returns:
        DataFrame with historical data, technical indicators, and news sentiment
    """
    logger.info(f"Preparing training data for {ticker} from {start_date} to {end_date}")
    
    # Validate ticker
    if not validate_ticker(ticker):
        logger.error(f"Invalid ticker: {ticker}")
        return pd.DataFrame()
    
    # Validate date range
    if not validate_date_range(start_date, end_date):
        logger.error(f"Invalid date range: {start_date} to {end_date}")
        return pd.DataFrame()
    
    # Fetch historical data
    historical_data = fetch_historical_data(ticker, start_date, end_date)
    
    if historical_data.empty:
        logger.error(f"No historical data found for {ticker}")
        return pd.DataFrame()
    
    # Calculate technical indicators
    data_with_indicators = calculate_technical_indicators(historical_data)
    
    # Fetch news data
    news_data = fetch_news(ticker, start_date, end_date)
    
    # Analyze news sentiment
    if not news_data.empty:
        sentiment_model = ModelFactory.get_sentiment_model()
        
        # Group news by date
        news_data['date'] = pd.to_datetime(news_data['date']).dt.date
        grouped_news = news_data.groupby('date')
        
        # Calculate daily sentiment scores
        daily_sentiment = {}
        for date, group in grouped_news:
            sentiment_results = []
            for _, row in group.iterrows():
                text = f"{row['title']} {row.get('content', '')}"
                sentiment_result = sentiment_model.analyze_sentiment(text)
                sentiment_results.append(sentiment_result['sentiment_score'])
            
            daily_sentiment[date] = np.mean(sentiment_results) if sentiment_results else 0.0
        
        # Add sentiment scores to the data
        data_with_indicators['news_sentiment'] = data_with_indicators.index.date.map(daily_sentiment).fillna(0)
    else:
        # Add empty sentiment column
        data_with_indicators['news_sentiment'] = 0.0
    
    logger.info(f"Prepared training data with {len(data_with_indicators)} rows and {len(data_with_indicators.columns)} columns")
    
    return data_with_indicators

def create_target_variables(data: pd.DataFrame, prediction_windows: Dict[str, int]) -> pd.DataFrame:
    """
    Create target variables for different prediction windows.
    
    Args:
        data: DataFrame with historical data and features
        prediction_windows: Dictionary mapping timeframe names to number of days
        
    Returns:
        DataFrame with target variables for different prediction windows
    """
    df = data.copy()
    
    # Create target variables for different prediction windows
    for timeframe, days in prediction_windows.items():
        # Future close price
        df[f'future_close_{timeframe}'] = df['close'].shift(-days)
        
        # Price change
        df[f'price_change_{timeframe}'] = df[f'future_close_{timeframe}'] - df['close']
        
        # Percentage change
        df[f'pct_change_{timeframe}'] = df[f'price_change_{timeframe}'] / df['close'] * 100
        
        # Direction (1 for up, 0 for down)
        df[f'direction_{timeframe}'] = (df[f'price_change_{timeframe}'] > 0).astype(int)
    
    return df

def train_models(ticker: str, start_date: str, end_date: str, models: List[str], timeframes: List[str]) -> None:
    """
    Train models for the specified ticker, timeframes, and model types.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for training data
        end_date: End date for training data
        models: List of model types to train ('xgboost', 'informer', 'dqn')
        timeframes: List of timeframes to train for ('intraday', 'short_term', 'medium_term', 'long_term')
    """
    logger.info(f"Training models for {ticker}: {models} for timeframes {timeframes}")
    
    # Prepare training data
    data = prepare_training_data(ticker, start_date, end_date)
    
    if data.empty:
        logger.error(f"No data available for training models for {ticker}")
        return
    
    # Create target variables
    prediction_windows = {timeframe: PREDICTION_WINDOWS[timeframe] for timeframe in timeframes}
    data_with_targets = create_target_variables(data, prediction_windows)
    
    # Drop rows with NaN target values
    data_with_targets = data_with_targets.dropna()
    
    if data_with_targets.empty:
        logger.error(f"No valid data with targets for training models for {ticker}")
        return
    
    # Define feature columns (excluding target variables and date-related columns)
    exclude_cols = ['open', 'high', 'low', 'volume', 'dividends', 'stock_splits']
    exclude_cols.extend([col for col in data_with_targets.columns if col.startswith('future_') or 
                        col.startswith('price_change_') or col.startswith('pct_change_') or 
                        col.startswith('direction_')])
    
    feature_columns = [col for col in data_with_targets.columns if col not in exclude_cols]
    
    logger.info(f"Using {len(feature_columns)} features for training: {feature_columns}")
    
    # Train models for each timeframe and model type
    for timeframe in timeframes:
        for model_type in models:
            logger.info(f"Training {model_type} model for {ticker} with timeframe {timeframe}")
            
            try:
                if model_type.lower() == 'xgboost':
                    model = ModelFactory.get_price_prediction_model('xgboost', ticker, timeframe)
                    model.target_column = f'future_close_{timeframe}'
                    model.train(data_with_targets, feature_columns)
                    
                elif model_type.lower() == 'informer':
                    model = ModelFactory.get_price_prediction_model('informer', ticker, timeframe)
                    model.target_column = f'future_close_{timeframe}'
                    model.train(data_with_targets, feature_columns)
                    
                elif model_type.lower() == 'dqn':
                    model = ModelFactory.get_trading_model(ticker, timeframe)
                    model.train(data_with_targets, feature_columns)
                    
                else:
                    logger.error(f"Invalid model type: {model_type}")
                    continue
                    
                logger.info(f"Successfully trained {model_type} model for {ticker} with timeframe {timeframe}")
                
            except Exception as e:
                logger.error(f"Error training {model_type} model for {ticker} with timeframe {timeframe}: {str(e)}")

def main():
    """
    Main function to parse arguments and train models.
    """
    parser = argparse.ArgumentParser(description='Train stock prediction models')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2018-01-01', help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--models', type=str, nargs='+', default=['xgboost', 'informer', 'dqn'], help='Model types to train')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['intraday', 'short_term', 'medium_term', 'long_term'], help='Timeframes to train for')
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Train models
    train_models(args.ticker, args.start_date, args.end_date, args.models, args.timeframes)

if __name__ == '__main__':
    main()