import os
import logging
from typing import Dict, List, Union, Optional, Any

from backend.ML_models.xgboost_model import XGBoostModel
from backend.ML_models.informer_model import InformerWrapper
from backend.ML_models.dqn_model import DQNAgent
from backend.ML_models.sentiment_model import SentimentModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating and managing different models.
    """
    def __init__(self):
        """
        Initialize the model factory.
        """
        self.models = {}
        self.sentiment_model = None
    
    def get_price_prediction_model(self, model_type: str, ticker: str, timeframe: str) -> Union[XGBoostModel, InformerWrapper, None]:
        """
        Get a price prediction model instance.
        
        Args:
            model_type: Type of model ('xgboost' or 'informer')
            ticker: Stock ticker symbol
            timeframe: Prediction timeframe
            
        Returns:
            Model instance or None if model type is invalid
        """
        model_key = f"{model_type}_{ticker}_{timeframe}"
        
        # Return existing model if already created
        if model_key in self.models:
            return self.models[model_key]
        
        # Create new model
        if model_type.lower() == 'xgboost':
            model = XGBoostModel(ticker, timeframe)
        elif model_type.lower() == 'informer':
            model = InformerWrapper(ticker, timeframe)
        else:
            logger.error(f"Invalid price prediction model type: {model_type}")
            return None
        
        # Store model in cache
        self.models[model_key] = model
        
        return model
    
    def get_trading_model(self, ticker: str, timeframe: str) -> DQNAgent:
        """
        Get a trading model (DQN) instance.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Trading timeframe
            
        Returns:
            DQN agent instance
        """
        model_key = f"dqn_{ticker}_{timeframe}"
        
        # Return existing model if already created
        if model_key in self.models:
            return self.models[model_key]
        
        # Create new model
        model = DQNAgent(ticker, timeframe)
        
        # Store model in cache
        self.models[model_key] = model
        
        return model
    
    def get_sentiment_model(self) -> SentimentModel:
        """
        Get the sentiment analysis model instance.
        
        Returns:
            Sentiment model instance
        """
        # Return existing model if already created
        if self.sentiment_model is not None:
            return self.sentiment_model
        
        # Create new model
        self.sentiment_model = SentimentModel()
        
        return self.sentiment_model
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get a list of available model types and their supported timeframes.
        
        Returns:
            Dictionary mapping model types to supported timeframes
        """
        return {
            'xgboost': ['intraday', 'short_term', 'medium_term', 'long_term'],
            'informer': ['intraday', 'short_term', 'medium_term', 'long_term'],
            'dqn': ['intraday', 'short_term', 'medium_term', 'long_term']
        }
    
    def get_model_info(self, model_type: str, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_type: Type of model
            ticker: Stock ticker symbol
            timeframe: Model timeframe
            
        Returns:
            Dictionary with model information
        """
        model_key = f"{model_type}_{ticker}_{timeframe}"
        model_path = os.path.join('models', f"{model_key}.{'pt' if model_type == 'informer' or model_type == 'dqn' else 'joblib'}")
        
        # Check if model exists
        model_exists = os.path.exists(model_path)
        
        # Get model instance
        model = None
        if model_type.lower() == 'xgboost':
            model = self.get_price_prediction_model('xgboost', ticker, timeframe)
        elif model_type.lower() == 'informer':
            model = self.get_price_prediction_model('informer', ticker, timeframe)
        elif model_type.lower() == 'dqn':
            model = self.get_trading_model(ticker, timeframe)
        
        # Get feature importance for XGBoost
        feature_importance = {}
        if model_type.lower() == 'xgboost' and model_exists:
            try:
                model.load_model()
                feature_importance = model.get_feature_importance()
            except Exception as e:
                logger.error(f"Error getting feature importance: {str(e)}")
        
        return {
            'model_type': model_type,
            'ticker': ticker,
            'timeframe': timeframe,
            'model_exists': model_exists,
            'model_path': model_path,
            'feature_importance': feature_importance
        }
    
    def clear_model_cache(self) -> None:
        """
        Clear the model cache to free up memory.
        """
        self.models = {}
        self.sentiment_model = None
        logger.info("Model cache cleared")

# Create a singleton instance
model_factory = ModelFactory()