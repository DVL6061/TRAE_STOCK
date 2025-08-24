import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime, timedelta

from backend.utils.config import MODEL_PARAMS

logger = logging.getLogger(__name__)

class XGBoostModel:
    """
    XGBoost model for stock price prediction.
    """
    def __init__(self, ticker: str, timeframe: str):
        """
        Initialize the XGBoost model.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Prediction timeframe (e.g., 'intraday', 'short_term', 'medium_term', 'long_term')
        """
        self.ticker = ticker
        self.timeframe = timeframe
        self.model = None
        self.feature_columns = []
        self.target_column = 'close'
        self.model_params = MODEL_PARAMS['xgboost'][timeframe]
        self.model_path = os.path.join('models', f'xgboost_{ticker}_{timeframe}.joblib')
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training or prediction.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            
        Returns:
            Tuple of features (X) and target (y) arrays
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Handle missing values
        df = df.dropna()
        
        # Extract features and target
        X = df[self.feature_columns].values
        y = df[self.target_column].values if self.target_column in df.columns else None
        
        return X, y
    
    def train(self, data: pd.DataFrame, feature_columns: List[str]) -> None:
        """
        Train the XGBoost model.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            feature_columns: List of column names to use as features
        """
        self.feature_columns = feature_columns
        X, y = self.preprocess_data(data)
        
        if X.shape[0] == 0 or y is None or len(y) == 0:
            logger.error(f"No valid data for training XGBoost model for {self.ticker}")
            return
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train the model
        logger.info(f"Training XGBoost model for {self.ticker} with {X.shape[0]} samples")
        self.model = xgb.train(self.model_params, dtrain)
        
        # Save the model
        self.save_model()
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            
        Returns:
            Array of predicted values
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                logger.error(f"No trained model found for {self.ticker}")
                return np.array([])
        
        X, _ = self.preprocess_data(data)
        
        if X.shape[0] == 0:
            logger.error(f"No valid data for prediction with XGBoost model for {self.ticker}")
            return np.array([])
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X)
        
        # Make predictions
        predictions = self.model.predict(dtest)
        
        return predictions
    
    def save_model(self) -> None:
        """
        Save the trained model to disk.
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'timeframe': self.timeframe,
                'ticker': self.ticker
            }, self.model_path)
            logger.info(f"Saved XGBoost model to {self.model_path}")
    
    def load_model(self) -> None:
        """
        Load a trained model from disk.
        """
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            logger.info(f"Loaded XGBoost model from {self.model_path}")
        else:
            logger.error(f"No model file found at {self.model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                logger.error(f"No trained model found for {self.ticker}")
                return {}
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        
        # Map feature indices to feature names
        feature_importance = {}
        for feature, score in importance.items():
            feature_idx = int(feature.replace('f', ''))
            if feature_idx < len(self.feature_columns):
                feature_importance[self.feature_columns[feature_idx]] = score
        
        return feature_importance