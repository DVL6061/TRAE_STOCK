import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from backend.utils.config import MODEL_PARAMS
from backend.core.fundamental_analyzer import FundamentalAnalyzer

logger = logging.getLogger(__name__)

class XGBoostModel:
    """
    Advanced XGBoost model for stock price prediction with feature engineering and SHAP explainability.
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
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'close'
        self.model_params = MODEL_PARAMS['xgboost'][timeframe]
        self.model_path = os.path.join('models', f'xgboost_{ticker}_{timeframe}.joblib')
        self.explainer = None
        self.feature_importance_dict = {}
        self.model_metrics = {}
        self.fundamental_analyzer = FundamentalAnalyzer()
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features for stock prediction.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_2d'] = df['close'].pct_change(2)
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_volatility_5d'] = df['price_change'].rolling(5).std()
        df['price_volatility_10d'] = df['price_change'].rolling(10).std()
        
        # Volume-based features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_10']
        
        # Price-Volume relationship
        df['price_volume_trend'] = df['price_change'] * df['volume_change']
        df['vwap'] = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
        df['price_vwap_ratio'] = df['close'] / df['vwap']
        
        # High-Low spread features
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['hl_spread_ma'] = df['hl_spread'].rolling(5).mean()
        df['hl_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap features
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_filled'] = ((df['low'] <= df['close'].shift(1)) & (df['gap'] > 0)).astype(int)
        
        # Momentum features
        df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
        df['momentum_7d'] = df['close'] / df['close'].shift(7) - 1
        df['momentum_14d'] = df['close'] / df['close'].shift(14) - 1
        
        # Moving average crossovers
        if 'sma_5' in df.columns and 'sma_20' in df.columns:
            df['sma_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
            df['sma_distance'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
        
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['ema_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
            df['ema_distance'] = (df['ema_12'] - df['ema_26']) / df['ema_26']
        
        # RSI-based features
        if 'rsi' in df.columns:
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            df['rsi_momentum'] = df['rsi'].diff()
        
        # MACD-based features
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            df['macd_momentum'] = df['macd'].diff()
        
        # Bollinger Bands features
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        
        # Time-based features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        # Sentiment features (if available)
        if 'news_sentiment' in df.columns:
            df['sentiment_ma_3'] = df['news_sentiment'].rolling(3).mean()
            df['sentiment_ma_7'] = df['news_sentiment'].rolling(7).mean()
            df['sentiment_change'] = df['news_sentiment'].diff()
            df['sentiment_volatility'] = df['news_sentiment'].rolling(5).std()
        
        # Lag features for target variable
        for lag in [1, 2, 3, 5, 7]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'return_lag_{lag}'] = df['price_change'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window).max()
        
        # Fundamental analysis features
        try:
            fundamental_features = self.fundamental_analyzer.get_fundamental_features(self.ticker)
            if fundamental_features:
                # Add fundamental features as constant values across all rows
                for feature_name, feature_value in fundamental_features.items():
                    df[f'fundamental_{feature_name}'] = feature_value
                
                logger.info(f"Added {len(fundamental_features)} fundamental features for {self.ticker}")
            else:
                logger.warning(f"No fundamental features available for {self.ticker}")
                
        except Exception as e:
            logger.warning(f"Error adding fundamental features for {self.ticker}: {e}")
        
        return df
        
    def select_features(self, data: pd.DataFrame) -> List[str]:
        """
        Automatically select relevant features for the model.
        
        Args:
            data: DataFrame with engineered features
            
        Returns:
            List of selected feature column names
        """
        # Exclude non-feature columns
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        
        # Get all numeric columns except target and excluded columns
        feature_cols = [col for col in data.columns 
                       if col not in exclude_cols and 
                       data[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        # Remove columns with too many NaN values (>50%)
        feature_cols = [col for col in feature_cols 
                       if data[col].notna().sum() / len(data) > 0.5]
        
        return feature_cols
    
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training or prediction with advanced feature engineering.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            is_training: Whether this is for training (True) or prediction (False)
            
        Returns:
            Tuple of features (X) and target (y) arrays
        """
        # Engineer features
        df = self.engineer_features(data)
        
        # Select features if not already done
        if not self.feature_columns:
            self.feature_columns = self.select_features(df)
            logger.info(f"Selected {len(self.feature_columns)} features for {self.ticker}")
        
        # Handle missing values
        df = df.dropna(subset=self.feature_columns + [self.target_column] if self.target_column in df.columns else self.feature_columns)
        
        if len(df) == 0:
            logger.warning(f"No valid data after preprocessing for {self.ticker}")
            return np.array([]), np.array([])
        
        # Extract features
        X = df[self.feature_columns].values
        
        # Scale features
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        # Extract target
        y = df[self.target_column].values if self.target_column in df.columns else None
        
        # Handle infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        if y is not None:
            y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X, y
    
    def train(self, data: pd.DataFrame, feature_columns: List[str] = None, 
              tune_hyperparameters: bool = True) -> None:
        """
        Train the XGBoost model with advanced features and hyperparameter tuning.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            feature_columns: List of column names to use as features (optional, auto-selected if None)
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        if feature_columns:
            self.feature_columns = feature_columns
            
        X, y = self.preprocess_data(data, is_training=True)
        
        if X.shape[0] == 0 or y is None or len(y) == 0:
            logger.error(f"No valid data for training XGBoost model for {self.ticker}")
            return
        
        logger.info(f"Training XGBoost model for {self.ticker} with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Hyperparameter tuning
        if tune_hyperparameters and X_train.shape[0] > 100:
            logger.info(f"Performing hyperparameter tuning for {self.ticker}")
            best_params = self._tune_hyperparameters(X_train, y_train)
            self.model_params.update(best_params)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_columns)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_columns)
        
        # Set up early stopping
        evals = [(dtrain, 'train'), (dval, 'eval')]
        
        # Train the model
        self.model = xgb.train(
            self.model_params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Calculate model metrics
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        
        self.model_metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        logger.info(f"Model training completed. Validation R²: {self.model_metrics['val_r2']:.4f}, RMSE: {self.model_metrics['val_rmse']:.4f}")
        
        # Initialize SHAP explainer
        try:
            sample_size = min(100, X_train.shape[0])
            self.explainer = shap.TreeExplainer(self.model)
            logger.info(f"SHAP explainer initialized for {self.ticker}")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
        
        # Save the model
        self.save_model()
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of best hyperparameters
        """
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Use a subset for faster tuning
        if X.shape[0] > 1000:
            indices = np.random.choice(X.shape[0], 1000, replace=False)
            X_tune, y_tune = X[indices], y[indices]
        else:
            X_tune, y_tune = X, y
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_tune, y_tune)
        
        return grid_search.best_params_
        
    def predict(self, data: pd.DataFrame) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Make predictions using the trained model with explainability.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            
        Returns:
            Dictionary containing predictions and explanations
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                logger.error(f"No trained model found for {self.ticker}")
                return {'predictions': np.array([]), 'explanations': {}}
        
        X, _ = self.preprocess_data(data, is_training=False)
        
        if X.shape[0] == 0:
            logger.error(f"No valid data for prediction with XGBoost model for {self.ticker}")
            return {'predictions': np.array([]), 'explanations': {}}
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X, feature_names=self.feature_columns)
        
        # Make predictions
        predictions = self.model.predict(dtest)
        
        # Generate SHAP explanations
        explanations = self.get_shap_explanations(X)
        
        return {
            'predictions': predictions,
            'explanations': explanations,
            'feature_importance': self.get_feature_importance(),
            'model_metrics': self.model_metrics
        }
    
    def get_shap_explanations(self, X: np.ndarray, max_samples: int = 10) -> Dict:
        """
        Generate SHAP explanations for predictions.
        
        Args:
            X: Feature matrix
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        if self.explainer is None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                logger.warning(f"Failed to create SHAP explainer: {e}")
                return {}
        
        try:
            # Limit samples for performance
            sample_X = X[:max_samples] if X.shape[0] > max_samples else X
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(sample_X)
            
            # Get feature contributions for the latest prediction
            if len(shap_values) > 0:
                latest_shap = shap_values[-1] if len(shap_values.shape) == 1 else shap_values[-1, :]
                
                # Create feature contribution dictionary
                feature_contributions = {
                    self.feature_columns[i]: float(latest_shap[i]) 
                    for i in range(min(len(self.feature_columns), len(latest_shap)))
                }
                
                # Sort by absolute contribution
                sorted_contributions = dict(sorted(
                    feature_contributions.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                ))
                
                return {
                    'feature_contributions': sorted_contributions,
                    'top_positive_features': {k: v for k, v in sorted_contributions.items() if v > 0}[:5],
                    'top_negative_features': {k: v for k, v in sorted_contributions.items() if v < 0}[:5],
                    'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
                }
            
        except Exception as e:
            logger.warning(f"Failed to generate SHAP explanations: {e}")
        
        return {}
    
    def predict_with_confidence(self, data: pd.DataFrame, n_estimators_list: List[int] = None) -> Dict:
        """
        Make predictions with confidence intervals using different numbers of estimators.
        
        Args:
            data: DataFrame containing stock data
            n_estimators_list: List of estimator counts to use for confidence estimation
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if n_estimators_list is None:
            n_estimators_list = [50, 100, 200, 300]
        
        X, _ = self.preprocess_data(data, is_training=False)
        
        if X.shape[0] == 0:
            return {'predictions': np.array([]), 'confidence_intervals': {}}
        
        predictions_list = []
        
        for n_est in n_estimators_list:
            if self.model.num_boosted_rounds() >= n_est:
                dtest = xgb.DMatrix(X, feature_names=self.feature_columns)
                pred = self.model.predict(dtest, iteration_range=(0, n_est))
                predictions_list.append(pred)
        
        if predictions_list:
            predictions_array = np.array(predictions_list)
            mean_pred = np.mean(predictions_array, axis=0)
            std_pred = np.std(predictions_array, axis=0)
            
            # Calculate confidence intervals (95%)
            confidence_intervals = {
                'lower_95': mean_pred - 1.96 * std_pred,
                'upper_95': mean_pred + 1.96 * std_pred,
                'lower_68': mean_pred - std_pred,
                'upper_68': mean_pred + std_pred,
                'std': std_pred
            }
            
            return {
                'predictions': mean_pred,
                'confidence_intervals': confidence_intervals
            }
        
        # Fallback to regular prediction
        result = self.predict(data)
        return {
            'predictions': result['predictions'],
            'confidence_intervals': {}
        }
    
    def save_model(self) -> None:
        """
        Save the trained model and all associated data to disk.
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'timeframe': self.timeframe,
                'ticker': self.ticker,
                'model_params': self.model_params,
                'model_metrics': self.model_metrics,
                'feature_importance_dict': self.feature_importance_dict,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Saved XGBoost model to {self.model_path} with metrics: {self.model_metrics}")
    
    def load_model(self) -> None:
        """
        Load a trained model and all associated data from disk.
        """
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.feature_columns = model_data.get('feature_columns', [])
                self.target_column = model_data.get('target_column', 'close')
                
                # Load additional attributes if available
                if 'scaler' in model_data:
                    self.scaler = model_data['scaler']
                if 'model_params' in model_data:
                    self.model_params = model_data['model_params']
                if 'model_metrics' in model_data:
                    self.model_metrics = model_data['model_metrics']
                if 'feature_importance_dict' in model_data:
                    self.feature_importance_dict = model_data['feature_importance_dict']
                
                # Initialize SHAP explainer
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception as e:
                    logger.warning(f"Failed to initialize SHAP explainer after loading: {e}")
                
                logger.info(f"Loaded XGBoost model from {self.model_path}")
                
            except Exception as e:
                logger.error(f"Failed to load model from {self.model_path}: {e}")
        else:
            logger.error(f"No model file found at {self.model_path}")
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                logger.error(f"No trained model found for {self.ticker}")
                return {}
        
        try:
            # Get feature importance
            importance = self.model.get_score(importance_type=importance_type)
            
            # Map feature indices to feature names
            feature_importance = {}
            for feature, score in importance.items():
                feature_idx = int(feature.replace('f', ''))
                if feature_idx < len(self.feature_columns):
                    feature_importance[self.feature_columns[feature_idx]] = float(score)
            
            # Sort by importance
            sorted_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            self.feature_importance_dict = sorted_importance
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def get_model_summary(self) -> Dict:
        """
        Get a comprehensive summary of the model.
        
        Returns:
            Dictionary containing model summary information
        """
        summary = {
            'ticker': self.ticker,
            'timeframe': self.timeframe,
            'model_type': 'XGBoost',
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'model_params': self.model_params,
            'model_metrics': self.model_metrics,
            'model_path': self.model_path,
            'has_explainer': self.explainer is not None
        }
        
        if self.model is not None:
            summary.update({
                'n_estimators': self.model.num_boosted_rounds(),
                'feature_importance': self.get_feature_importance()
            })
        
        return summary
    
    def generate_trading_signal(self, current_price: float, predicted_price: float, 
                              confidence_interval: Dict = None) -> Dict:
        """
        Generate trading signals based on predictions.
        
        Args:
            current_price: Current stock price
            predicted_price: Predicted stock price
            confidence_interval: Confidence interval for prediction
            
        Returns:
            Dictionary containing trading signal and reasoning
        """
        price_change_pct = (predicted_price - current_price) / current_price * 100
        
        # Define thresholds based on timeframe
        thresholds = {
            'intraday': {'buy': 0.5, 'sell': -0.5},
            'short_term': {'buy': 1.0, 'sell': -1.0},
            'medium_term': {'buy': 2.0, 'sell': -2.0},
            'long_term': {'buy': 5.0, 'sell': -5.0}
        }
        
        threshold = thresholds.get(self.timeframe, thresholds['medium_term'])
        
        # Determine signal
        if price_change_pct >= threshold['buy']:
            signal = 'BUY'
            strength = min(price_change_pct / threshold['buy'], 3.0)  # Cap at 3x
        elif price_change_pct <= threshold['sell']:
            signal = 'SELL'
            strength = min(abs(price_change_pct) / abs(threshold['sell']), 3.0)
        else:
            signal = 'HOLD'
            strength = 1.0
        
        # Adjust strength based on confidence
        if confidence_interval and 'std' in confidence_interval:
            uncertainty = confidence_interval['std'][-1] if hasattr(confidence_interval['std'], '__getitem__') else confidence_interval['std']
            confidence_factor = max(0.1, 1.0 - (uncertainty / abs(predicted_price)) * 2)
            strength *= confidence_factor
        
        return {
            'signal': signal,
            'strength': round(strength, 2),
            'price_change_pct': round(price_change_pct, 2),
            'current_price': current_price,
            'predicted_price': round(predicted_price, 2),
            'timeframe': self.timeframe,
            'reasoning': f"Predicted {price_change_pct:.2f}% change over {self.timeframe} timeframe"
        }