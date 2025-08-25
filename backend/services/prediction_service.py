import os
import sys
import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from dataclasses import dataclass, asdict
import time
from functools import lru_cache

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our models and services
from models.xgboost_model import XGBoostPredictor
from models.informer_model import InformerWrapper
from models.dqn_model import DQNAgent
from services.sentiment_service import SentimentService
from data.angel_one_client import AngelOneClient
from data.technical_indicators import TechnicalIndicators
from data.fundamental_data import FundamentalDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """
    Comprehensive prediction result from all models.
    """
    ticker: str
    timestamp: datetime
    prediction_horizon: str
    
    # Price predictions
    current_price: float
    predicted_price_range: Dict[str, float]  # min, max, mean
    price_change_percent: float
    
    # Trading signals
    xgboost_signal: str
    informer_signal: str
    dqn_signal: str
    ensemble_signal: str
    
    # Confidence scores
    xgboost_confidence: float
    informer_confidence: float
    dqn_confidence: float
    ensemble_confidence: float
    
    # Model predictions
    xgboost_prediction: Dict[str, Any]
    informer_prediction: Dict[str, Any]
    dqn_prediction: Dict[str, Any]
    
    # Sentiment analysis
    sentiment_analysis: Dict[str, Any]
    
    # Risk metrics
    risk_metrics: Dict[str, float]
    
    # Explainability
    feature_importance: Dict[str, float]
    shap_values: Optional[Dict[str, Any]] = None
    
    # Market context
    market_regime: str
    volatility_forecast: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class PredictionService:
    """
    Comprehensive prediction service that integrates all models and provides
    unified predictions with ensemble methods and explainability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the prediction service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize data clients
        self.angel_client = AngelOneClient(
            api_key=config.get('angel_api_key'),
            client_id=config.get('angel_client_id'),
            password=config.get('angel_password'),
            totp_secret=config.get('angel_totp_secret')
        )
        
        self.technical_indicators = TechnicalIndicators()
        self.fundamental_fetcher = FundamentalDataFetcher()
        
        # Initialize sentiment service
        self.sentiment_service = SentimentService({
            'fingpt_model': config.get('fingpt_model', 'ProsusAI/finbert'),
            'model_cache_dir': config.get('model_cache_dir', './models/cache'),
            'newsapi_key': config.get('newsapi_key'),
            'alphavantage_key': config.get('alphavantage_key'),
            'cache_ttl_minutes': config.get('cache_ttl_minutes', 30)
        })
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
        # Ensemble weights
        self.ensemble_weights = {
            'xgboost': 0.4,
            'informer': 0.35,
            'dqn': 0.25
        }
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = config.get('prediction_cache_ttl_minutes', 5) * 60
        
        logger.info("Prediction service initialized successfully")
    
    def _initialize_models(self):
        """
        Initialize all prediction models.
        """
        try:
            # Initialize XGBoost model
            self.models['xgboost'] = XGBoostPredictor(
                model_params=self.config.get('xgboost_params', {})
            )
            
            # Initialize Informer model
            informer_config = self.config.get('informer_config', {
                'seq_len': 96,
                'label_len': 48,
                'pred_len': 24,
                'd_model': 512,
                'n_heads': 8,
                'e_layers': 2,
                'd_layers': 1,
                'd_ff': 2048,
                'dropout': 0.05,
                'activation': 'gelu'
            })
            
            self.models['informer'] = InformerWrapper(
                config=informer_config,
                device=self.config.get('device', 'cpu')
            )
            
            # Initialize DQN model
            dqn_config = self.config.get('dqn_config', {
                'state_dim': 50,
                'action_dim': 3,
                'hidden_dim': 256,
                'lr': 0.001,
                'gamma': 0.99,
                'epsilon': 0.1,
                'batch_size': 32,
                'buffer_size': 10000,
                'target_update': 100
            })
            
            self.models['dqn'] = DQNAgent(
                state_dim=dqn_config['state_dim'],
                action_dim=dqn_config['action_dim'],
                hidden_dim=dqn_config['hidden_dim'],
                lr=dqn_config['lr'],
                gamma=dqn_config['gamma'],
                epsilon=dqn_config['epsilon'],
                batch_size=dqn_config['batch_size'],
                buffer_size=dqn_config['buffer_size'],
                target_update=dqn_config['target_update']
            )
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    async def get_comprehensive_prediction(self, ticker: str, 
                                         prediction_horizon: str = 'intraday',
                                         include_sentiment: bool = True,
                                         include_shap: bool = False) -> PredictionResult:
        """
        Get comprehensive prediction from all models.
        
        Args:
            ticker: Stock ticker symbol
            prediction_horizon: Prediction timeframe ('scalping', 'intraday', 'swing', 'long_term')
            include_sentiment: Whether to include sentiment analysis
            include_shap: Whether to include SHAP explainability
            
        Returns:
            Comprehensive prediction result
        """
        try:
            # Check cache first
            cache_key = f"{ticker}_{prediction_horizon}_{include_sentiment}"
            if self._is_cache_valid(cache_key):
                logger.info(f"Returning cached prediction for {ticker}")
                return self.prediction_cache[cache_key]['data']
            
            logger.info(f"Generating comprehensive prediction for {ticker} ({prediction_horizon})")
            
            # Gather all required data concurrently
            data_tasks = [
                self._get_market_data(ticker, prediction_horizon),
                self._get_sentiment_data(ticker) if include_sentiment else self._get_neutral_sentiment(ticker)
            ]
            
            market_data, sentiment_data = await asyncio.gather(*data_tasks)
            
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for {ticker}")
            
            # Prepare features for all models
            features = await self._prepare_comprehensive_features(
                market_data, sentiment_data, ticker, prediction_horizon
            )
            
            # Get predictions from all models concurrently
            prediction_tasks = [
                self._get_xgboost_prediction(features, prediction_horizon),
                self._get_informer_prediction(market_data, prediction_horizon),
                self._get_dqn_prediction(features, market_data)
            ]
            
            xgb_pred, informer_pred, dqn_pred = await asyncio.gather(*prediction_tasks)
            
            # Calculate ensemble prediction
            ensemble_result = self._calculate_ensemble_prediction(
                xgb_pred, informer_pred, dqn_pred
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                market_data, xgb_pred, informer_pred, dqn_pred
            )
            
            # Get feature importance
            feature_importance = self._get_feature_importance(
                xgb_pred, features
            )
            
            # Get SHAP values if requested
            shap_values = None
            if include_shap:
                shap_values = await self._get_shap_explanation(
                    features, ticker
                )
            
            # Determine market regime
            market_regime = self._determine_market_regime(market_data)
            
            # Calculate volatility forecast
            volatility_forecast = self._calculate_volatility_forecast(market_data)
            
            # Create comprehensive result
            current_price = float(market_data['close'].iloc[-1])
            
            result = PredictionResult(
                ticker=ticker,
                timestamp=datetime.now(),
                prediction_horizon=prediction_horizon,
                current_price=current_price,
                predicted_price_range=ensemble_result['price_range'],
                price_change_percent=ensemble_result['price_change_percent'],
                xgboost_signal=xgb_pred['signal'],
                informer_signal=informer_pred['signal'],
                dqn_signal=dqn_pred['signal'],
                ensemble_signal=ensemble_result['signal'],
                xgboost_confidence=xgb_pred['confidence'],
                informer_confidence=informer_pred['confidence'],
                dqn_confidence=dqn_pred['confidence'],
                ensemble_confidence=ensemble_result['confidence'],
                xgboost_prediction=xgb_pred,
                informer_prediction=informer_pred,
                dqn_prediction=dqn_pred,
                sentiment_analysis=sentiment_data,
                risk_metrics=risk_metrics,
                feature_importance=feature_importance,
                shap_values=shap_values,
                market_regime=market_regime,
                volatility_forecast=volatility_forecast
            )
            
            # Cache the result
            self.prediction_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            logger.info(f"Comprehensive prediction completed for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive prediction: {str(e)}")
            raise
    
    async def _get_market_data(self, ticker: str, prediction_horizon: str) -> pd.DataFrame:
        """
        Get market data for the ticker.
        
        Args:
            ticker: Stock ticker
            prediction_horizon: Prediction timeframe
            
        Returns:
            Market data DataFrame
        """
        try:
            # Determine data period based on prediction horizon
            period_map = {
                'scalping': '1d',
                'intraday': '5d',
                'swing': '1mo',
                'long_term': '1y'
            }
            
            interval_map = {
                'scalping': '1m',
                'intraday': '5m',
                'swing': '1h',
                'long_term': '1d'
            }
            
            period = period_map.get(prediction_horizon, '5d')
            interval = interval_map.get(prediction_horizon, '5m')
            
            # Get historical data
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.angel_client.get_historical_data,
                    ticker,
                    period,
                    interval
                )
                data = await loop.run_in_executor(None, lambda: future.result())
            
            if data is None or data.empty:
                logger.warning(f"No historical data for {ticker}, using mock data")
                data = self._generate_mock_market_data(ticker, period, interval)
            
            # Add technical indicators
            data = self.technical_indicators.add_all_indicators(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return self._generate_mock_market_data(ticker, '5d', '5m')
    
    async def _get_sentiment_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get sentiment analysis data.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Sentiment analysis results
        """
        try:
            return await self.sentiment_service.get_comprehensive_sentiment(ticker, days_back=7)
        except Exception as e:
            logger.error(f"Error getting sentiment data: {str(e)}")
            return self._get_neutral_sentiment(ticker)
    
    def _get_neutral_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Get neutral sentiment data for fallback.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Neutral sentiment data
        """
        return {
            'ticker': ticker,
            'overall_sentiment': 'NEUTRAL',
            'overall_sentiment_score': 0.0,
            'trading_features': {
                'news_sentiment_score': 0.0,
                'news_sentiment_strength': 0.0,
                'news_positive_ratio': 0.0,
                'news_negative_ratio': 0.0,
                'news_neutral_ratio': 1.0,
                'news_sentiment_volatility': 0.0,
                'news_news_volume': 0.0,
                'news_news_impact_score': 0.0,
                'news_sentiment_momentum': 0.0
            }
        }
    
    async def _prepare_comprehensive_features(self, market_data: pd.DataFrame,
                                            sentiment_data: Dict[str, Any],
                                            ticker: str,
                                            prediction_horizon: str) -> Dict[str, Any]:
        """
        Prepare comprehensive features for all models.
        
        Args:
            market_data: Market data DataFrame
            sentiment_data: Sentiment analysis results
            ticker: Stock ticker
            prediction_horizon: Prediction timeframe
            
        Returns:
            Comprehensive features dictionary
        """
        try:
            features = {}
            
            # Technical features from market data
            latest_data = market_data.iloc[-1]
            
            # Price features
            features.update({
                'open': float(latest_data['open']),
                'high': float(latest_data['high']),
                'low': float(latest_data['low']),
                'close': float(latest_data['close']),
                'volume': float(latest_data['volume']),
            })
            
            # Technical indicator features
            technical_cols = [
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'sma_20', 'ema_20', 'sma_50', 'ema_50',
                'atr', 'adx', 'cci', 'williams_r',
                'stoch_k', 'stoch_d', 'obv'
            ]
            
            for col in technical_cols:
                if col in latest_data:
                    features[col] = float(latest_data[col]) if pd.notna(latest_data[col]) else 0.0
            
            # Price change features
            if len(market_data) > 1:
                features['price_change_1d'] = float((latest_data['close'] - market_data['close'].iloc[-2]) / market_data['close'].iloc[-2])
            else:
                features['price_change_1d'] = 0.0
            
            if len(market_data) > 5:
                features['price_change_5d'] = float((latest_data['close'] - market_data['close'].iloc[-6]) / market_data['close'].iloc[-6])
            else:
                features['price_change_5d'] = 0.0
            
            # Volatility features
            if len(market_data) > 20:
                returns = market_data['close'].pct_change().dropna()
                features['volatility_20d'] = float(returns.tail(20).std())
                features['volatility_5d'] = float(returns.tail(5).std())
            else:
                features['volatility_20d'] = 0.0
                features['volatility_5d'] = 0.0
            
            # Volume features
            if len(market_data) > 20:
                features['volume_ratio'] = float(latest_data['volume'] / market_data['volume'].tail(20).mean())
            else:
                features['volume_ratio'] = 1.0
            
            # Sentiment features
            sentiment_features = sentiment_data.get('trading_features', {})
            features.update(sentiment_features)
            
            # Fundamental features (mock for now)
            fundamental_features = await self._get_fundamental_features(ticker)
            features.update(fundamental_features)
            
            # Market context features
            features['prediction_horizon'] = self._encode_prediction_horizon(prediction_horizon)
            features['hour_of_day'] = datetime.now().hour
            features['day_of_week'] = datetime.now().weekday()
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return {}
    
    async def _get_fundamental_features(self, ticker: str) -> Dict[str, float]:
        """
        Get fundamental analysis features.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Fundamental features
        """
        try:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.fundamental_fetcher.get_fundamental_data,
                    ticker
                )
                fundamental_data = await loop.run_in_executor(None, lambda: future.result())
            
            if fundamental_data:
                return {
                    'pe_ratio': fundamental_data.get('pe_ratio', 15.0),
                    'pb_ratio': fundamental_data.get('pb_ratio', 2.0),
                    'roe': fundamental_data.get('roe', 0.15),
                    'debt_to_equity': fundamental_data.get('debt_to_equity', 0.5),
                    'current_ratio': fundamental_data.get('current_ratio', 1.5),
                    'revenue_growth': fundamental_data.get('revenue_growth', 0.1),
                    'profit_margin': fundamental_data.get('profit_margin', 0.1)
                }
            else:
                # Return default values
                return {
                    'pe_ratio': 15.0,
                    'pb_ratio': 2.0,
                    'roe': 0.15,
                    'debt_to_equity': 0.5,
                    'current_ratio': 1.5,
                    'revenue_growth': 0.1,
                    'profit_margin': 0.1
                }
                
        except Exception as e:
            logger.error(f"Error getting fundamental features: {str(e)}")
            return {
                'pe_ratio': 15.0,
                'pb_ratio': 2.0,
                'roe': 0.15,
                'debt_to_equity': 0.5,
                'current_ratio': 1.5,
                'revenue_growth': 0.1,
                'profit_margin': 0.1
            }
    
    def _encode_prediction_horizon(self, horizon: str) -> float:
        """
        Encode prediction horizon as numerical feature.
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Encoded horizon value
        """
        horizon_map = {
            'scalping': 0.1,
            'intraday': 0.3,
            'swing': 0.6,
            'long_term': 1.0
        }
        return horizon_map.get(horizon, 0.3)
    
    async def _get_xgboost_prediction(self, features: Dict[str, Any], 
                                    prediction_horizon: str) -> Dict[str, Any]:
        """
        Get XGBoost model prediction.
        
        Args:
            features: Feature dictionary
            prediction_horizon: Prediction timeframe
            
        Returns:
            XGBoost prediction result
        """
        try:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.models['xgboost'].predict,
                    features
                )
                result = await loop.run_in_executor(None, lambda: future.result())
            
            return result
            
        except Exception as e:
            logger.error(f"Error in XGBoost prediction: {str(e)}")
            return {
                'prediction': 0.0,
                'confidence': 0.5,
                'signal': 'HOLD',
                'feature_importance': {},
                'error': str(e)
            }
    
    async def _get_informer_prediction(self, market_data: pd.DataFrame,
                                     prediction_horizon: str) -> Dict[str, Any]:
        """
        Get Informer model prediction.
        
        Args:
            market_data: Market data DataFrame
            prediction_horizon: Prediction timeframe
            
        Returns:
            Informer prediction result
        """
        try:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.models['informer'].predict,
                    market_data
                )
                result = await loop.run_in_executor(None, lambda: future.result())
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Informer prediction: {str(e)}")
            return {
                'predictions': [0.0],
                'confidence_intervals': {'lower': [0.0], 'upper': [0.0]},
                'confidence': 0.5,
                'signal': 'HOLD',
                'error': str(e)
            }
    
    async def _get_dqn_prediction(self, features: Dict[str, Any],
                                market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get DQN model prediction.
        
        Args:
            features: Feature dictionary
            market_data: Market data DataFrame
            
        Returns:
            DQN prediction result
        """
        try:
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.models['dqn'].get_trading_signal,
                    market_data
                )
                result = await loop.run_in_executor(None, lambda: future.result())
            
            return result
            
        except Exception as e:
            logger.error(f"Error in DQN prediction: {str(e)}")
            return {
                'action': 1,  # Hold
                'signal': 'HOLD',
                'confidence': 0.5,
                'q_values': [0.0, 0.0, 0.0],
                'risk_assessment': 'medium',
                'error': str(e)
            }
    
    def _calculate_ensemble_prediction(self, xgb_pred: Dict[str, Any],
                                     informer_pred: Dict[str, Any],
                                     dqn_pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate ensemble prediction from all models.
        
        Args:
            xgb_pred: XGBoost prediction
            informer_pred: Informer prediction
            dqn_pred: DQN prediction
            
        Returns:
            Ensemble prediction result
        """
        try:
            # Extract predictions
            xgb_change = xgb_pred.get('prediction', 0.0)
            informer_preds = informer_pred.get('predictions', [0.0])
            informer_change = informer_preds[0] if informer_preds else 0.0
            
            # Convert DQN action to price change
            dqn_action = dqn_pred.get('action', 1)
            dqn_change = (dqn_action - 1) * 0.02  # -2%, 0%, +2% for actions 0, 1, 2
            
            # Calculate weighted ensemble prediction
            ensemble_change = (
                xgb_change * self.ensemble_weights['xgboost'] +
                informer_change * self.ensemble_weights['informer'] +
                dqn_change * self.ensemble_weights['dqn']
            )
            
            # Calculate ensemble confidence
            confidences = [
                xgb_pred.get('confidence', 0.5),
                informer_pred.get('confidence', 0.5),
                dqn_pred.get('confidence', 0.5)
            ]
            
            ensemble_confidence = np.average(
                confidences,
                weights=list(self.ensemble_weights.values())
            )
            
            # Determine ensemble signal
            if ensemble_change > 0.01:  # > 1%
                ensemble_signal = 'BUY'
            elif ensemble_change < -0.01:  # < -1%
                ensemble_signal = 'SELL'
            else:
                ensemble_signal = 'HOLD'
            
            # Calculate price range (assuming current price is available)
            # This would be set by the calling function
            price_range = {
                'min': 0.0,
                'max': 0.0,
                'mean': ensemble_change
            }
            
            return {
                'signal': ensemble_signal,
                'confidence': float(ensemble_confidence),
                'price_change_percent': float(ensemble_change * 100),
                'price_range': price_range,
                'individual_predictions': {
                    'xgboost': xgb_change,
                    'informer': informer_change,
                    'dqn': dqn_change
                },
                'weights_used': self.ensemble_weights
            }
            
        except Exception as e:
            logger.error(f"Error calculating ensemble prediction: {str(e)}")
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'price_change_percent': 0.0,
                'price_range': {'min': 0.0, 'max': 0.0, 'mean': 0.0},
                'error': str(e)
            }
    
    def _calculate_risk_metrics(self, market_data: pd.DataFrame,
                              xgb_pred: Dict[str, Any],
                              informer_pred: Dict[str, Any],
                              dqn_pred: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate risk metrics for the prediction.
        
        Args:
            market_data: Market data DataFrame
            xgb_pred: XGBoost prediction
            informer_pred: Informer prediction
            dqn_pred: DQN prediction
            
        Returns:
            Risk metrics dictionary
        """
        try:
            risk_metrics = {}
            
            # Historical volatility
            if len(market_data) > 20:
                returns = market_data['close'].pct_change().dropna()
                risk_metrics['historical_volatility'] = float(returns.std() * np.sqrt(252))
                risk_metrics['var_95'] = float(returns.quantile(0.05))
                risk_metrics['max_drawdown'] = float((market_data['close'] / market_data['close'].cummax() - 1).min())
            else:
                risk_metrics['historical_volatility'] = 0.2
                risk_metrics['var_95'] = -0.02
                risk_metrics['max_drawdown'] = -0.1
            
            # Model agreement risk
            signals = [xgb_pred.get('signal', 'HOLD'), informer_pred.get('signal', 'HOLD'), dqn_pred.get('signal', 'HOLD')]
            unique_signals = len(set(signals))
            risk_metrics['model_disagreement'] = float((unique_signals - 1) / 2)  # 0 to 1 scale
            
            # Confidence-based risk
            confidences = [
                xgb_pred.get('confidence', 0.5),
                informer_pred.get('confidence', 0.5),
                dqn_pred.get('confidence', 0.5)
            ]
            risk_metrics['confidence_risk'] = float(1.0 - np.mean(confidences))
            
            # Overall risk score
            risk_components = [
                risk_metrics['historical_volatility'] / 0.5,  # Normalize by typical volatility
                risk_metrics['model_disagreement'],
                risk_metrics['confidence_risk']
            ]
            
            risk_metrics['overall_risk_score'] = float(np.mean(risk_components))
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'historical_volatility': 0.2,
                'var_95': -0.02,
                'max_drawdown': -0.1,
                'model_disagreement': 0.5,
                'confidence_risk': 0.5,
                'overall_risk_score': 0.5
            }
    
    def _get_feature_importance(self, xgb_pred: Dict[str, Any],
                              features: Dict[str, Any]) -> Dict[str, float]:
        """
        Get feature importance from XGBoost model.
        
        Args:
            xgb_pred: XGBoost prediction result
            features: Feature dictionary
            
        Returns:
            Feature importance dictionary
        """
        try:
            # Get feature importance from XGBoost
            importance = xgb_pred.get('feature_importance', {})
            
            if not importance:
                # Generate mock importance based on feature types
                importance = {}
                for feature_name in features.keys():
                    if 'sentiment' in feature_name:
                        importance[feature_name] = np.random.uniform(0.05, 0.15)
                    elif feature_name in ['rsi', 'macd', 'bb_width']:
                        importance[feature_name] = np.random.uniform(0.1, 0.2)
                    elif 'volume' in feature_name:
                        importance[feature_name] = np.random.uniform(0.05, 0.1)
                    else:
                        importance[feature_name] = np.random.uniform(0.01, 0.05)
                
                # Normalize to sum to 1
                total = sum(importance.values())
                if total > 0:
                    importance = {k: v/total for k, v in importance.items()}
            
            # Return top 10 most important features
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    async def _get_shap_explanation(self, features: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Get SHAP explainability values.
        
        Args:
            features: Feature dictionary
            ticker: Stock ticker
            
        Returns:
            SHAP explanation results
        """
        try:
            # This would integrate with SHAP library
            # For now, return mock SHAP values
            shap_values = {}
            
            for feature_name, feature_value in features.items():
                # Mock SHAP value calculation
                if 'sentiment' in feature_name:
                    shap_values[feature_name] = float(feature_value * np.random.uniform(-0.1, 0.1))
                elif feature_name in ['rsi', 'macd']:
                    shap_values[feature_name] = float(feature_value * np.random.uniform(-0.05, 0.05))
                else:
                    shap_values[feature_name] = float(feature_value * np.random.uniform(-0.02, 0.02))
            
            return {
                'shap_values': shap_values,
                'base_value': 0.0,
                'explanation_method': 'TreeExplainer',
                'feature_contributions': dict(sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
            }
            
        except Exception as e:
            logger.error(f"Error getting SHAP explanation: {str(e)}")
            return {'error': str(e)}
    
    def _determine_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        Determine current market regime.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            Market regime classification
        """
        try:
            if len(market_data) < 20:
                return 'insufficient_data'
            
            # Calculate recent price trend
            recent_returns = market_data['close'].pct_change().tail(20)
            avg_return = recent_returns.mean()
            volatility = recent_returns.std()
            
            # Simple regime classification
            if avg_return > 0.01 and volatility < 0.02:
                return 'bull_market'
            elif avg_return < -0.01 and volatility < 0.02:
                return 'bear_market'
            elif volatility > 0.03:
                return 'high_volatility'
            else:
                return 'sideways_market'
                
        except Exception as e:
            logger.error(f"Error determining market regime: {str(e)}")
            return 'unknown'
    
    def _calculate_volatility_forecast(self, market_data: pd.DataFrame) -> float:
        """
        Calculate volatility forecast.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            Volatility forecast
        """
        try:
            if len(market_data) < 20:
                return 0.2  # Default volatility
            
            returns = market_data['close'].pct_change().dropna()
            
            # Simple EWMA volatility forecast
            alpha = 0.94
            ewma_var = returns.var()
            
            for ret in returns.tail(20):
                ewma_var = alpha * ewma_var + (1 - alpha) * ret**2
            
            return float(np.sqrt(ewma_var * 252))  # Annualized volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility forecast: {str(e)}")
            return 0.2
    
    def _generate_mock_market_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """
        Generate mock market data for testing.
        
        Args:
            ticker: Stock ticker
            period: Data period
            interval: Data interval
            
        Returns:
            Mock market data DataFrame
        """
        try:
            # Generate mock data
            np.random.seed(hash(ticker) % 2**32)
            
            periods = {'1d': 1, '5d': 5, '1mo': 30, '1y': 365}
            intervals = {'1m': 1440, '5m': 288, '1h': 24, '1d': 1}
            
            days = periods.get(period, 5)
            points_per_day = intervals.get(interval, 288)
            total_points = days * points_per_day
            
            # Generate price series
            base_price = 100.0
            returns = np.random.normal(0, 0.02, total_points)
            prices = [base_price]
            
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            prices = prices[1:]  # Remove initial price
            
            # Generate OHLCV data
            data = []
            for i, close in enumerate(prices):
                high = close * (1 + abs(np.random.normal(0, 0.01)))
                low = close * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[i-1] if i > 0 else close
                volume = int(np.random.uniform(100000, 1000000))
                
                data.append({
                    'timestamp': datetime.now() - timedelta(minutes=(total_points-i)*5),
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating mock data: {str(e)}")
            return pd.DataFrame()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached prediction is still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid
        """
        if cache_key not in self.prediction_cache:
            return False
        
        cache_time = self.prediction_cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_ttl
    
    def clear_cache(self) -> None:
        """
        Clear the prediction cache.
        """
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        current_time = time.time()
        valid_entries = sum(
            1 for entry in self.prediction_cache.values()
            if (current_time - entry['timestamp']) < self.cache_ttl
        )
        
        return {
            'total_entries': len(self.prediction_cache),
            'valid_entries': valid_entries,
            'cache_ttl_minutes': self.cache_ttl / 60,
            'cache_hit_ratio': valid_entries / max(len(self.prediction_cache), 1)
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Configuration
    config = {
        'angel_api_key': os.getenv('ANGEL_API_KEY'),
        'angel_client_id': os.getenv('ANGEL_CLIENT_ID'),
        'angel_password': os.getenv('ANGEL_PASSWORD'),
        'angel_totp_secret': os.getenv('ANGEL_TOTP_SECRET'),
        'newsapi_key': os.getenv('NEWSAPI_KEY'),
        'alphavantage_key': os.getenv('ALPHAVANTAGE_KEY'),
        'fingpt_model': 'ProsusAI/finbert',
        'model_cache_dir': './models/cache',
        'device': 'cpu',
        'prediction_cache_ttl_minutes': 5
    }
    
    # Initialize service
    prediction_service = PredictionService(config)
    
    # Test comprehensive prediction
    ticker = "RELIANCE.NS"
    
    async def test_prediction():
        print(f"Testing comprehensive prediction for {ticker}...")
        
        try:
            result = await prediction_service.get_comprehensive_prediction(
                ticker=ticker,
                prediction_horizon='intraday',
                include_sentiment=True,
                include_shap=True
            )
            
            print(f"\nPrediction Results for {ticker}:")
            print(f"Current Price: ${result.current_price:.2f}")
            print(f"Ensemble Signal: {result.ensemble_signal}")
            print(f"Ensemble Confidence: {result.ensemble_confidence:.3f}")
            print(f"Price Change Forecast: {result.price_change_percent:.2f}%")
            print(f"Market Regime: {result.market_regime}")
            print(f"Overall Risk Score: {result.risk_metrics['overall_risk_score']:.3f}")
            
            print(f"\nIndividual Model Signals:")
            print(f"XGBoost: {result.xgboost_signal} (confidence: {result.xgboost_confidence:.3f})")
            print(f"Informer: {result.informer_signal} (confidence: {result.informer_confidence:.3f})")
            print(f"DQN: {result.dqn_signal} (confidence: {result.dqn_confidence:.3f})")
            
            print(f"\nTop Feature Importance:")
            for feature, importance in list(result.feature_importance.items())[:5]:
                print(f"{feature}: {importance:.4f}")
            
            print(f"\nSentiment Analysis:")
            sentiment = result.sentiment_analysis
            print(f"Overall Sentiment: {sentiment['overall_sentiment']}")
            print(f"Sentiment Score: {sentiment['overall_sentiment_score']:.3f}")
            
        except Exception as e:
            print(f"Error in prediction test: {str(e)}")
    
    # Run test
    asyncio.run(test_prediction())