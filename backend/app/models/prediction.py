from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum

class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class ModelType(str, Enum):
    """Supported ML model types."""
    XGBOOST = "xgboost"
    INFORMER = "informer"
    DQN = "dqn"

class SentimentLabel(str, Enum):
    """Sentiment analysis labels."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class PredictionWindow(str, Enum):
    """Prediction time windows."""
    SCALPING = "scalping"  # 5 minutes
    INTRADAY = "intraday"  # 1 hour
    SWING = "swing"        # 1 day
    POSITION = "position"  # 1 week
    LONG_TERM = "long_term" # 1 month

class XGBoostPrediction(BaseModel):
    """XGBoost model prediction result."""
    predicted_price: float = Field(..., gt=0, description="Predicted stock price")
    price_range_low: float = Field(..., gt=0, description="Lower bound of price prediction")
    price_range_high: float = Field(..., gt=0, description="Upper bound of price prediction")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    signal: SignalType = Field(..., description="Trading signal")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    model_accuracy: Optional[float] = Field(None, ge=0, le=1, description="Model accuracy score")
    
class InformerPrediction(BaseModel):
    """Informer transformer model prediction result."""
    predicted_prices: List[float] = Field(..., description="Sequence of predicted prices")
    prediction_timestamps: List[datetime] = Field(..., description="Timestamps for predictions")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    signal: SignalType = Field(..., description="Trading signal")
    attention_weights: Optional[Dict[str, Any]] = Field(None, description="Attention mechanism weights")
    volatility_forecast: Optional[float] = Field(None, ge=0, description="Predicted volatility")
    
class DQNPrediction(BaseModel):
    """DQN reinforcement learning prediction result."""
    action: SignalType = Field(..., description="Recommended trading action")
    action_probability: float = Field(..., ge=0, le=1, description="Action probability")
    q_values: Dict[str, float] = Field(..., description="Q-values for each action")
    confidence: float = Field(..., ge=0, le=1, description="Action confidence score")
    expected_return: float = Field(..., description="Expected return from action")
    risk_score: float = Field(..., ge=0, le=1, description="Risk assessment score")
    position_size: Optional[float] = Field(None, ge=0, le=1, description="Recommended position size")
    
class SentimentAnalysis(BaseModel):
    """News sentiment analysis result."""
    label: SentimentLabel = Field(..., description="Overall sentiment label")
    score: float = Field(..., ge=-1, le=1, description="Sentiment score (-1 to 1)")
    confidence: float = Field(..., ge=0, le=1, description="Sentiment confidence")
    news_count: int = Field(..., ge=0, description="Number of news articles analyzed")
    positive_count: int = Field(..., ge=0, description="Number of positive articles")
    negative_count: int = Field(..., ge=0, description="Number of negative articles")
    neutral_count: int = Field(..., ge=0, description="Number of neutral articles")
    weighted_score: float = Field(..., ge=-1, le=1, description="Importance-weighted sentiment score")
    
class SHAPExplanation(BaseModel):
    """SHAP explainability results."""
    feature_contributions: Dict[str, float] = Field(..., description="Feature contribution to prediction")
    base_value: float = Field(..., description="Base prediction value")
    prediction_value: float = Field(..., description="Final prediction value")
    top_positive_features: List[Dict[str, Union[str, float]]] = Field(..., description="Top positive contributing features")
    top_negative_features: List[Dict[str, Union[str, float]]] = Field(..., description="Top negative contributing features")
    
class ConsensusPrediction(BaseModel):
    """Consensus prediction from multiple models."""
    final_signal: SignalType = Field(..., description="Final consensus trading signal")
    confidence: float = Field(..., ge=0, le=1, description="Consensus confidence score")
    predicted_price: float = Field(..., gt=0, description="Consensus predicted price")
    price_range_low: float = Field(..., gt=0, description="Lower bound of consensus prediction")
    price_range_high: float = Field(..., gt=0, description="Upper bound of consensus prediction")
    model_weights: Dict[str, float] = Field(..., description="Weight given to each model")
    agreement_score: float = Field(..., ge=0, le=1, description="Inter-model agreement score")
    
class PredictionResult(BaseModel):
    """Complete prediction result from all models."""
    ticker: str = Field(..., description="Stock ticker symbol")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    timeframe: PredictionWindow = Field(..., description="Prediction timeframe")
    current_price: float = Field(..., gt=0, description="Current stock price")
    
    # Individual model predictions
    xgboost: XGBoostPrediction
    informer: InformerPrediction
    dqn: DQNPrediction
    
    # Sentiment and explainability
    sentiment: SentimentAnalysis
    shap_explanation: Optional[SHAPExplanation] = Field(None, description="SHAP explainability results")
    
    # Consensus prediction
    consensus: ConsensusPrediction
    
    # Metadata
    model_versions: Dict[str, str] = Field(default_factory=dict, description="Model version information")
    processing_time: float = Field(..., ge=0, description="Total processing time in seconds")
    
class PredictionRequest(BaseModel):
    """Request for stock prediction."""
    ticker: str = Field(..., description="Stock ticker symbol")
    timeframe: PredictionWindow = Field(..., description="Prediction timeframe")
    include_shap: bool = Field(default=False, description="Include SHAP explainability")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    models: Optional[List[ModelType]] = Field(None, description="Specific models to use (default: all)")
    
class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    tickers: List[str] = Field(..., min_items=1, max_items=50, description="List of stock tickers")
    timeframe: PredictionWindow = Field(..., description="Prediction timeframe")
    include_shap: bool = Field(default=False, description="Include SHAP explainability")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    models: Optional[List[ModelType]] = Field(None, description="Specific models to use (default: all)")
    
class PredictionHistory(BaseModel):
    """Historical prediction record."""
    id: str = Field(..., description="Unique prediction ID")
    ticker: str = Field(..., description="Stock ticker symbol")
    prediction_timestamp: datetime = Field(..., description="When prediction was made")
    target_timestamp: datetime = Field(..., description="Target time for prediction")
    predicted_price: float = Field(..., gt=0, description="Predicted price")
    actual_price: Optional[float] = Field(None, gt=0, description="Actual price (if available)")
    prediction_accuracy: Optional[float] = Field(None, ge=0, le=1, description="Prediction accuracy")
    signal: SignalType = Field(..., description="Trading signal given")
    timeframe: PredictionWindow = Field(..., description="Prediction timeframe")
    
class ModelPerformance(BaseModel):
    """Model performance metrics."""
    model_type: ModelType = Field(..., description="Type of model")
    ticker: str = Field(..., description="Stock ticker symbol")
    timeframe: PredictionWindow = Field(..., description="Prediction timeframe")
    
    # Accuracy metrics
    mae: float = Field(..., ge=0, description="Mean Absolute Error")
    mse: float = Field(..., ge=0, description="Mean Squared Error")
    rmse: float = Field(..., ge=0, description="Root Mean Squared Error")
    mape: float = Field(..., ge=0, description="Mean Absolute Percentage Error")
    r2_score: float = Field(..., ge=-1, le=1, description="R-squared score")
    
    # Trading performance
    signal_accuracy: float = Field(..., ge=0, le=1, description="Trading signal accuracy")
    precision: float = Field(..., ge=0, le=1, description="Precision score")
    recall: float = Field(..., ge=0, le=1, description="Recall score")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score")
    
    # Time-based metrics
    last_updated: datetime = Field(..., description="Last model update")
    training_samples: int = Field(..., ge=0, description="Number of training samples")
    evaluation_period: str = Field(..., description="Evaluation period")
    
class PredictionAlert(BaseModel):
    """Prediction-based alert."""
    id: str = Field(..., description="Unique alert ID")
    ticker: str = Field(..., description="Stock ticker symbol")
    alert_type: str = Field(..., description="Type of alert")
    message: str = Field(..., description="Alert message")
    severity: str = Field(..., description="Alert severity level")
    timestamp: datetime = Field(..., description="Alert timestamp")
    prediction_data: Dict[str, Any] = Field(..., description="Related prediction data")
    is_active: bool = Field(default=True, description="Whether alert is active")
    
    @validator('severity')
    def validate_severity(cls, v):
        allowed_severities = ['low', 'medium', 'high', 'critical']
        if v.lower() not in allowed_severities:
            raise ValueError(f'Severity must be one of {allowed_severities}')
        return v.lower()