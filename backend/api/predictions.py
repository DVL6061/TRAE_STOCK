from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import logging
from datetime import datetime, timedelta
import json

# Import prediction models and utilities
from core.prediction_engine import (
    generate_price_prediction,
    generate_trading_signal,
    get_prediction_explanation
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Models for request and response
class PredictionRequest(BaseModel):
    ticker: str
    prediction_window: str  # "1d", "3d", "1w", "2w", "1m", "3m", etc.
    include_news_sentiment: bool = True
    include_technical_indicators: bool = True
    include_explanation: bool = True

class TradingSignalRequest(BaseModel):
    ticker: str
    timeframe: str  # "scalping", "intraday", "short_term", "medium_term", "long_term"
    risk_tolerance: Optional[str] = "moderate"  # "low", "moderate", "high"
    include_explanation: bool = True

class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    strategy: str  # "ml_prediction", "dqn_rl", "combined"
    initial_capital: float = 100000.0  # Default 1 lakh INR

# Endpoints
@router.post("/price")
async def predict_price(request: PredictionRequest):
    """Generate price prediction for a stock"""
    try:
        # This would call the actual implementation in core.prediction_engine
        prediction = await generate_price_prediction(
            ticker=request.ticker,
            prediction_window=request.prediction_window,
            include_news_sentiment=request.include_news_sentiment,
            include_technical_indicators=request.include_technical_indicators
        )
        
        # Add explanation if requested
        if request.include_explanation:
            explanation = await get_prediction_explanation(
                ticker=request.ticker,
                prediction=prediction,
                prediction_window=request.prediction_window
            )
            prediction["explanation"] = explanation
            
        return prediction
    except Exception as e:
        logger.error(f"Error generating price prediction for {request.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate price prediction: {str(e)}")

@router.post("/trading-signal")
async def get_trading_signal(request: TradingSignalRequest):
    """Generate trading signal (buy/sell/hold) for a stock"""
    try:
        # This would call the actual implementation in core.prediction_engine
        signal = await generate_trading_signal(
            ticker=request.ticker,
            timeframe=request.timeframe,
            risk_tolerance=request.risk_tolerance
        )
        
        # Add explanation if requested
        if request.include_explanation:
            explanation = await get_prediction_explanation(
                ticker=request.ticker,
                prediction=signal,
                prediction_window=request.timeframe
            )
            signal["explanation"] = explanation
            
        return signal
    except Exception as e:
        logger.error(f"Error generating trading signal for {request.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate trading signal: {str(e)}")

@router.post("/backtest")
async def backtest_strategy(request: BacktestRequest):
    """Backtest a trading strategy"""
    try:
        # This would be implemented to run backtests
        # For now, return a placeholder
        return {
            "ticker": request.ticker,
            "strategy": request.strategy,
            "period": f"{request.start_date} to {request.end_date}",
            "initial_capital": request.initial_capital,
            "final_capital": request.initial_capital * 1.15,  # Placeholder 15% return
            "return_pct": 15.0,
            "annualized_return": 12.5,
            "max_drawdown": -8.3,
            "sharpe_ratio": 1.2,
            "trades": [
                # Sample trades would be here
            ]
        }
    except Exception as e:
        logger.error(f"Error backtesting strategy for {request.ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to backtest strategy: {str(e)}")

@router.get("/models")
async def get_available_models():
    """Get list of available prediction models"""
    try:
        # This would be implemented to list available models
        # For now, return a placeholder
        return {
            "models": [
                {
                    "id": "xgboost_base",
                    "name": "XGBoost Baseline",
                    "description": "Baseline tabular prediction model using XGBoost",
                    "supported_timeframes": ["1d", "3d", "1w", "2w"]
                },
                {
                    "id": "informer_transformer",
                    "name": "Informer Transformer",
                    "description": "Transformer-based time-series forecasting model",
                    "supported_timeframes": ["1d", "3d", "1w", "2w", "1m", "3m"]
                },
                {
                    "id": "dqn_rl",
                    "name": "DQN Reinforcement Learning",
                    "description": "Deep Q-Network for trading strategy optimization",
                    "supported_timeframes": ["intraday", "short_term", "medium_term"]
                },
                {
                    "id": "ensemble",
                    "name": "Ensemble Model",
                    "description": "Combination of multiple models with news sentiment",
                    "supported_timeframes": ["1d", "3d", "1w", "2w", "1m"]
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching available models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch available models: {str(e)}")