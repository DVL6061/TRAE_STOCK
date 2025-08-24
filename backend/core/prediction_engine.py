import pandas as pd
import numpy as np
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import json

# Import data fetching and news processing utilities
from core.data_fetcher import fetch_historical_data, fetch_technical_indicators
from core.news_processor import fetch_news, analyze_sentiment

# This would be replaced with actual ML model imports
# import xgboost as xgb
# import torch
# import shap

logger = logging.getLogger(__name__)

# Mock prediction models for development
class MockXGBoostModel:
    """Mock XGBoost model for development"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        logger.info(f"Initialized mock XGBoost model for {ticker}")
        
    async def predict(self, data: pd.DataFrame, prediction_window: str) -> np.ndarray:
        # Generate mock predictions
        last_close = data["close"].iloc[-1]
        
        # Map prediction window to number of days
        days_map = {
            "1d": 1, "3d": 3, "1w": 7, "2w": 14, "1m": 30, "3m": 90
        }
        days = days_map.get(prediction_window, 1)
        
        # Generate trend based on recent data
        recent_trend = data["close"].iloc[-5:].pct_change().mean()
        
        # Add some randomness to the prediction
        trend_factor = recent_trend * 10 + np.random.normal(0, 0.01)
        predicted_change = trend_factor * days / 10
        
        # Calculate predicted price
        predicted_price = last_close * (1 + predicted_change)
        
        # Add some uncertainty for confidence interval
        std_dev = last_close * 0.02 * np.sqrt(days / 7)  # Increasing with time
        
        return {
            "predicted_price": round(predicted_price, 2),
            "confidence_interval": {
                "lower": round(predicted_price - 1.96 * std_dev, 2),
                "upper": round(predicted_price + 1.96 * std_dev, 2)
            },
            "predicted_change": round(predicted_change * 100, 2),
            "prediction_window": prediction_window
        }

class MockInformerModel:
    """Mock Informer (Transformer) model for development"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        logger.info(f"Initialized mock Informer model for {ticker}")
        
    async def predict(self, data: pd.DataFrame, prediction_window: str) -> np.ndarray:
        # Similar to XGBoost but with different characteristics
        last_close = data["close"].iloc[-1]
        
        # Map prediction window to number of days
        days_map = {
            "1d": 1, "3d": 3, "1w": 7, "2w": 14, "1m": 30, "3m": 90
        }
        days = days_map.get(prediction_window, 1)
        
        # Generate trend based on longer historical data
        long_trend = data["close"].pct_change().mean()
        
        # Add some randomness to the prediction
        trend_factor = long_trend * 15 + np.random.normal(0, 0.008)
        predicted_change = trend_factor * days / 15
        
        # Calculate predicted price
        predicted_price = last_close * (1 + predicted_change)
        
        # Add some uncertainty for confidence interval
        # Informer typically has narrower confidence intervals for longer predictions
        std_dev = last_close * 0.015 * np.sqrt(days / 10)
        
        return {
            "predicted_price": round(predicted_price, 2),
            "confidence_interval": {
                "lower": round(predicted_price - 1.96 * std_dev, 2),
                "upper": round(predicted_price + 1.96 * std_dev, 2)
            },
            "predicted_change": round(predicted_change * 100, 2),
            "prediction_window": prediction_window
        }

class MockDQNModel:
    """Mock Deep Q-Network model for development"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        logger.info(f"Initialized mock DQN model for {ticker}")
        
    async def predict_action(self, data: pd.DataFrame, risk_tolerance: str) -> Dict[str, Any]:
        # Generate mock trading signal
        last_close = data["close"].iloc[-1]
        
        # Calculate recent price momentum
        returns = data["close"].pct_change().dropna()
        momentum_5d = returns.iloc[-5:].mean()
        momentum_20d = returns.iloc[-20:].mean()
        
        # Calculate mock technical signals
        rsi_value = 50 + momentum_5d * 1000  # Mock RSI
        rsi_value = max(0, min(100, rsi_value))  # Bound between 0 and 100
        
        macd_value = momentum_5d * 10  # Mock MACD
        macd_signal = momentum_20d * 10  # Mock MACD signal
        
        # Determine action based on signals
        if rsi_value > 70 and macd_value < macd_signal:
            action = "sell"
            confidence = 0.7 + abs(rsi_value - 70) / 100
        elif rsi_value < 30 and macd_value > macd_signal:
            action = "buy"
            confidence = 0.7 + abs(30 - rsi_value) / 100
        else:
            # Adjust hold confidence based on how close to neutral
            distance_from_neutral = abs(rsi_value - 50) / 50
            action = "hold"
            confidence = 0.5 + (0.5 - distance_from_neutral)
        
        # Adjust based on risk tolerance
        if risk_tolerance == "low":
            # More conservative - bias toward hold
            if action != "hold":
                confidence *= 0.8
        elif risk_tolerance == "high":
            # More aggressive - boost buy/sell confidence
            if action != "hold":
                confidence = min(0.95, confidence * 1.2)
        
        return {
            "action": action,
            "confidence": round(min(0.95, confidence), 2),
            "price": last_close,
            "signals": {
                "rsi": round(rsi_value, 2),
                "macd": round(macd_value, 4),
                "macd_signal": round(macd_signal, 4),
                "momentum_5d": round(momentum_5d * 100, 2),
                "momentum_20d": round(momentum_20d * 100, 2)
            }
        }

# Model factory function
async def get_model(model_type: str, ticker: str):
    """Get the appropriate prediction model"""
    if model_type == "xgboost":
        return MockXGBoostModel(ticker)
    elif model_type == "informer":
        return MockInformerModel(ticker)
    elif model_type == "dqn":
        return MockDQNModel(ticker)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

async def generate_price_prediction(
    ticker: str,
    prediction_window: str,
    include_news_sentiment: bool = True,
    include_technical_indicators: bool = True
) -> Dict[str, Any]:
    """Generate price prediction for a stock"""
    try:
        # Fetch historical data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        historical_data = await fetch_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval="1d"  # Daily data for predictions
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Add technical indicators if requested
        if include_technical_indicators:
            indicators = await fetch_technical_indicators(
                ticker=ticker,
                indicators=["rsi", "macd", "sma", "ema", "bb"],
                period=14,
                start_date=start_date,
                end_date=end_date
            )
            
            # In a real implementation, these would be properly merged with the dataframe
            # For now, we'll just note that they were included
            technical_indicators_included = True
        else:
            technical_indicators_included = False
        
        # Fetch and analyze news sentiment if requested
        if include_news_sentiment:
            news_start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            news_items = await fetch_news(
                ticker=ticker,
                start_date=news_start_date,
                end_date=end_date
            )
            
            # Analyze sentiment for each news item
            for item in news_items:
                sentiment = await analyze_sentiment(item["title"] + " " + item["description"])
                item["sentiment"] = sentiment
            
            # Calculate overall sentiment score
            sentiment_scores = [item["sentiment"]["score"] for item in news_items]
            avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            # In a real implementation, this would be incorporated into the prediction model
            # For now, we'll just note the sentiment score
            news_sentiment_score = avg_sentiment_score
            news_sentiment_included = True
        else:
            news_sentiment_score = 0
            news_sentiment_included = False
            news_items = []
        
        # Choose appropriate model based on prediction window
        if prediction_window in ["1d", "3d", "1w"]:
            model = await get_model("xgboost", ticker)
        else:
            model = await get_model("informer", ticker)
        
        # Generate prediction
        prediction = await model.predict(df, prediction_window)
        
        # Adjust prediction based on news sentiment (in a real implementation)
        if include_news_sentiment and news_sentiment_score != 0:
            # Simple adjustment for demonstration
            sentiment_adjustment = news_sentiment_score * 0.02 * prediction["predicted_price"]
            prediction["predicted_price"] += sentiment_adjustment
            prediction["confidence_interval"]["lower"] += sentiment_adjustment
            prediction["confidence_interval"]["upper"] += sentiment_adjustment
            prediction["predicted_change"] += news_sentiment_score * 2  # 2 percentage points per unit of sentiment
            
            # Round values
            prediction["predicted_price"] = round(prediction["predicted_price"], 2)
            prediction["confidence_interval"]["lower"] = round(prediction["confidence_interval"]["lower"], 2)
            prediction["confidence_interval"]["upper"] = round(prediction["confidence_interval"]["upper"], 2)
            prediction["predicted_change"] = round(prediction["predicted_change"], 2)
        
        # Prepare response
        response = {
            "ticker": ticker,
            "current_price": df["close"].iloc[-1],
            "prediction_date": datetime.now().isoformat(),
            "prediction_window": prediction_window,
            "predicted_price": prediction["predicted_price"],
            "confidence_interval": prediction["confidence_interval"],
            "predicted_change_percent": prediction["predicted_change"],
            "technical_indicators_included": technical_indicators_included,
            "news_sentiment_included": news_sentiment_included
        }
        
        # Add news sentiment information if included
        if include_news_sentiment:
            response["news_sentiment"] = {
                "score": round(news_sentiment_score, 2),
                "news_count": len(news_items),
                "top_news": sorted(
                    news_items,
                    key=lambda x: abs(x["sentiment"]["score"]),
                    reverse=True
                )[:3]  # Top 3 most impactful news
            }
        
        return response
    except Exception as e:
        logger.error(f"Error generating price prediction for {ticker}: {str(e)}")
        raise Exception(f"Failed to generate price prediction: {str(e)}")

async def generate_trading_signal(
    ticker: str,
    timeframe: str,
    risk_tolerance: str = "moderate"
) -> Dict[str, Any]:
    """Generate trading signal (buy/sell/hold) for a stock"""
    try:
        # Fetch historical data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        historical_data = await fetch_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval="1d"  # Daily data for signals
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Fetch technical indicators
        indicators = await fetch_technical_indicators(
            ticker=ticker,
            indicators=["rsi", "macd", "sma", "ema", "bb", "adx", "stoch"],
            period=14,
            start_date=start_date,
            end_date=end_date
        )
        
        # Fetch recent news and sentiment
        news_start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        news_items = await fetch_news(
            ticker=ticker,
            start_date=news_start_date,
            end_date=end_date
        )
        
        # Analyze sentiment for each news item
        for item in news_items:
            sentiment = await analyze_sentiment(item["title"] + " " + item["description"])
            item["sentiment"] = sentiment
        
        # Calculate overall sentiment score
        sentiment_scores = [item["sentiment"]["score"] for item in news_items]
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Use DQN model for trading signals
        model = await get_model("dqn", ticker)
        
        # Generate trading signal
        signal = await model.predict_action(df, risk_tolerance)
        
        # Adjust signal based on news sentiment (in a real implementation)
        if avg_sentiment_score > 0.5 and signal["action"] == "hold":
            # Strong positive sentiment might upgrade hold to buy
            signal["action"] = "buy"
            signal["confidence"] = min(0.95, signal["confidence"] * 0.8 + avg_sentiment_score * 0.2)
        elif avg_sentiment_score < -0.5 and signal["action"] == "hold":
            # Strong negative sentiment might upgrade hold to sell
            signal["action"] = "sell"
            signal["confidence"] = min(0.95, signal["confidence"] * 0.8 + abs(avg_sentiment_score) * 0.2)
        
        # Prepare response
        response = {
            "ticker": ticker,
            "current_price": df["close"].iloc[-1],
            "signal_date": datetime.now().isoformat(),
            "timeframe": timeframe,
            "risk_tolerance": risk_tolerance,
            "action": signal["action"],
            "confidence": signal["confidence"],
            "technical_signals": signal["signals"],
            "news_sentiment_score": round(avg_sentiment_score, 2),
            "news_count": len(news_items)
        }
        
        # Add top influential news
        response["top_influential_news"] = sorted(
            news_items,
            key=lambda x: abs(x["sentiment"]["score"]),
            reverse=True
        )[:3]  # Top 3 most impactful news
        
        return response
    except Exception as e:
        logger.error(f"Error generating trading signal for {ticker}: {str(e)}")
        raise Exception(f"Failed to generate trading signal: {str(e)}")

async def get_prediction_explanation(
    ticker: str,
    prediction: Dict[str, Any],
    prediction_window: str
) -> Dict[str, Any]:
    """Generate explanation for a prediction using SHAP values"""
    try:
        # In a real implementation, this would use SHAP values to explain the prediction
        # For now, we'll generate a mock explanation
        
        # Extract relevant information from the prediction
        if "action" in prediction:
            # This is a trading signal
            action = prediction["action"]
            confidence = prediction["confidence"]
            technical_signals = prediction["technical_signals"]
            news_sentiment_score = prediction.get("news_sentiment_score", 0)
            
            # Generate explanation based on action
            if action == "buy":
                main_factors = []
                if technical_signals["rsi"] < 40:
                    main_factors.append(f"RSI is relatively low at {technical_signals['rsi']}, indicating potential overselling")
                if technical_signals["macd"] > technical_signals["macd_signal"]:
                    main_factors.append("MACD is above signal line, suggesting positive momentum")
                if technical_signals["momentum_5d"] > 0:
                    main_factors.append(f"Recent 5-day momentum is positive at {technical_signals['momentum_5d']}%")
                if news_sentiment_score > 0:
                    main_factors.append(f"News sentiment is positive with a score of {news_sentiment_score}")
                
                explanation = {
                    "summary": f"The model recommends a BUY action with {confidence*100:.1f}% confidence based on technical indicators and news sentiment.",
                    "main_factors": main_factors,
                    "factor_weights": {
                        "technical_analysis": 0.7,
                        "news_sentiment": 0.3
                    }
                }
            elif action == "sell":
                main_factors = []
                if technical_signals["rsi"] > 60:
                    main_factors.append(f"RSI is relatively high at {technical_signals['rsi']}, indicating potential overbuying")
                if technical_signals["macd"] < technical_signals["macd_signal"]:
                    main_factors.append("MACD is below signal line, suggesting negative momentum")
                if technical_signals["momentum_5d"] < 0:
                    main_factors.append(f"Recent 5-day momentum is negative at {technical_signals['momentum_5d']}%")
                if news_sentiment_score < 0:
                    main_factors.append(f"News sentiment is negative with a score of {news_sentiment_score}")
                
                explanation = {
                    "summary": f"The model recommends a SELL action with {confidence*100:.1f}% confidence based on technical indicators and news sentiment.",
                    "main_factors": main_factors,
                    "factor_weights": {
                        "technical_analysis": 0.7,
                        "news_sentiment": 0.3
                    }
                }
            else:  # hold
                main_factors = []
                if 40 <= technical_signals["rsi"] <= 60:
                    main_factors.append(f"RSI is neutral at {technical_signals['rsi']}")
                if abs(technical_signals["macd"] - technical_signals["macd_signal"]) < 0.001:
                    main_factors.append("MACD is close to signal line, suggesting sideways movement")
                if abs(technical_signals["momentum_5d"]) < 1:
                    main_factors.append(f"Recent 5-day momentum is flat at {technical_signals['momentum_5d']}%")
                if abs(news_sentiment_score) < 0.2:
                    main_factors.append(f"News sentiment is neutral with a score of {news_sentiment_score}")
                
                explanation = {
                    "summary": f"The model recommends a HOLD action with {confidence*100:.1f}% confidence based on technical indicators and news sentiment.",
                    "main_factors": main_factors,
                    "factor_weights": {
                        "technical_analysis": 0.7,
                        "news_sentiment": 0.3
                    }
                }
        else:
            # This is a price prediction
            predicted_price = prediction["predicted_price"]
            current_price = prediction["current_price"]
            predicted_change = prediction["predicted_change_percent"]
            news_sentiment_included = prediction.get("news_sentiment_included", False)
            news_sentiment = prediction.get("news_sentiment", {"score": 0})
            
            # Generate explanation based on prediction
            main_factors = []
            if predicted_change > 0:
                main_factors.append(f"Historical price trend shows upward momentum of approximately {predicted_change/2:.1f}%")
                if news_sentiment_included and news_sentiment["score"] > 0:
                    main_factors.append(f"Positive news sentiment with a score of {news_sentiment['score']} contributes approximately {predicted_change/2:.1f}% to the prediction")
            else:
                main_factors.append(f"Historical price trend shows downward momentum of approximately {predicted_change/2:.1f}%")
                if news_sentiment_included and news_sentiment["score"] < 0:
                    main_factors.append(f"Negative news sentiment with a score of {news_sentiment['score']} contributes approximately {predicted_change/2:.1f}% to the prediction")
            
            # Add prediction window context
            window_context = {
                "1d": "short-term (1 day)",
                "3d": "short-term (3 days)",
                "1w": "short-term (1 week)",
                "2w": "medium-term (2 weeks)",
                "1m": "medium-term (1 month)",
                "3m": "long-term (3 months)"
            }
            timeframe = window_context.get(prediction_window, prediction_window)
            
            explanation = {
                "summary": f"The model predicts a {predicted_change:.1f}% {'increase' if predicted_change > 0 else 'decrease'} in {ticker} price over a {timeframe} timeframe, from ₹{current_price:.2f} to ₹{predicted_price:.2f}.",
                "main_factors": main_factors,
                "factor_weights": {
                    "historical_price_trend": 0.5,
                    "technical_indicators": 0.3,
                    "news_sentiment": 0.2 if news_sentiment_included else 0
                }
            }
        
        return explanation
    except Exception as e:
        logger.error(f"Error generating prediction explanation for {ticker}: {str(e)}")
        raise Exception(f"Failed to generate prediction explanation: {str(e)}")