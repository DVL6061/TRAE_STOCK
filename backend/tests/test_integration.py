#!/usr/bin/env python3
"""
Integration tests for the complete system workflow
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np

# Import system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from main import app
from services.data_fetcher import DataFetcher
from services.prediction_service import PredictionService
from services.news_service import NewsService
from ml_models.model_factory import ModelFactory
from utils.technical_indicators import TechnicalIndicators

# Test client
client = TestClient(app)

class TestDataPipeline:
    """Test complete data pipeline integration"""
    
    def setup_method(self):
        """Setup test components"""
        self.data_fetcher = DataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.model_factory = ModelFactory()
    
    @patch('yfinance.download')
    def test_historical_data_to_indicators_pipeline(self, mock_yf_download):
        """Test pipeline from historical data fetch to technical indicators"""
        # Mock Yahoo Finance data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.randn(100) * 10 + 2000,
            'High': np.random.randn(100) * 10 + 2020,
            'Low': np.random.randn(100) * 10 + 1980,
            'Close': np.random.randn(100) * 10 + 2000,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        mock_yf_download.return_value = mock_data
        
        # Test complete pipeline
        symbol = "RELIANCE"
        
        # 1. Fetch historical data
        historical_data = self.data_fetcher.get_historical_data(
            symbol=symbol,
            start_date='2023-01-01',
            end_date='2023-04-10',
            interval='1d'
        )
        
        assert not historical_data.empty
        assert 'close' in historical_data.columns
        
        # 2. Calculate technical indicators
        indicators = self.technical_indicators.calculate_all_indicators(historical_data)
        
        assert not indicators.empty
        assert 'sma_20' in indicators.columns
        assert 'rsi' in indicators.columns
        
        # 3. Combine data for ML model
        combined_data = pd.concat([historical_data, indicators], axis=1)
        combined_data = combined_data.dropna()
        
        assert not combined_data.empty
        assert len(combined_data.columns) > len(historical_data.columns)
    
    @patch('services.data_fetcher.DataFetcher.get_historical_data')
    def test_data_to_prediction_pipeline(self, mock_get_historical):
        """Test pipeline from data to ML prediction"""
        # Mock historical data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'open': np.random.randn(100) * 10 + 2000,
            'high': np.random.randn(100) * 10 + 2020,
            'low': np.random.randn(100) * 10 + 1980,
            'close': np.random.randn(100) * 10 + 2000,
            'volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        mock_get_historical.return_value = mock_data
        
        # Create XGBoost model
        xgb_model = self.model_factory.create_model('xgboost')
        
        # Calculate indicators
        indicators = self.technical_indicators.calculate_all_indicators(mock_data)
        combined_data = pd.concat([mock_data, indicators], axis=1).dropna()
        
        if len(combined_data) > 50:  # Ensure sufficient data
            # Prepare features
            feature_columns = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'ema_12', 'rsi']
            available_columns = [col for col in feature_columns if col in combined_data.columns]
            
            X = combined_data[available_columns].values
            y = combined_data['close'].shift(-1).dropna().values
            
            # Ensure same length
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            if len(X) > 20:  # Minimum data for training
                # Train model
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                xgb_model.train(X_train, y_train)
                
                # Make prediction
                predictions = xgb_model.predict(X_test)
                
                assert predictions is not None
                assert len(predictions) == len(X_test)

class TestAPIIntegration:
    """Test API endpoint integration"""
    
    @patch('services.prediction_service.PredictionService.get_prediction')
    @patch('services.data_fetcher.DataFetcher.get_historical_data')
    def test_prediction_api_integration(self, mock_get_historical, mock_get_prediction):
        """Test complete prediction API workflow"""
        # Mock data
        mock_data = pd.DataFrame({
            'open': [2000, 2010, 2020],
            'high': [2050, 2060, 2070],
            'low': [1950, 1960, 1970],
            'close': [2040, 2050, 2060],
            'volume': [1000000, 1100000, 1200000]
        })
        mock_get_historical.return_value = mock_data
        
        mock_prediction = {
            "symbol": "RELIANCE",
            "predicted_price": 2500.50,
            "confidence": 0.85,
            "direction": "up",
            "timestamp": datetime.now().isoformat(),
            "model_used": "XGBoost",
            "technical_indicators": {
                "rsi": 65.2,
                "macd": 12.5,
                "sma_20": 2480.0
            }
        }
        mock_get_prediction.return_value = mock_prediction
        
        # Test API call
        response = client.get("/api/v1/prediction/RELIANCE")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "RELIANCE"
        assert "predicted_price" in data
        assert "confidence" in data
    
    @patch('services.news_service.NewsService.get_latest_news')
    def test_news_api_integration(self, mock_get_news):
        """Test news API integration"""
        mock_news = [
            {
                "title": "Market Update",
                "content": "Stock market shows positive trends...",
                "sentiment": "positive",
                "sentiment_score": 0.7,
                "source": "Economic Times",
                "timestamp": datetime.now().isoformat()
            }
        ]
        mock_get_news.return_value = mock_news
        
        response = client.get("/api/v1/news/latest?limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert "news" in data
        assert len(data["news"]) > 0
    
    def test_health_check_integration(self):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

class TestWebSocketIntegration:
    """Test WebSocket integration"""
    
    def test_websocket_connection_flow(self):
        """Test complete WebSocket connection and data flow"""
        with client.websocket_connect("/ws") as websocket:
            # Test connection establishment
            data = websocket.receive_json()
            assert "type" in data
            
            # Test subscription
            websocket.send_json({
                "action": "subscribe",
                "symbol": "RELIANCE"
            })
            
            # Should receive subscription confirmation
            response = websocket.receive_json()
            assert "type" in response
    
    @patch('services.data_fetcher.DataFetcher.get_real_time_data')
    def test_websocket_real_time_data_flow(self, mock_get_realtime):
        """Test real-time data flow through WebSocket"""
        mock_data = {
            "symbol": "RELIANCE",
            "price": 2450.75,
            "change": 15.25,
            "change_percent": 0.63,
            "volume": 1500000,
            "timestamp": datetime.now().isoformat()
        }
        mock_get_realtime.return_value = mock_data
        
        with client.websocket_connect("/ws") as websocket:
            # Subscribe to symbol
            websocket.send_json({
                "action": "subscribe",
                "symbol": "RELIANCE"
            })
            
            # Should receive real-time data (mocked)
            # Note: In actual implementation, this would be pushed by the server
            # Here we're testing the connection capability
            response = websocket.receive_json()
            assert "type" in response

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    @patch('yfinance.download')
    @patch('services.prediction_service.PredictionService.get_prediction')
    def test_complete_prediction_workflow(self, mock_get_prediction, mock_yf_download):
        """Test complete workflow from data fetch to prediction API"""
        # Mock Yahoo Finance data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        mock_yf_data = pd.DataFrame({
            'Open': np.random.randn(100) * 10 + 2000,
            'High': np.random.randn(100) * 10 + 2020,
            'Low': np.random.randn(100) * 10 + 1980,
            'Close': np.random.randn(100) * 10 + 2000,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        mock_yf_download.return_value = mock_yf_data
        
        # Mock prediction service
        mock_prediction = {
            "symbol": "RELIANCE",
            "predicted_price": 2500.50,
            "confidence": 0.85,
            "direction": "up",
            "timestamp": datetime.now().isoformat(),
            "model_used": "XGBoost",
            "technical_indicators": {
                "rsi": 65.2,
                "macd": 12.5,
                "sma_20": 2480.0
            },
            "shap_values": {
                "feature_importance": {
                    "close": 0.3,
                    "volume": 0.2,
                    "rsi": 0.15
                }
            }
        }
        mock_get_prediction.return_value = mock_prediction
        
        # Test complete workflow
        symbol = "RELIANCE"
        
        # 1. Get historical data (via API)
        response = client.get(f"/api/v1/market/historical/{symbol}?days=30")
        # Should work even if mocked
        
        # 2. Get prediction (via API)
        response = client.get(f"/api/v1/prediction/{symbol}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == symbol
        assert "predicted_price" in data
        assert "confidence" in data
        assert "technical_indicators" in data
    
    @patch('services.news_service.NewsService.get_latest_news')
    @patch('services.news_service.NewsService.analyze_sentiment')
    def test_news_sentiment_workflow(self, mock_analyze_sentiment, mock_get_news):
        """Test complete news and sentiment analysis workflow"""
        # Mock news data
        mock_news = [
            {
                "title": "Reliance Industries reports strong Q4 results",
                "content": "Reliance Industries posted strong quarterly results with increased revenue...",
                "source": "Economic Times",
                "timestamp": datetime.now().isoformat(),
                "url": "https://example.com/news1"
            }
        ]
        mock_get_news.return_value = mock_news
        
        # Mock sentiment analysis
        mock_sentiment = {
            "sentiment": "positive",
            "sentiment_score": 0.8,
            "confidence": 0.9
        }
        mock_analyze_sentiment.return_value = mock_sentiment
        
        # Test workflow
        response = client.get("/api/v1/news/latest?limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert "news" in data
    
    def test_batch_prediction_workflow(self):
        """Test batch prediction workflow"""
        with patch('services.prediction_service.PredictionService.get_batch_predictions') as mock_batch:
            mock_predictions = {
                "RELIANCE": {
                    "predicted_price": 2500.50,
                    "confidence": 0.85,
                    "direction": "up"
                },
                "TCS": {
                    "predicted_price": 3200.75,
                    "confidence": 0.78,
                    "direction": "up"
                },
                "INFY": {
                    "predicted_price": 1450.25,
                    "confidence": 0.82,
                    "direction": "down"
                }
            }
            mock_batch.return_value = mock_predictions
            
            payload = {"symbols": ["RELIANCE", "TCS", "INFY"]}
            response = client.post("/api/v1/predictions/batch", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            for symbol in payload["symbols"]:
                assert symbol in data
                assert "predicted_price" in data[symbol]
                assert "confidence" in data[symbol]

class TestErrorHandlingIntegration:
    """Test error handling across the system"""
    
    def test_api_error_propagation(self):
        """Test how errors propagate through the API"""
        # Test with invalid symbol
        response = client.get("/api/v1/prediction/INVALID_SYMBOL")
        # Should handle gracefully
        assert response.status_code in [400, 404, 422, 500]
    
    def test_service_error_handling(self):
        """Test service layer error handling"""
        with patch('services.prediction_service.PredictionService.get_prediction') as mock_prediction:
            mock_prediction.side_effect = Exception("Service unavailable")
            
            response = client.get("/api/v1/prediction/RELIANCE")
            assert response.status_code == 500
            
            data = response.json()
            assert "error" in data
    
    def test_data_validation_errors(self):
        """Test data validation error handling"""
        # Test invalid batch prediction payload
        invalid_payload = {"invalid_field": "invalid_value"}
        response = client.post("/api/v1/predictions/batch", json=invalid_payload)
        
        assert response.status_code == 422
    
    def test_websocket_error_handling(self):
        """Test WebSocket error handling"""
        with client.websocket_connect("/ws") as websocket:
            # Send invalid message
            websocket.send_json({"invalid": "message"})
            
            # Should receive error response or handle gracefully
            try:
                response = websocket.receive_json()
                # If we get a response, it should indicate an error
                if "error" in response:
                    assert response["error"] is not None
            except Exception:
                # Connection might be closed due to invalid message
                pass

class TestPerformanceIntegration:
    """Test system performance under load"""
    
    @pytest.mark.slow
    def test_concurrent_api_requests(self):
        """Test handling of concurrent API requests"""
        import concurrent.futures
        import time
        
        def make_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Check results
        status_codes = [result[0] for result in results]
        response_times = [result[1] for result in results]
        
        # All requests should succeed
        assert all(code == 200 for code in status_codes)
        
        # Response times should be reasonable (< 5 seconds)
        assert all(time < 5.0 for time in response_times)
    
    @pytest.mark.slow
    def test_large_batch_prediction(self):
        """Test large batch prediction performance"""
        with patch('services.prediction_service.PredictionService.get_batch_predictions') as mock_batch:
            # Mock large batch response
            large_batch = {f"SYMBOL_{i}": {
                "predicted_price": 1000 + i,
                "confidence": 0.8,
                "direction": "up" if i % 2 == 0 else "down"
            } for i in range(100)}
            
            mock_batch.return_value = large_batch
            
            symbols = [f"SYMBOL_{i}" for i in range(100)]
            payload = {"symbols": symbols}
            
            start_time = time.time()
            response = client.post("/api/v1/predictions/batch", json=payload)
            end_time = time.time()
            
            assert response.status_code == 200
            assert (end_time - start_time) < 10.0  # Should complete within 10 seconds

# Pytest configuration
pytest_plugins = ["pytest_asyncio"]

# Run tests with: python -m pytest tests/test_integration.py -v
# Run with slow tests: python -m pytest tests/test_integration.py -v --runslow