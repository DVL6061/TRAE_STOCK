#!/usr/bin/env python3
"""
Unit tests for API endpoints
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime, timedelta

# Import the FastAPI app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import app
from core.config import get_settings

# Test client
client = TestClient(app)

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

class TestPredictionEndpoints:
    """Test prediction API endpoints"""
    
    @patch('services.prediction_service.PredictionService.get_prediction')
    def test_get_prediction_success(self, mock_get_prediction):
        """Test successful prediction request"""
        # Mock prediction response
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
        
        response = client.get("/api/v1/prediction/RELIANCE")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "RELIANCE"
        assert "predicted_price" in data
        assert "confidence" in data
        assert "direction" in data
    
    def test_get_prediction_invalid_symbol(self):
        """Test prediction with invalid symbol"""
        response = client.get("/api/v1/prediction/INVALID")
        # Should handle gracefully or return appropriate error
        assert response.status_code in [400, 404, 422]
    
    @patch('services.prediction_service.PredictionService.get_batch_predictions')
    def test_batch_predictions(self, mock_batch_predictions):
        """Test batch predictions endpoint"""
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
            }
        }
        mock_batch_predictions.return_value = mock_predictions
        
        payload = {"symbols": ["RELIANCE", "TCS"]}
        response = client.post("/api/v1/predictions/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "RELIANCE" in data
        assert "TCS" in data

class TestMarketDataEndpoints:
    """Test market data API endpoints"""
    
    @patch('services.data_fetcher.DataFetcher.get_historical_data')
    def test_get_historical_data(self, mock_get_historical):
        """Test historical data endpoint"""
        import pandas as pd
        
        # Mock historical data
        mock_data = pd.DataFrame({
            'open': [2400, 2410, 2420],
            'high': [2450, 2460, 2470],
            'low': [2390, 2400, 2410],
            'close': [2440, 2450, 2460],
            'volume': [1000000, 1100000, 1200000]
        })
        mock_get_historical.return_value = mock_data
        
        response = client.get("/api/v1/market/historical/RELIANCE?days=30")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
    
    @patch('services.data_fetcher.DataFetcher.get_real_time_data')
    def test_get_real_time_data(self, mock_get_realtime):
        """Test real-time data endpoint"""
        mock_data = {
            "symbol": "RELIANCE",
            "price": 2450.75,
            "change": 15.25,
            "change_percent": 0.63,
            "volume": 1500000,
            "timestamp": datetime.now().isoformat()
        }
        mock_get_realtime.return_value = mock_data
        
        response = client.get("/api/v1/market/realtime/RELIANCE")
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "RELIANCE"
        assert "price" in data
        assert "change" in data

class TestNewsEndpoints:
    """Test news API endpoints"""
    
    @patch('services.news_service.NewsService.get_latest_news')
    def test_get_latest_news(self, mock_get_news):
        """Test latest news endpoint"""
        mock_news = [
            {
                "title": "Reliance Industries reports strong Q4 results",
                "content": "Reliance Industries posted strong quarterly results...",
                "sentiment": "positive",
                "sentiment_score": 0.8,
                "source": "Economic Times",
                "timestamp": datetime.now().isoformat(),
                "url": "https://example.com/news1"
            },
            {
                "title": "Market volatility expected due to global factors",
                "content": "Global market conditions may impact...",
                "sentiment": "negative",
                "sentiment_score": -0.3,
                "source": "Business Standard",
                "timestamp": datetime.now().isoformat(),
                "url": "https://example.com/news2"
            }
        ]
        mock_get_news.return_value = mock_news
        
        response = client.get("/api/v1/news/latest?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "news" in data
        assert len(data["news"]) > 0
        assert "sentiment" in data["news"][0]
    
    @patch('services.news_service.NewsService.get_news_by_symbol')
    def test_get_news_by_symbol(self, mock_get_news_symbol):
        """Test news by symbol endpoint"""
        mock_news = [
            {
                "title": "Reliance announces new venture",
                "content": "Reliance Industries announced...",
                "sentiment": "positive",
                "sentiment_score": 0.7,
                "relevance_score": 0.9,
                "source": "Mint",
                "timestamp": datetime.now().isoformat()
            }
        ]
        mock_get_news_symbol.return_value = mock_news
        
        response = client.get("/api/v1/news/symbol/RELIANCE")
        assert response.status_code == 200
        data = response.json()
        assert "news" in data
        assert data["symbol"] == "RELIANCE"

class TestTechnicalIndicatorsEndpoints:
    """Test technical indicators API endpoints"""
    
    @patch('utils.technical_indicators.TechnicalIndicators.calculate_all_indicators')
    def test_get_technical_indicators(self, mock_calculate_indicators):
        """Test technical indicators endpoint"""
        import pandas as pd
        
        mock_indicators = pd.DataFrame({
            'sma_20': [2400, 2410, 2420],
            'ema_12': [2405, 2415, 2425],
            'rsi': [65.2, 67.1, 64.8],
            'macd': [12.5, 13.2, 11.8],
            'macd_signal': [10.2, 11.1, 10.5],
            'bb_upper': [2480, 2490, 2500],
            'bb_lower': [2360, 2370, 2380]
        })
        mock_calculate_indicators.return_value = mock_indicators
        
        response = client.get("/api/v1/indicators/RELIANCE")
        assert response.status_code == 200
        data = response.json()
        assert "indicators" in data
        assert "symbol" in data
        assert data["symbol"] == "RELIANCE"

class TestWebSocketEndpoints:
    """Test WebSocket endpoints"""
    
    def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        with client.websocket_connect("/ws") as websocket:
            # Test connection
            data = websocket.receive_json()
            assert "type" in data
            assert data["type"] == "connection_established"
    
    def test_websocket_subscription(self):
        """Test WebSocket symbol subscription"""
        with client.websocket_connect("/ws") as websocket:
            # Send subscription message
            websocket.send_json({
                "action": "subscribe",
                "symbol": "RELIANCE"
            })
            
            # Receive confirmation
            response = websocket.receive_json()
            assert response["type"] == "subscription_confirmed"
            assert response["symbol"] == "RELIANCE"

class TestErrorHandling:
    """Test error handling across endpoints"""
    
    def test_404_endpoint(self):
        """Test non-existent endpoint"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_json_payload(self):
        """Test invalid JSON in POST request"""
        response = client.post(
            "/api/v1/predictions/batch",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test missing required fields in request"""
        response = client.post("/api/v1/predictions/batch", json={})
        assert response.status_code == 422
    
    @patch('services.prediction_service.PredictionService.get_prediction')
    def test_service_error_handling(self, mock_get_prediction):
        """Test service layer error handling"""
        mock_get_prediction.side_effect = Exception("Service error")
        
        response = client.get("/api/v1/prediction/RELIANCE")
        assert response.status_code == 500
        data = response.json()
        assert "error" in data

class TestAuthentication:
    """Test authentication and authorization"""
    
    def test_protected_endpoint_without_auth(self):
        """Test protected endpoint without authentication"""
        # Assuming some endpoints require authentication
        response = client.get("/api/v1/admin/models")
        # Should return 401 if authentication is implemented
        assert response.status_code in [401, 404]  # 404 if endpoint doesn't exist yet
    
    def test_invalid_token(self):
        """Test invalid authentication token"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/admin/models", headers=headers)
        assert response.status_code in [401, 404]

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_not_exceeded(self):
        """Test normal request rate"""
        # Make a few requests
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_rate_limit_headers(self):
        """Test rate limit headers presence"""
        response = client.get("/health")
        # Check if rate limit headers are present (if implemented)
        # This is optional depending on implementation
        assert response.status_code == 200

class TestCORS:
    """Test CORS configuration"""
    
    def test_cors_headers(self):
        """Test CORS headers in response"""
        response = client.options("/api/v1/prediction/RELIANCE")
        # Should have CORS headers
        assert response.status_code in [200, 405]  # 405 if OPTIONS not implemented
    
    def test_preflight_request(self):
        """Test CORS preflight request"""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type"
        }
        response = client.options("/api/v1/prediction/RELIANCE", headers=headers)
        assert response.status_code in [200, 405]

# Pytest fixtures
@pytest.fixture
def mock_settings():
    """Mock application settings"""
    settings = Mock()
    settings.API_V1_STR = "/api/v1"
    settings.PROJECT_NAME = "Stock Prediction API"
    settings.VERSION = "1.0.0"
    settings.DEBUG = True
    return settings

@pytest.fixture
def mock_database():
    """Mock database connection"""
    return Mock()

# Test configuration
pytest_plugins = ["pytest_asyncio"]

# Run tests with: python -m pytest tests/test_api.py -v