#!/usr/bin/env python3
"""
Pytest configuration and fixtures
"""

import pytest
import asyncio
import tempfile
import os
import sys
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from main import app
from core.config import get_settings

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests as ML model tests"
    )

def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--runintegration",
        action="store_true",
        default=False,
        help="run integration tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers"""
    if config.getoption("--runslow"):
        # Don't skip slow tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
    
    if not config.getoption("--runintegration"):
        skip_integration = pytest.mark.skip(reason="need --runintegration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

# Global fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def mock_settings():
    """Mock application settings"""
    settings = Mock()
    settings.API_V1_STR = "/api/v1"
    settings.PROJECT_NAME = "Stock Prediction API"
    settings.VERSION = "1.0.0"
    settings.DEBUG = True
    settings.SECRET_KEY = "test-secret-key"
    settings.ALGORITHM = "HS256"
    settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30
    settings.YAHOO_FINANCE_API_KEY = "test-yahoo-key"
    settings.ANGEL_ONE_API_KEY = "test-angel-key"
    settings.NEWS_API_KEY = "test-news-key"
    settings.REDIS_URL = "redis://localhost:6379"
    settings.DATABASE_URL = "sqlite:///./test.db"
    return settings

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data
    base_price = 2000
    price_changes = np.random.randn(100) * 0.02  # 2% daily volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    data = pd.DataFrame({
        'open': [p * (1 + np.random.randn() * 0.005) for p in prices],
        'high': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 1000000, 100)
    }, index=dates)
    
    return data

@pytest.fixture
def sample_news_data():
    """Generate sample news data for testing"""
    return [
        {
            "title": "Reliance Industries reports strong Q4 results",
            "content": "Reliance Industries posted strong quarterly results with increased revenue and profit margins...",
            "sentiment": "positive",
            "sentiment_score": 0.8,
            "source": "Economic Times",
            "timestamp": datetime.now().isoformat(),
            "url": "https://example.com/news1",
            "relevance_score": 0.9
        },
        {
            "title": "Market volatility expected due to global factors",
            "content": "Global market conditions may impact Indian stock markets in the coming weeks...",
            "sentiment": "negative",
            "sentiment_score": -0.3,
            "source": "Business Standard",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "url": "https://example.com/news2",
            "relevance_score": 0.7
        },
        {
            "title": "Tech stocks show mixed performance",
            "content": "Technology sector shows mixed signals with some stocks gaining while others decline...",
            "sentiment": "neutral",
            "sentiment_score": 0.1,
            "source": "Mint",
            "timestamp": (datetime.now() - timedelta(hours=4)).isoformat(),
            "url": "https://example.com/news3",
            "relevance_score": 0.6
        }
    ]

@pytest.fixture
def sample_prediction_data():
    """Generate sample prediction data for testing"""
    return {
        "symbol": "RELIANCE",
        "predicted_price": 2500.50,
        "confidence": 0.85,
        "direction": "up",
        "timestamp": datetime.now().isoformat(),
        "model_used": "XGBoost",
        "prediction_horizon": "1d",
        "technical_indicators": {
            "rsi": 65.2,
            "macd": 12.5,
            "macd_signal": 10.2,
            "sma_20": 2480.0,
            "ema_12": 2485.5,
            "bb_upper": 2520.0,
            "bb_lower": 2440.0,
            "stoch_k": 75.3,
            "stoch_d": 72.1
        },
        "shap_values": {
            "feature_importance": {
                "close": 0.3,
                "volume": 0.2,
                "rsi": 0.15,
                "macd": 0.12,
                "sma_20": 0.1,
                "ema_12": 0.08,
                "bb_upper": 0.05
            }
        },
        "risk_metrics": {
            "volatility": 0.025,
            "var_95": -0.045,
            "max_drawdown": -0.08
        }
    }

@pytest.fixture
def sample_technical_indicators():
    """Generate sample technical indicators data"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    return pd.DataFrame({
        'sma_20': np.random.randn(100) * 10 + 2000,
        'ema_12': np.random.randn(100) * 10 + 2005,
        'ema_26': np.random.randn(100) * 10 + 1995,
        'rsi': np.random.uniform(20, 80, 100),
        'macd': np.random.randn(100) * 5,
        'macd_signal': np.random.randn(100) * 3,
        'macd_histogram': np.random.randn(100) * 2,
        'bb_upper': np.random.randn(100) * 10 + 2050,
        'bb_middle': np.random.randn(100) * 10 + 2000,
        'bb_lower': np.random.randn(100) * 10 + 1950,
        'stoch_k': np.random.uniform(0, 100, 100),
        'stoch_d': np.random.uniform(0, 100, 100),
        'williams_r': np.random.uniform(-100, 0, 100),
        'atr': np.random.uniform(10, 50, 100),
        'adx': np.random.uniform(0, 100, 100)
    }, index=dates)

@pytest.fixture
def mock_ml_features():
    """Generate mock ML features for testing"""
    np.random.seed(42)
    return {
        'X_train': np.random.randn(100, 15),
        'y_train': np.random.randn(100) * 100 + 2000,
        'X_test': np.random.randn(20, 15),
        'y_test': np.random.randn(20) * 100 + 2000
    }

@pytest.fixture
def temp_model_file():
    """Create temporary file for model testing"""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        yield tmp.name
    # Cleanup
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)

@pytest.fixture
def mock_websocket_message():
    """Generate mock WebSocket message"""
    return {
        "type": "real_time_data",
        "symbol": "RELIANCE",
        "data": {
            "price": 2450.75,
            "change": 15.25,
            "change_percent": 0.63,
            "volume": 1500000,
            "bid": 2450.50,
            "ask": 2451.00,
            "timestamp": datetime.now().isoformat()
        }
    }

@pytest.fixture
def mock_batch_symbols():
    """Generate mock batch symbols for testing"""
    return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

@pytest.fixture
def mock_error_response():
    """Generate mock error response"""
    return {
        "error": "Service temporarily unavailable",
        "error_code": "SERVICE_ERROR",
        "timestamp": datetime.now().isoformat(),
        "details": "The requested service is currently experiencing issues"
    }

# Database fixtures
@pytest.fixture
def mock_database_session():
    """Mock database session"""
    session = Mock()
    session.query.return_value = session
    session.filter.return_value = session
    session.first.return_value = None
    session.all.return_value = []
    session.add.return_value = None
    session.commit.return_value = None
    session.rollback.return_value = None
    return session

# API mocking fixtures
@pytest.fixture
def mock_yahoo_finance():
    """Mock Yahoo Finance API responses"""
    with patch('yfinance.download') as mock_download:
        yield mock_download

@pytest.fixture
def mock_angel_one_api():
    """Mock Angel One API responses"""
    with patch('services.angel_one_service.AngelOneService') as mock_service:
        yield mock_service

@pytest.fixture
def mock_news_api():
    """Mock News API responses"""
    with patch('services.news_service.NewsService') as mock_service:
        yield mock_service

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    with patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_client.delete.return_value = 1
        mock_client.exists.return_value = False
        mock_redis.return_value = mock_client
        yield mock_client

# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance testing"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test"""
    test_files = []
    
    def add_file(filepath):
        test_files.append(filepath)
    
    yield add_file
    
    # Cleanup
    for filepath in test_files:
        if os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except Exception:
                pass  # Ignore cleanup errors

# Logging fixtures
@pytest.fixture
def capture_logs(caplog):
    """Capture logs for testing"""
    import logging
    caplog.set_level(logging.INFO)
    return caplog

# Environment fixtures
@pytest.fixture
def test_env_vars():
    """Set test environment variables"""
    test_vars = {
        'ENVIRONMENT': 'test',
        'DEBUG': 'true',
        'SECRET_KEY': 'test-secret-key',
        'DATABASE_URL': 'sqlite:///./test.db',
        'REDIS_URL': 'redis://localhost:6379/1'
    }
    
    # Set environment variables
    original_vars = {}
    for key, value in test_vars.items():
        original_vars[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_vars
    
    # Restore original environment variables
    for key, original_value in original_vars.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

# Custom assertions
class CustomAssertions:
    """Custom assertion helpers"""
    
    @staticmethod
    def assert_valid_prediction(prediction_data):
        """Assert that prediction data is valid"""
        required_fields = ['symbol', 'predicted_price', 'confidence', 'direction']
        for field in required_fields:
            assert field in prediction_data, f"Missing required field: {field}"
        
        assert isinstance(prediction_data['predicted_price'], (int, float))
        assert 0 <= prediction_data['confidence'] <= 1
        assert prediction_data['direction'] in ['up', 'down', 'neutral']
    
    @staticmethod
    def assert_valid_ohlcv_data(ohlcv_data):
        """Assert that OHLCV data is valid"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in ohlcv_data.columns, f"Missing required column: {col}"
        
        # Check data relationships
        assert (ohlcv_data['high'] >= ohlcv_data['low']).all()
        assert (ohlcv_data['high'] >= ohlcv_data['open']).all()
        assert (ohlcv_data['high'] >= ohlcv_data['close']).all()
        assert (ohlcv_data['low'] <= ohlcv_data['open']).all()
        assert (ohlcv_data['low'] <= ohlcv_data['close']).all()
        assert (ohlcv_data['volume'] >= 0).all()
    
    @staticmethod
    def assert_valid_technical_indicators(indicators_data):
        """Assert that technical indicators are valid"""
        if 'rsi' in indicators_data.columns:
            rsi_values = indicators_data['rsi'].dropna()
            assert (rsi_values >= 0).all() and (rsi_values <= 100).all()
        
        if 'stoch_k' in indicators_data.columns:
            stoch_values = indicators_data['stoch_k'].dropna()
            assert (stoch_values >= 0).all() and (stoch_values <= 100).all()
        
        if 'williams_r' in indicators_data.columns:
            williams_values = indicators_data['williams_r'].dropna()
            assert (williams_values >= -100).all() and (williams_values <= 0).all()

@pytest.fixture
def custom_assertions():
    """Provide custom assertion helpers"""
    return CustomAssertions()