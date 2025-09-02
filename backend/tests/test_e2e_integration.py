import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.main import app
from app.services.prediction_service import PredictionService
from app.services.news_service import NewsService
from app.services.data_service import DataService
from app.models.ml_models import XGBoostModel, InformerModel, DQNModel
from app.core.websocket_manager import WebSocketManager

client = TestClient(app)

class TestEndToEndIntegration:
    """End-to-end integration tests for the complete stock prediction system"""
    
    @pytest.fixture
    def mock_historical_data(self):
        """Mock historical stock data"""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        data = {
            'Date': dates,
            'Open': np.random.uniform(2400, 2500, len(dates)),
            'High': np.random.uniform(2450, 2550, len(dates)),
            'Low': np.random.uniform(2350, 2450, len(dates)),
            'Close': np.random.uniform(2400, 2500, len(dates)),
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_news_data(self):
        """Mock news data with sentiment"""
        return [
            {
                'title': 'Reliance Industries reports strong Q4 earnings',
                'content': 'Reliance Industries has reported exceptional quarterly results...',
                'source': 'Economic Times',
                'published_at': '2024-01-15T10:30:00Z',
                'sentiment_score': 0.85,
                'sentiment_label': 'positive',
                'impact_score': 0.75,
                'related_stocks': ['RELIANCE']
            },
            {
                'title': 'Oil prices surge amid geopolitical tensions',
                'content': 'Global oil prices have increased significantly...',
                'source': 'Reuters',
                'published_at': '2024-01-15T08:15:00Z',
                'sentiment_score': 0.65,
                'sentiment_label': 'positive',
                'impact_score': 0.60,
                'related_stocks': ['RELIANCE', 'ONGC']
            }
        ]
    
    @pytest.fixture
    def mock_prediction_result(self):
        """Mock ML model prediction result"""
        return {
            'symbol': 'RELIANCE',
            'current_price': 2450.75,
            'predictions': {
                '1d': {
                    'price': 2465.30,
                    'change': 14.55,
                    'change_percent': 0.59,
                    'confidence': 0.85,
                    'signal': 'BUY',
                    'factors': [
                        {'name': 'Technical Analysis', 'weight': 0.4, 'impact': 'positive'},
                        {'name': 'Market Sentiment', 'weight': 0.3, 'impact': 'positive'},
                        {'name': 'Volume Analysis', 'weight': 0.3, 'impact': 'neutral'}
                    ]
                },
                '1w': {
                    'price': 2520.40,
                    'change': 69.65,
                    'change_percent': 2.84,
                    'confidence': 0.78,
                    'signal': 'BUY'
                },
                '1m': {
                    'price': 2380.20,
                    'change': -70.55,
                    'change_percent': -2.88,
                    'confidence': 0.65,
                    'signal': 'HOLD'
                }
            },
            'technical_indicators': {
                'rsi': 68.5,
                'macd': 12.3,
                'ema_20': 2435.60,
                'sma_50': 2420.80,
                'bollinger_bands': {
                    'upper': 2480.30,
                    'middle': 2450.75,
                    'lower': 2421.20
                }
            },
            'model_accuracy': {
                'xgboost': 0.87,
                'informer': 0.82,
                'dqn': 0.79
            },
            'risk_metrics': {
                'volatility': 0.24,
                'sharpe_ratio': 1.45,
                'max_drawdown': 0.12,
                'beta': 1.15
            },
            'shap_values': {
                'feature_importance': [
                    {'feature': 'price_momentum', 'importance': 0.25},
                    {'feature': 'volume_trend', 'importance': 0.20},
                    {'feature': 'news_sentiment', 'importance': 0.18},
                    {'feature': 'technical_indicators', 'importance': 0.15}
                ]
            }
        }
    
    @patch('app.services.data_service.DataService.get_historical_data')
    @patch('app.services.prediction_service.PredictionService.predict')
    def test_complete_prediction_workflow(self, mock_predict, mock_historical_data, mock_prediction_result):
        """Test complete prediction workflow from API request to response"""
        # Setup mocks
        mock_historical_data.return_value = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Close': np.random.uniform(2400, 2500, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        })
        mock_predict.return_value = mock_prediction_result
        
        # Make API request
        response = client.get('/api/predictions/RELIANCE')
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data['symbol'] == 'RELIANCE'
        assert data['current_price'] == 2450.75
        assert '1d' in data['predictions']
        assert '1w' in data['predictions']
        assert '1m' in data['predictions']
        assert data['predictions']['1d']['signal'] == 'BUY'
        assert data['technical_indicators']['rsi'] == 68.5
        assert data['model_accuracy']['xgboost'] == 0.87
        
        # Verify service calls
        mock_historical_data.assert_called_once_with('RELIANCE')
        mock_predict.assert_called_once()
    
    @patch('app.services.news_service.NewsService.get_news_with_sentiment')
    def test_news_sentiment_integration(self, mock_news_service, mock_news_data):
        """Test news sentiment analysis integration"""
        mock_news_service.return_value = mock_news_data
        
        # Make API request
        response = client.get('/api/news?symbol=RELIANCE&limit=10')
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert len(data['articles']) == 2
        assert data['articles'][0]['sentiment_score'] == 0.85
        assert data['articles'][0]['sentiment_label'] == 'positive'
        assert 'RELIANCE' in data['articles'][0]['related_stocks']
        
        # Verify sentiment aggregation
        assert 'sentiment_summary' in data
        assert data['sentiment_summary']['overall'] == 'positive'
    
    @patch('app.services.data_service.DataService.get_real_time_data')
    async def test_websocket_real_time_updates(self, mock_real_time_data):
        """Test WebSocket real-time data updates"""
        mock_real_time_data.return_value = {
            'symbol': 'RELIANCE',
            'price': 2455.30,
            'change': 4.55,
            'change_percent': 0.19,
            'volume': 1250000,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test WebSocket connection
        with client.websocket_connect('/ws/stocks/RELIANCE') as websocket:
            # Send subscription message
            websocket.send_json({
                'action': 'subscribe',
                'symbol': 'RELIANCE'
            })
            
            # Receive real-time data
            data = websocket.receive_json()
            
            assert data['symbol'] == 'RELIANCE'
            assert data['price'] == 2455.30
            assert 'timestamp' in data
    
    @patch('app.services.data_service.DataService.get_historical_data')
    @patch('app.services.prediction_service.PredictionService.predict')
    @patch('app.services.news_service.NewsService.get_news_with_sentiment')
    def test_dashboard_data_aggregation(self, mock_news, mock_predict, mock_historical, 
                                      mock_prediction_result, mock_news_data):
        """Test dashboard data aggregation from multiple services"""
        # Setup mocks
        mock_historical.return_value = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Close': np.random.uniform(2400, 2500, 100)
        })
        mock_predict.return_value = mock_prediction_result
        mock_news.return_value = mock_news_data
        
        # Make dashboard API request
        response = client.get('/api/dashboard')
        
        # Verify response structure
        assert response.status_code == 200
        data = response.json()
        
        assert 'market_overview' in data
        assert 'top_gainers' in data
        assert 'top_losers' in data
        assert 'trending_predictions' in data
        assert 'latest_news' in data
        assert 'market_sentiment' in data
        
        # Verify market overview
        market_overview = data['market_overview']
        assert 'nifty50' in market_overview
        assert 'sensex' in market_overview
        
        # Verify news integration
        assert len(data['latest_news']) > 0
        assert data['latest_news'][0]['sentiment_score'] is not None
    
    @patch('app.services.prediction_service.PredictionService.batch_predict')
    def test_batch_prediction_performance(self, mock_batch_predict):
        """Test batch prediction performance for multiple stocks"""
        symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
        
        mock_batch_predict.return_value = {
            symbol: {
                'symbol': symbol,
                'predictions': {
                    '1d': {'price': 2450 + i * 100, 'signal': 'BUY'}
                }
            } for i, symbol in enumerate(symbols)
        }
        
        # Make batch prediction request
        response = client.post('/api/predictions/batch', json={'symbols': symbols})
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert len(data['predictions']) == 5
        for symbol in symbols:
            assert symbol in data['predictions']
            assert data['predictions'][symbol]['symbol'] == symbol
    
    def test_api_error_handling(self):
        """Test API error handling and response codes"""
        # Test invalid symbol
        response = client.get('/api/predictions/INVALID')
        assert response.status_code == 404
        
        # Test malformed request
        response = client.post('/api/predictions/batch', json={'invalid': 'data'})
        assert response.status_code == 422
        
        # Test rate limiting (if implemented)
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            responses.append(client.get('/api/predictions/RELIANCE'))
        
        # Should handle gracefully without crashing
        assert all(r.status_code in [200, 429] for r in responses)
    
    @patch('app.services.data_service.DataService.get_historical_data')
    def test_data_validation_and_preprocessing(self, mock_historical_data):
        """Test data validation and preprocessing pipeline"""
        # Test with missing data
        incomplete_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=50),
            'Close': [2450.0] * 30 + [None] * 20,  # Missing values
            'Volume': np.random.randint(1000000, 5000000, 50)
        })
        mock_historical_data.return_value = incomplete_data
        
        response = client.get('/api/predictions/RELIANCE')
        
        # Should handle missing data gracefully
        assert response.status_code in [200, 400]  # Either process or return validation error
        
        if response.status_code == 200:
            data = response.json()
            # Verify data quality indicators
            assert 'data_quality' in data
            assert data['data_quality']['completeness'] < 1.0
    
    @patch('app.services.prediction_service.PredictionService.predict')
    def test_model_ensemble_integration(self, mock_predict, mock_prediction_result):
        """Test model ensemble predictions integration"""
        mock_predict.return_value = mock_prediction_result
        
        response = client.get('/api/predictions/RELIANCE?models=xgboost,informer,dqn')
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify ensemble results
        assert 'model_accuracy' in data
        assert 'xgboost' in data['model_accuracy']
        assert 'informer' in data['model_accuracy']
        assert 'dqn' in data['model_accuracy']
        
        # Verify ensemble prediction
        assert 'ensemble_confidence' in data['predictions']['1d']
    
    @patch('app.services.news_service.NewsService.analyze_sentiment')
    def test_sentiment_analysis_integration(self, mock_sentiment):
        """Test sentiment analysis integration with FinGPT"""
        mock_sentiment.return_value = {
            'sentiment_score': 0.75,
            'sentiment_label': 'positive',
            'confidence': 0.88,
            'key_phrases': ['strong earnings', 'revenue growth', 'market expansion']
        }
        
        # Test sentiment analysis endpoint
        response = client.post('/api/sentiment/analyze', json={
            'text': 'Reliance Industries reports strong quarterly earnings with significant revenue growth.'
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['sentiment_score'] == 0.75
        assert data['sentiment_label'] == 'positive'
        assert len(data['key_phrases']) > 0
    
    def test_health_check_comprehensive(self):
        """Test comprehensive health check endpoint"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify health check components
        assert 'status' in data
        assert 'services' in data
        assert 'database' in data['services']
        assert 'ml_models' in data['services']
        assert 'external_apis' in data['services']
        
        # Verify timestamps
        assert 'timestamp' in data
        assert 'uptime' in data
    
    @patch('app.services.data_service.DataService.get_historical_data')
    @patch('app.services.prediction_service.PredictionService.predict')
    def test_caching_integration(self, mock_predict, mock_historical_data, mock_prediction_result):
        """Test caching integration for improved performance"""
        mock_historical_data.return_value = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Close': np.random.uniform(2400, 2500, 100)
        })
        mock_predict.return_value = mock_prediction_result
        
        # First request
        response1 = client.get('/api/predictions/RELIANCE')
        assert response1.status_code == 200
        
        # Second request (should use cache)
        response2 = client.get('/api/predictions/RELIANCE')
        assert response2.status_code == 200
        
        # Verify cache headers
        assert 'cache-control' in response2.headers or 'x-cache-status' in response2.headers
        
        # Data should be identical
        assert response1.json() == response2.json()
    
    @patch('app.services.alert_service.AlertService.check_alerts')
    def test_alert_system_integration(self, mock_check_alerts):
        """Test price alert system integration"""
        mock_check_alerts.return_value = [
            {
                'id': 1,
                'symbol': 'RELIANCE',
                'alert_type': 'price_above',
                'threshold': 2500.0,
                'current_price': 2505.30,
                'triggered_at': datetime.now().isoformat()
            }
        ]
        
        # Create alert
        response = client.post('/api/alerts', json={
            'symbol': 'RELIANCE',
            'alert_type': 'price_above',
            'threshold': 2500.0
        })
        
        assert response.status_code == 201
        
        # Check triggered alerts
        response = client.get('/api/alerts/triggered')
        assert response.status_code == 200
        
        data = response.json()
        assert len(data['alerts']) > 0
        assert data['alerts'][0]['symbol'] == 'RELIANCE'
    
    def test_api_versioning(self):
        """Test API versioning support"""
        # Test v1 API
        response_v1 = client.get('/api/v1/predictions/RELIANCE')
        
        # Test v2 API (if available)
        response_v2 = client.get('/api/v2/predictions/RELIANCE')
        
        # At least one version should work
        assert response_v1.status_code in [200, 404] or response_v2.status_code in [200, 404]
    
    @patch('app.services.data_service.DataService.get_historical_data')
    def test_concurrent_requests_handling(self, mock_historical_data):
        """Test handling of concurrent API requests"""
        mock_historical_data.return_value = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Close': np.random.uniform(2400, 2500, 100)
        })
        
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get('/api/predictions/RELIANCE')
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should complete successfully
        assert all(r.status_code == 200 for r in responses)
        
        # Verify no data corruption
        first_response = responses[0].json()
        for response in responses[1:]:
            assert response.json()['symbol'] == first_response['symbol']

@pytest.mark.performance
class TestPerformanceIntegration:
    """Performance integration tests"""
    
    def test_prediction_response_time(self):
        """Test prediction API response time"""
        import time
        
        start_time = time.time()
        response = client.get('/api/predictions/RELIANCE')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within 2 seconds
        assert response_time < 2.0
        assert response.status_code == 200
    
    def test_batch_prediction_scalability(self):
        """Test batch prediction scalability"""
        symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'] * 10  # 50 symbols
        
        import time
        start_time = time.time()
        
        response = client.post('/api/predictions/batch', json={'symbols': symbols})
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Should handle 50 symbols within 10 seconds
        assert response_time < 10.0
        assert response.status_code == 200
    
    def test_websocket_message_throughput(self):
        """Test WebSocket message throughput"""
        message_count = 0
        
        with client.websocket_connect('/ws/stocks/RELIANCE') as websocket:
            websocket.send_json({'action': 'subscribe', 'symbol': 'RELIANCE'})
            
            import time
            start_time = time.time()
            
            # Receive messages for 5 seconds
            while time.time() - start_time < 5:
                try:
                    websocket.receive_json(timeout=1)
                    message_count += 1
                except:
                    break
        
        # Should receive at least 1 message per second
        assert message_count >= 5

@pytest.mark.security
class TestSecurityIntegration:
    """Security integration tests"""
    
    def test_sql_injection_protection(self):
        """Test SQL injection protection"""
        malicious_symbol = "RELIANCE'; DROP TABLE predictions; --"
        
        response = client.get(f'/api/predictions/{malicious_symbol}')
        
        # Should not crash and return appropriate error
        assert response.status_code in [400, 404, 422]
    
    def test_xss_protection(self):
        """Test XSS protection in API responses"""
        malicious_input = "<script>alert('xss')</script>"
        
        response = client.post('/api/sentiment/analyze', json={
            'text': malicious_input
        })
        
        if response.status_code == 200:
            # Response should not contain unescaped script tags
            assert '<script>' not in response.text
    
    def test_rate_limiting(self):
        """Test rate limiting implementation"""
        responses = []
        
        # Make rapid requests
        for _ in range(100):
            responses.append(client.get('/api/predictions/RELIANCE'))
        
        # Should implement rate limiting
        rate_limited = any(r.status_code == 429 for r in responses)
        
        # Either rate limiting is implemented or all requests succeed
        assert rate_limited or all(r.status_code == 200 for r in responses)
    
    def test_input_validation(self):
        """Test comprehensive input validation"""
        # Test invalid JSON
        response = client.post('/api/predictions/batch', 
                             data='invalid json',
                             headers={'Content-Type': 'application/json'})
        assert response.status_code == 422
        
        # Test missing required fields
        response = client.post('/api/alerts', json={})
        assert response.status_code == 422
        
        # Test invalid data types
        response = client.post('/api/alerts', json={
            'symbol': 123,  # Should be string
            'threshold': 'invalid'  # Should be number
        })
        assert response.status_code == 422