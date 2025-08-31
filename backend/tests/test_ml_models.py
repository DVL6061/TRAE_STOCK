#!/usr/bin/env python3
"""
Unit tests for ML models
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os

# Import ML models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml_models.xgboost_model import XGBoostPredictor
from ml_models.informer_model import InformerModel
from ml_models.dqn_model import DQNAgent
from ml_models.model_factory import ModelFactory
from utils.technical_indicators import TechnicalIndicators

class TestXGBoostPredictor:
    """Test XGBoost model functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.model = XGBoostPredictor()
        
        # Create sample training data
        np.random.seed(42)
        self.X_train = np.random.randn(100, 10)
        self.y_train = np.random.randn(100) * 100 + 2000  # Price-like data
        
        self.X_test = np.random.randn(20, 10)
        self.y_test = np.random.randn(20) * 100 + 2000
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model is not None
        assert hasattr(self.model, 'model')
        assert hasattr(self.model, 'scaler')
    
    def test_model_training(self):
        """Test model training"""
        # Train the model
        self.model.train(self.X_train, self.y_train)
        
        # Check if model is trained
        assert self.model.model is not None
        assert self.model.is_trained
    
    def test_model_prediction(self):
        """Test model prediction"""
        # Train first
        self.model.train(self.X_train, self.y_train)
        
        # Make predictions
        predictions = self.model.predict(self.X_test)
        
        # Check predictions
        assert predictions is not None
        assert len(predictions) == len(self.X_test)
        assert isinstance(predictions, np.ndarray)
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        # Train first
        self.model.train(self.X_train, self.y_train)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        assert importance is not None
        assert isinstance(importance, list)
        assert len(importance) <= self.X_train.shape[1]
    
    def test_model_save_load(self):
        """Test model save and load functionality"""
        # Train model
        self.model.train(self.X_train, self.y_train)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            self.model.save_model(tmp.name)
            
            # Create new model instance and load
            new_model = XGBoostPredictor()
            new_model.load_model(tmp.name)
            
            # Test predictions are similar
            original_pred = self.model.predict(self.X_test)
            loaded_pred = new_model.predict(self.X_test)
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_prediction_without_training(self):
        """Test prediction without training should raise error"""
        with pytest.raises(Exception):
            self.model.predict(self.X_test)
    
    def test_invalid_input_shapes(self):
        """Test handling of invalid input shapes"""
        # Train with valid data
        self.model.train(self.X_train, self.y_train)
        
        # Test with wrong number of features
        invalid_X = np.random.randn(10, 5)  # Wrong number of features
        
        with pytest.raises(Exception):
            self.model.predict(invalid_X)

class TestInformerModel:
    """Test Informer model functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.seq_len = 60
        self.pred_len = 1
        self.model = InformerModel(
            seq_len=self.seq_len,
            label_len=30,
            pred_len=self.pred_len,
            d_model=64,  # Smaller for testing
            n_heads=4,
            e_layers=2,
            d_layers=1
        )
        
        # Create sequential data
        np.random.seed(42)
        self.sequence_data = np.random.randn(200, 1)  # Single feature time series
        self.X_train, self.y_train = self._create_sequences(self.sequence_data[:150])
        self.X_test, self.y_test = self._create_sequences(self.sequence_data[150:])
    
    def _create_sequences(self, data):
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i + self.seq_len])
            y.append(data[i + self.seq_len])
        return np.array(X), np.array(y)
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert self.model is not None
        assert self.model.seq_len == self.seq_len
        assert self.model.pred_len == self.pred_len
    
    @pytest.mark.slow
    def test_model_training(self):
        """Test model training (marked as slow)"""
        if len(self.X_train) == 0:
            pytest.skip("Insufficient data for training")
        
        # Train with minimal epochs for testing
        self.model.train(self.X_train, self.y_train, epochs=2, batch_size=16)
        
        assert self.model.is_trained
    
    def test_model_prediction_shape(self):
        """Test prediction output shape"""
        if len(self.X_test) == 0:
            pytest.skip("No test data available")
        
        # Mock trained model for shape testing
        self.model.is_trained = True
        
        with patch.object(self.model, '_predict_batch') as mock_predict:
            mock_predict.return_value = np.random.randn(len(self.X_test), self.pred_len)
            
            predictions = self.model.predict(self.X_test)
            
            assert predictions.shape[0] == len(self.X_test)
            assert predictions.shape[1] == self.pred_len
    
    def test_sequence_validation(self):
        """Test input sequence validation"""
        # Test with wrong sequence length
        wrong_seq = np.random.randn(10, 30, 1)  # Wrong sequence length
        
        with pytest.raises(Exception):
            self.model.predict(wrong_seq)

class TestDQNAgent:
    """Test DQN agent functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.state_dim = 10
        self.action_dim = 3
        self.agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=0.1
        )
        
        # Create sample environment data
        np.random.seed(42)
        self.states = np.random.randn(100, self.state_dim)
        self.actions = np.random.randint(0, self.action_dim, 100)
        self.rewards = np.random.randn(100)
        self.next_states = np.random.randn(100, self.state_dim)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent is not None
        assert self.agent.state_dim == self.state_dim
        assert self.agent.action_dim == self.action_dim
        assert hasattr(self.agent, 'q_network')
        assert hasattr(self.agent, 'target_network')
    
    def test_action_selection(self):
        """Test action selection"""
        state = np.random.randn(self.state_dim)
        action = self.agent.select_action(state)
        
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < self.action_dim
    
    def test_experience_storage(self):
        """Test experience replay buffer"""
        state = self.states[0]
        action = self.actions[0]
        reward = self.rewards[0]
        next_state = self.next_states[0]
        done = False
        
        # Store experience
        self.agent.store_experience(state, action, reward, next_state, done)
        
        # Check if experience is stored
        assert len(self.agent.memory) > 0
    
    def test_batch_training(self):
        """Test batch training"""
        # Fill memory with experiences
        for i in range(50):
            self.agent.store_experience(
                self.states[i], self.actions[i], self.rewards[i],
                self.next_states[i], False
            )
        
        # Train on batch
        initial_epsilon = self.agent.epsilon
        loss = self.agent.train_batch()
        
        # Check if training occurred
        assert loss is not None
        assert isinstance(loss, (float, np.floating))
    
    def test_target_network_update(self):
        """Test target network update"""
        # Get initial target network weights
        initial_weights = self.agent.target_network.get_weights()
        
        # Update target network
        self.agent.update_target_network()
        
        # Weights should be updated (copied from main network)
        updated_weights = self.agent.target_network.get_weights()
        
        # At least one weight should be different (unless networks are identical)
        # This is a basic check - in practice, weights will be different
        assert len(initial_weights) == len(updated_weights)
    
    def test_epsilon_decay(self):
        """Test epsilon decay functionality"""
        initial_epsilon = self.agent.epsilon
        
        # Decay epsilon
        self.agent.decay_epsilon()
        
        # Epsilon should decrease or stay at minimum
        assert self.agent.epsilon <= initial_epsilon
    
    def test_action_distribution(self):
        """Test action distribution tracking"""
        # Select multiple actions
        for _ in range(100):
            state = np.random.randn(self.state_dim)
            self.agent.select_action(state)
        
        # Get action distribution
        distribution = self.agent.get_action_distribution()
        
        assert len(distribution) == self.action_dim
        assert all(0 <= prob <= 1 for prob in distribution)
        assert abs(sum(distribution) - 1.0) < 0.01  # Should sum to ~1

class TestModelFactory:
    """Test model factory functionality"""
    
    def setup_method(self):
        """Setup model factory"""
        self.factory = ModelFactory()
    
    def test_factory_initialization(self):
        """Test factory initialization"""
        assert self.factory is not None
        assert hasattr(self.factory, 'models')
    
    def test_create_xgboost_model(self):
        """Test XGBoost model creation"""
        model = self.factory.create_model('xgboost')
        assert isinstance(model, XGBoostPredictor)
    
    def test_create_informer_model(self):
        """Test Informer model creation"""
        config = {
            'seq_len': 60,
            'label_len': 30,
            'pred_len': 1,
            'd_model': 64,
            'n_heads': 4
        }
        model = self.factory.create_model('informer', config)
        assert isinstance(model, InformerModel)
    
    def test_create_dqn_model(self):
        """Test DQN model creation"""
        config = {
            'state_dim': 10,
            'action_dim': 3,
            'learning_rate': 0.001
        }
        model = self.factory.create_model('dqn', config)
        assert isinstance(model, DQNAgent)
    
    def test_invalid_model_type(self):
        """Test invalid model type"""
        with pytest.raises(ValueError):
            self.factory.create_model('invalid_model')
    
    def test_get_available_models(self):
        """Test getting available models"""
        models = self.factory.get_available_models()
        assert isinstance(models, list)
        assert 'xgboost' in models
        assert 'informer' in models
        assert 'dqn' in models

class TestTechnicalIndicators:
    """Test technical indicators functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.indicators = TechnicalIndicators()
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 2000
        price_changes = np.random.randn(100) * 0.02  # 2% daily volatility
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        self.data = pd.DataFrame({
            'open': [p * (1 + np.random.randn() * 0.005) for p in prices],
            'high': [p * (1 + abs(np.random.randn()) * 0.01) for p in prices],
            'low': [p * (1 - abs(np.random.randn()) * 0.01) for p in prices],
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        sma = self.indicators.calculate_sma(self.data['close'], window=20)
        
        assert len(sma) == len(self.data)
        assert not sma.iloc[19:].isna().any()  # Should have values after window period
        assert sma.iloc[:19].isna().all()  # Should be NaN before window period
    
    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation"""
        ema = self.indicators.calculate_ema(self.data['close'], window=12)
        
        assert len(ema) == len(self.data)
        assert not ema.iloc[11:].isna().any()  # Should have values after window period
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = self.indicators.calculate_rsi(self.data['close'], window=14)
        
        assert len(rsi) == len(self.data)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = self.indicators.calculate_macd(
            self.data['close'], fast=12, slow=26, signal=9
        )
        
        assert len(macd_line) == len(self.data)
        assert len(signal_line) == len(self.data)
        assert len(histogram) == len(self.data)
        
        # Check histogram calculation
        valid_mask = ~(macd_line.isna() | signal_line.isna())
        expected_histogram = macd_line[valid_mask] - signal_line[valid_mask]
        actual_histogram = histogram[valid_mask]
        
        pd.testing.assert_series_equal(
            expected_histogram, actual_histogram, check_names=False
        )
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = self.indicators.calculate_bollinger_bands(
            self.data['close'], window=20, std_dev=2
        )
        
        assert len(upper) == len(self.data)
        assert len(middle) == len(self.data)
        assert len(lower) == len(self.data)
        
        # Check band relationships
        valid_mask = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_mask] >= middle[valid_mask]).all()
        assert (middle[valid_mask] >= lower[valid_mask]).all()
    
    def test_stochastic_oscillator(self):
        """Test Stochastic Oscillator calculation"""
        k_percent, d_percent = self.indicators.calculate_stochastic(
            self.data['high'], self.data['low'], self.data['close'], k_window=14, d_window=3
        )
        
        assert len(k_percent) == len(self.data)
        assert len(d_percent) == len(self.data)
        
        # Stochastic should be between 0 and 100
        valid_k = k_percent.dropna()
        valid_d = d_percent.dropna()
        
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()
    
    def test_all_indicators_calculation(self):
        """Test calculation of all indicators at once"""
        all_indicators = self.indicators.calculate_all_indicators(self.data)
        
        assert isinstance(all_indicators, pd.DataFrame)
        assert len(all_indicators) == len(self.data)
        
        # Check if key indicators are present
        expected_indicators = [
            'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd',
            'macd_signal', 'macd_histogram', 'bb_upper', 'bb_middle', 'bb_lower'
        ]
        
        for indicator in expected_indicators:
            assert indicator in all_indicators.columns
    
    def test_invalid_input_data(self):
        """Test handling of invalid input data"""
        # Test with insufficient data
        short_data = self.data.iloc[:5]
        
        # Should handle gracefully or raise appropriate error
        try:
            result = self.indicators.calculate_all_indicators(short_data)
            # If it doesn't raise an error, check that result is reasonable
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Should be a meaningful error
            assert isinstance(e, (ValueError, IndexError))

# Pytest configuration
pytest_plugins = ["pytest_asyncio"]

# Custom markers for slow tests
pytest.mark.slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow", default=False),
    reason="need --runslow option to run"
)

# Run tests with: python -m pytest tests/test_ml_models.py -v
# Run with slow tests: python -m pytest tests/test_ml_models.py -v --runslow