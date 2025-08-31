#!/usr/bin/env python3
"""
Model Accuracy Verification Script
Tests and validates all ML models with comprehensive metrics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.config import get_settings
from ml_models.xgboost_model import XGBoostPredictor
from ml_models.informer_model import InformerModel
from ml_models.dqn_model import DQNAgent
from ml_models.model_factory import ModelFactory
from services.data_fetcher import DataFetcher
from utils.technical_indicators import TechnicalIndicators
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_verification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelAccuracyVerifier:
    """Comprehensive model accuracy verification and testing"""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_fetcher = DataFetcher()
        self.technical_indicators = TechnicalIndicators()
        self.model_factory = ModelFactory()
        self.results = {}
        
    def prepare_test_data(self, symbol: str = "RELIANCE", days: int = 365) -> Dict[str, Any]:
        """Prepare comprehensive test dataset"""
        logger.info(f"Preparing test data for {symbol} ({days} days)")
        
        try:
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get OHLCV data
            historical_data = self.data_fetcher.get_historical_data(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if historical_data.empty:
                raise ValueError(f"No historical data found for {symbol}")
            
            # Calculate technical indicators
            indicators = self.technical_indicators.calculate_all_indicators(historical_data)
            
            # Combine data
            combined_data = pd.concat([historical_data, indicators], axis=1)
            combined_data = combined_data.dropna()
            
            # Create features and targets
            features = self._create_features(combined_data)
            targets = self._create_targets(combined_data)
            
            # Split data
            train_size = int(len(combined_data) * 0.8)
            
            train_data = {
                'features': features[:train_size],
                'targets': targets[:train_size],
                'raw_data': combined_data[:train_size]
            }
            
            test_data = {
                'features': features[train_size:],
                'targets': targets[train_size:],
                'raw_data': combined_data[train_size:]
            }
            
            logger.info(f"Data prepared: {len(train_data['features'])} training, {len(test_data['features'])} testing samples")
            
            return {
                'symbol': symbol,
                'train': train_data,
                'test': test_data,
                'full_data': combined_data
            }
            
        except Exception as e:
            logger.error(f"Error preparing test data: {str(e)}")
            raise
    
    def _create_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create feature matrix from OHLCV and technical indicators"""
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd',
            'macd_signal', 'macd_histogram', 'bb_upper', 'bb_middle', 'bb_lower',
            'stoch_k', 'stoch_d', 'williams_r', 'atr', 'adx'
        ]
        
        # Select available columns
        available_columns = [col for col in feature_columns if col in data.columns]
        features = data[available_columns].values
        
        # Handle any remaining NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def _create_targets(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create target variables for different prediction tasks"""
        targets = {}
        
        # Price prediction (next day close price)
        targets['price'] = data['close'].shift(-1).dropna().values
        
        # Direction prediction (up/down)
        price_change = data['close'].pct_change().shift(-1)
        targets['direction'] = (price_change > 0).astype(int).dropna().values
        
        # Volatility prediction
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=5).std().shift(-1)
        targets['volatility'] = volatility.dropna().values
        
        # Ensure all targets have the same length
        min_length = min(len(v) for v in targets.values())
        for key in targets:
            targets[key] = targets[key][:min_length]
        
        return targets
    
    def test_xgboost_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test XGBoost model accuracy"""
        logger.info("Testing XGBoost model...")
        
        try:
            # Initialize model
            xgb_model = XGBoostPredictor()
            
            # Prepare data
            X_train = data['train']['features']
            y_train = data['train']['targets']['price']
            X_test = data['test']['features']
            y_test = data['test']['targets']['price']
            
            # Ensure same length
            min_train_len = min(len(X_train), len(y_train))
            min_test_len = min(len(X_test), len(y_test))
            
            X_train = X_train[:min_train_len]
            y_train = y_train[:min_train_len]
            X_test = X_test[:min_test_len]
            y_test = y_test[:min_test_len]
            
            # Train model
            xgb_model.train(X_train, y_train)
            
            # Make predictions
            train_pred = xgb_model.predict(X_train)
            test_pred = xgb_model.predict(X_test)
            
            # Calculate metrics
            results = {
                'model': 'XGBoost',
                'train_metrics': {
                    'mse': mean_squared_error(y_train, train_pred),
                    'mae': mean_absolute_error(y_train, train_pred),
                    'r2': r2_score(y_train, train_pred),
                    'mape': np.mean(np.abs((y_train - train_pred) / y_train)) * 100
                },
                'test_metrics': {
                    'mse': mean_squared_error(y_test, test_pred),
                    'mae': mean_absolute_error(y_test, test_pred),
                    'r2': r2_score(y_test, test_pred),
                    'mape': np.mean(np.abs((y_test - test_pred) / y_test)) * 100
                },
                'predictions': {
                    'train': train_pred.tolist()[:100],  # First 100 for storage
                    'test': test_pred.tolist()[:100]
                },
                'feature_importance': xgb_model.get_feature_importance()
            }
            
            logger.info(f"XGBoost Test R²: {results['test_metrics']['r2']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"XGBoost testing failed: {str(e)}")
            return {'model': 'XGBoost', 'error': str(e)}
    
    def test_informer_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Informer model accuracy"""
        logger.info("Testing Informer model...")
        
        try:
            # Initialize model
            informer_model = InformerModel(
                seq_len=60,
                label_len=30,
                pred_len=1,
                d_model=512,
                n_heads=8,
                e_layers=2,
                d_layers=1
            )
            
            # Prepare sequential data
            X_train, y_train = self._prepare_sequential_data(
                data['train']['raw_data'], seq_len=60
            )
            X_test, y_test = self._prepare_sequential_data(
                data['test']['raw_data'], seq_len=60
            )
            
            if len(X_train) == 0 or len(X_test) == 0:
                raise ValueError("Insufficient data for sequential modeling")
            
            # Train model (simplified for verification)
            informer_model.train(X_train, y_train, epochs=10, batch_size=32)
            
            # Make predictions
            train_pred = informer_model.predict(X_train)
            test_pred = informer_model.predict(X_test)
            
            # Calculate metrics
            results = {
                'model': 'Informer',
                'train_metrics': {
                    'mse': mean_squared_error(y_train, train_pred),
                    'mae': mean_absolute_error(y_train, train_pred),
                    'r2': r2_score(y_train, train_pred)
                },
                'test_metrics': {
                    'mse': mean_squared_error(y_test, test_pred),
                    'mae': mean_absolute_error(y_test, test_pred),
                    'r2': r2_score(y_test, test_pred)
                },
                'sequence_length': 60,
                'prediction_horizon': 1
            }
            
            logger.info(f"Informer Test R²: {results['test_metrics']['r2']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Informer testing failed: {str(e)}")
            return {'model': 'Informer', 'error': str(e)}
    
    def test_dqn_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test DQN model performance"""
        logger.info("Testing DQN model...")
        
        try:
            # Initialize DQN agent
            state_dim = data['train']['features'].shape[1]
            action_dim = 3  # Buy, Hold, Sell
            
            dqn_agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=0.001,
                gamma=0.95,
                epsilon=0.1
            )
            
            # Prepare trading environment data
            train_env_data = self._prepare_trading_environment(
                data['train']['raw_data']
            )
            test_env_data = self._prepare_trading_environment(
                data['test']['raw_data']
            )
            
            # Train DQN (simplified)
            train_rewards = dqn_agent.train_on_batch(
                train_env_data['states'],
                train_env_data['actions'],
                train_env_data['rewards'],
                episodes=100
            )
            
            # Test DQN
            test_rewards = dqn_agent.evaluate(
                test_env_data['states'],
                test_env_data['rewards']
            )
            
            results = {
                'model': 'DQN',
                'train_metrics': {
                    'avg_reward': np.mean(train_rewards),
                    'total_reward': np.sum(train_rewards),
                    'win_rate': np.mean(np.array(train_rewards) > 0)
                },
                'test_metrics': {
                    'avg_reward': np.mean(test_rewards),
                    'total_reward': np.sum(test_rewards),
                    'win_rate': np.mean(np.array(test_rewards) > 0)
                },
                'action_distribution': dqn_agent.get_action_distribution()
            }
            
            logger.info(f"DQN Test Win Rate: {results['test_metrics']['win_rate']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"DQN testing failed: {str(e)}")
            return {'model': 'DQN', 'error': str(e)}
    
    def _prepare_sequential_data(self, data: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for time series models"""
        if len(data) < seq_len + 1:
            return np.array([]), np.array([])
        
        # Use close price as the main feature
        prices = data['close'].values
        
        X, y = [], []
        for i in range(len(prices) - seq_len):
            X.append(prices[i:i + seq_len])
            y.append(prices[i + seq_len])
        
        return np.array(X), np.array(y)
    
    def _prepare_trading_environment(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare trading environment data for DQN"""
        prices = data['close'].values
        returns = np.diff(prices) / prices[:-1]
        
        states = []
        actions = []
        rewards = []
        
        for i in range(1, len(returns)):
            # State: recent price changes and technical indicators
            state = returns[max(0, i-5):i]  # Last 5 returns
            if len(state) < 5:
                state = np.pad(state, (5-len(state), 0), 'constant')
            
            # Simple action based on return
            if returns[i] > 0.01:
                action = 0  # Buy
            elif returns[i] < -0.01:
                action = 2  # Sell
            else:
                action = 1  # Hold
            
            # Reward based on next period return
            reward = returns[i] if i < len(returns) - 1 else 0
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards)
        }
    
    def test_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test technical indicators accuracy"""
        logger.info("Testing technical indicators...")
        
        try:
            raw_data = data['full_data']
            
            # Test indicator calculations
            indicators_test = {
                'sma_accuracy': self._test_sma_accuracy(raw_data),
                'ema_accuracy': self._test_ema_accuracy(raw_data),
                'rsi_accuracy': self._test_rsi_accuracy(raw_data),
                'macd_accuracy': self._test_macd_accuracy(raw_data),
                'bollinger_accuracy': self._test_bollinger_accuracy(raw_data)
            }
            
            results = {
                'model': 'Technical Indicators',
                'test_results': indicators_test,
                'overall_accuracy': np.mean(list(indicators_test.values()))
            }
            
            logger.info(f"Technical Indicators Overall Accuracy: {results['overall_accuracy']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Technical indicators testing failed: {str(e)}")
            return {'model': 'Technical Indicators', 'error': str(e)}
    
    def _test_sma_accuracy(self, data: pd.DataFrame) -> float:
        """Test SMA calculation accuracy"""
        if 'sma_20' not in data.columns:
            return 0.0
        
        # Manual SMA calculation
        manual_sma = data['close'].rolling(window=20).mean()
        calculated_sma = data['sma_20']
        
        # Compare non-NaN values
        valid_mask = ~(manual_sma.isna() | calculated_sma.isna())
        if valid_mask.sum() == 0:
            return 0.0
        
        correlation = np.corrcoef(
            manual_sma[valid_mask], 
            calculated_sma[valid_mask]
        )[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _test_ema_accuracy(self, data: pd.DataFrame) -> float:
        """Test EMA calculation accuracy"""
        if 'ema_12' not in data.columns:
            return 0.0
        
        # Manual EMA calculation
        manual_ema = data['close'].ewm(span=12).mean()
        calculated_ema = data['ema_12']
        
        valid_mask = ~(manual_ema.isna() | calculated_ema.isna())
        if valid_mask.sum() == 0:
            return 0.0
        
        correlation = np.corrcoef(
            manual_ema[valid_mask], 
            calculated_ema[valid_mask]
        )[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _test_rsi_accuracy(self, data: pd.DataFrame) -> float:
        """Test RSI calculation accuracy"""
        if 'rsi' not in data.columns:
            return 0.0
        
        # Check RSI bounds (should be between 0 and 100)
        rsi_values = data['rsi'].dropna()
        if len(rsi_values) == 0:
            return 0.0
        
        valid_range = ((rsi_values >= 0) & (rsi_values <= 100)).mean()
        return valid_range
    
    def _test_macd_accuracy(self, data: pd.DataFrame) -> float:
        """Test MACD calculation accuracy"""
        if 'macd' not in data.columns or 'macd_signal' not in data.columns:
            return 0.0
        
        # Check if MACD histogram is correctly calculated
        if 'macd_histogram' in data.columns:
            manual_histogram = data['macd'] - data['macd_signal']
            calculated_histogram = data['macd_histogram']
            
            valid_mask = ~(manual_histogram.isna() | calculated_histogram.isna())
            if valid_mask.sum() == 0:
                return 0.0
            
            correlation = np.corrcoef(
                manual_histogram[valid_mask],
                calculated_histogram[valid_mask]
            )[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.5  # Partial credit if histogram not available
    
    def _test_bollinger_accuracy(self, data: pd.DataFrame) -> float:
        """Test Bollinger Bands calculation accuracy"""
        required_cols = ['bb_upper', 'bb_middle', 'bb_lower']
        if not all(col in data.columns for col in required_cols):
            return 0.0
        
        # Check if upper > middle > lower
        valid_order = (
            (data['bb_upper'] >= data['bb_middle']) & 
            (data['bb_middle'] >= data['bb_lower'])
        ).mean()
        
        return valid_order
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive accuracy report"""
        report = []
        report.append("=" * 80)
        report.append("MODEL ACCURACY VERIFICATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Symbol: {results.get('symbol', 'N/A')}")
        report.append("")
        
        for model_name, model_results in results.items():
            if model_name == 'symbol':
                continue
                
            report.append(f"{'='*20} {model_name.upper()} {'='*20}")
            
            if 'error' in model_results:
                report.append(f"❌ ERROR: {model_results['error']}")
                report.append("")
                continue
            
            # Training metrics
            if 'train_metrics' in model_results:
                report.append("📊 TRAINING METRICS:")
                for metric, value in model_results['train_metrics'].items():
                    report.append(f"  {metric.upper()}: {value:.4f}")
                report.append("")
            
            # Testing metrics
            if 'test_metrics' in model_results:
                report.append("🎯 TESTING METRICS:")
                for metric, value in model_results['test_metrics'].items():
                    report.append(f"  {metric.upper()}: {value:.4f}")
                report.append("")
            
            # Model-specific information
            if model_name == 'XGBoost' and 'feature_importance' in model_results:
                report.append("🔍 TOP FEATURES:")
                importance = model_results['feature_importance']
                for i, (feature, score) in enumerate(importance[:5]):
                    report.append(f"  {i+1}. {feature}: {score:.4f}")
                report.append("")
            
            if model_name == 'DQN' and 'action_distribution' in model_results:
                report.append("🎮 ACTION DISTRIBUTION:")
                actions = ['Buy', 'Hold', 'Sell']
                for i, prob in enumerate(model_results['action_distribution']):
                    report.append(f"  {actions[i]}: {prob:.2%}")
                report.append("")
        
        # Summary
        report.append("=" * 80)
        report.append("SUMMARY")
        report.append("=" * 80)
        
        successful_models = []
        failed_models = []
        
        for model_name, model_results in results.items():
            if model_name == 'symbol':
                continue
            if 'error' in model_results:
                failed_models.append(model_name)
            else:
                successful_models.append(model_name)
        
        report.append(f"✅ Successful Models: {', '.join(successful_models)}")
        if failed_models:
            report.append(f"❌ Failed Models: {', '.join(failed_models)}")
        
        report.append("")
        report.append("Recommendations:")
        
        # Add recommendations based on results
        for model_name, model_results in results.items():
            if model_name == 'symbol' or 'error' in model_results:
                continue
            
            if 'test_metrics' in model_results:
                if 'r2' in model_results['test_metrics']:
                    r2 = model_results['test_metrics']['r2']
                    if r2 > 0.8:
                        report.append(f"  🟢 {model_name}: Excellent performance (R² = {r2:.3f})")
                    elif r2 > 0.6:
                        report.append(f"  🟡 {model_name}: Good performance (R² = {r2:.3f})")
                    else:
                        report.append(f"  🔴 {model_name}: Needs improvement (R² = {r2:.3f})")
        
        return "\n".join(report)
    
    def run_full_verification(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Run complete model verification"""
        if symbols is None:
            symbols = ["RELIANCE", "TCS", "INFY"]
        
        all_results = {}
        
        for symbol in symbols:
            logger.info(f"Starting verification for {symbol}")
            
            try:
                # Prepare test data
                data = self.prepare_test_data(symbol)
                
                # Test all models
                symbol_results = {
                    'symbol': symbol,
                    'XGBoost': self.test_xgboost_model(data),
                    'Informer': self.test_informer_model(data),
                    'DQN': self.test_dqn_model(data),
                    'Technical_Indicators': self.test_technical_indicators(data)
                }
                
                all_results[symbol] = symbol_results
                
                # Generate and save report
                report = self.generate_report(symbol_results)
                
                # Save report to file
                report_file = f"model_verification_report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(report_file, 'w') as f:
                    f.write(report)
                
                logger.info(f"Report saved: {report_file}")
                print(report)
                
            except Exception as e:
                logger.error(f"Verification failed for {symbol}: {str(e)}")
                all_results[symbol] = {'error': str(e)}
        
        return all_results

def main():
    """Main verification function"""
    logger.info("Starting Model Accuracy Verification")
    
    verifier = ModelAccuracyVerifier()
    
    # Run verification
    results = verifier.run_full_verification()
    
    # Save consolidated results
    results_file = f"model_verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Verification complete. Results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    main()