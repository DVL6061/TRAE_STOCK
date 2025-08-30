import os
import sys
import logging
import schedule
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.ML_models.train_models import train_models, prepare_training_data
from backend.ML_models.model_factory import model_factory
from backend.utils.config import TRAINING_CONFIG, PREDICTION_WINDOWS
from backend.utils.helpers import validate_ticker
from backend.app.config import DATABASE_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'training_scheduler.log'))
    ]
)

logger = logging.getLogger(__name__)

class TrainingScheduler:
    """
    Automated training scheduler for ML models with performance monitoring and retraining logic.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the training scheduler.
        
        Args:
            config: Configuration dictionary for training parameters
        """
        self.config = config or {
            'retrain_frequency_days': 7,  # Retrain every 7 days
            'performance_threshold': 0.05,  # Retrain if performance drops by 5%
            'min_data_points': 1000,  # Minimum data points required for training
            'max_concurrent_trainings': 3,  # Maximum concurrent training jobs
            'default_tickers': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
            'training_timeframes': ['intraday', 'short_term', 'medium_term', 'long_term'],
            'model_types': ['xgboost', 'informer', 'dqn']
        }
        
        self.is_running = False
        self.scheduler_thread = None
        self.performance_history = {}
        self.last_training_times = {}
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        logger.info("Training scheduler initialized")
    
    def start_scheduler(self):
        """
        Start the automated training scheduler.
        """
        if self.is_running:
            logger.warning("Training scheduler is already running")
            return
        
        self.is_running = True
        
        # Schedule daily performance checks
        schedule.every().day.at("02:00").do(self._check_and_retrain_models)
        
        # Schedule weekly full retraining
        schedule.every().sunday.at("03:00").do(self._full_retrain_all_models)
        
        # Schedule monthly model cleanup
        schedule.every().month.do(self._cleanup_old_models)
        
        # Start scheduler in a separate thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Training scheduler started")
    
    def stop_scheduler(self):
        """
        Stop the automated training scheduler.
        """
        self.is_running = False
        schedule.clear()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Training scheduler stopped")
    
    def _run_scheduler(self):
        """
        Run the scheduler loop.
        """
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _check_and_retrain_models(self):
        """
        Check model performance and retrain if necessary.
        """
        logger.info("Starting daily model performance check")
        
        try:
            tickers_to_retrain = []
            
            for ticker in self.config['default_tickers']:
                if self._should_retrain_model(ticker):
                    tickers_to_retrain.append(ticker)
            
            if tickers_to_retrain:
                logger.info(f"Retraining models for tickers: {tickers_to_retrain}")
                self._retrain_models_parallel(tickers_to_retrain)
            else:
                logger.info("No models require retraining based on performance metrics")
                
        except Exception as e:
            logger.error(f"Error during model performance check: {str(e)}")
    
    def _should_retrain_model(self, ticker: str) -> bool:
        """
        Determine if a model should be retrained based on performance and time criteria.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Boolean indicating if model should be retrained
        """
        try:
            # Check if enough time has passed since last training
            last_training = self.last_training_times.get(ticker)
            if last_training:
                days_since_training = (datetime.now() - last_training).days
                if days_since_training < self.config['retrain_frequency_days']:
                    return False
            
            # Check model performance
            current_performance = self._evaluate_model_performance(ticker)
            if current_performance is None:
                logger.warning(f"Could not evaluate performance for {ticker}, scheduling retrain")
                return True
            
            # Compare with historical performance
            historical_performance = self.performance_history.get(ticker, [])
            if historical_performance:
                avg_historical = np.mean([p['accuracy'] for p in historical_performance[-5:]])
                performance_drop = avg_historical - current_performance['accuracy']
                
                if performance_drop > self.config['performance_threshold']:
                    logger.info(f"Performance drop detected for {ticker}: {performance_drop:.4f}")
                    return True
            
            # Check data freshness
            if self._is_data_stale(ticker):
                logger.info(f"Stale data detected for {ticker}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain criteria for {ticker}: {str(e)}")
            return True  # Err on the side of retraining
    
    def _evaluate_model_performance(self, ticker: str) -> Optional[Dict]:
        """
        Evaluate current model performance.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with performance metrics or None if evaluation fails
        """
        try:
            # Get recent data for evaluation
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            evaluation_data = prepare_training_data(ticker, start_date, end_date)
            if evaluation_data.empty or len(evaluation_data) < 10:
                return None
            
            # Evaluate XGBoost model performance (primary model)
            try:
                xgb_model = model_factory.get_price_prediction_model('xgboost', ticker, 'short_term')
                if hasattr(xgb_model, 'model') and xgb_model.model is not None:
                    predictions = xgb_model.predict(evaluation_data)
                    if predictions is not None and len(predictions) > 0:
                        # Calculate accuracy metrics
                        actual_prices = evaluation_data['close'].iloc[-len(predictions):].values
                        mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
                        accuracy = max(0, 100 - mape)  # Convert MAPE to accuracy percentage
                        
                        return {
                            'accuracy': accuracy / 100,  # Normalize to 0-1
                            'mape': mape,
                            'evaluation_date': datetime.now(),
                            'data_points': len(predictions)
                        }
            except Exception as e:
                logger.warning(f"Could not evaluate XGBoost model for {ticker}: {str(e)}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating model performance for {ticker}: {str(e)}")
            return None
    
    def _is_data_stale(self, ticker: str) -> bool:
        """
        Check if the training data is stale and needs refresh.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Boolean indicating if data is stale
        """
        try:
            # Check if we have recent data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            recent_data = prepare_training_data(ticker, start_date, end_date)
            
            # Consider data stale if we have less than 3 days of recent data
            return len(recent_data) < 3
            
        except Exception as e:
            logger.error(f"Error checking data staleness for {ticker}: {str(e)}")
            return True  # Assume stale if we can't check
    
    def _retrain_models_parallel(self, tickers: List[str]):
        """
        Retrain models for multiple tickers in parallel.
        
        Args:
            tickers: List of ticker symbols to retrain
        """
        with ThreadPoolExecutor(max_workers=self.config['max_concurrent_trainings']) as executor:
            # Submit training jobs
            future_to_ticker = {
                executor.submit(self._retrain_single_ticker, ticker): ticker 
                for ticker in tickers
            }
            
            # Process completed jobs
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        logger.info(f"Successfully retrained models for {ticker}")
                        self.last_training_times[ticker] = datetime.now()
                    else:
                        logger.error(f"Failed to retrain models for {ticker}")
                except Exception as e:
                    logger.error(f"Error retraining models for {ticker}: {str(e)}")
    
    def _retrain_single_ticker(self, ticker: str) -> bool:
        """
        Retrain models for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Boolean indicating success
        """
        try:
            logger.info(f"Starting retraining for {ticker}")
            
            # Prepare training data (last 2 years)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            
            # Validate ticker
            if not validate_ticker(ticker):
                logger.error(f"Invalid ticker: {ticker}")
                return False
            
            # Train models
            train_models(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                models=self.config['model_types'],
                timeframes=self.config['training_timeframes']
            )
            
            # Update performance history
            performance = self._evaluate_model_performance(ticker)
            if performance:
                if ticker not in self.performance_history:
                    self.performance_history[ticker] = []
                self.performance_history[ticker].append(performance)
                
                # Keep only last 10 performance records
                self.performance_history[ticker] = self.performance_history[ticker][-10:]
            
            return True
            
        except Exception as e:
            logger.error(f"Error retraining models for {ticker}: {str(e)}")
            return False
    
    def _full_retrain_all_models(self):
        """
        Perform full retraining of all models (weekly schedule).
        """
        logger.info("Starting weekly full model retraining")
        
        try:
            self._retrain_models_parallel(self.config['default_tickers'])
            logger.info("Weekly full model retraining completed")
        except Exception as e:
            logger.error(f"Error during full model retraining: {str(e)}")
    
    def _cleanup_old_models(self):
        """
        Clean up old model files to save disk space.
        """
        logger.info("Starting monthly model cleanup")
        
        try:
            models_dir = 'models'
            if not os.path.exists(models_dir):
                return
            
            # Remove model files older than 3 months
            cutoff_date = datetime.now() - timedelta(days=90)
            
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) < cutoff_date.timestamp():
                        try:
                            os.remove(file_path)
                            logger.info(f"Removed old model file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Could not remove file {file_path}: {str(e)}")
            
            logger.info("Monthly model cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during model cleanup: {str(e)}")
    
    def manual_retrain(self, ticker: str, models: List[str] = None, timeframes: List[str] = None) -> bool:
        """
        Manually trigger retraining for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            models: List of model types to train (default: all)
            timeframes: List of timeframes to train (default: all)
            
        Returns:
            Boolean indicating success
        """
        models = models or self.config['model_types']
        timeframes = timeframes or self.config['training_timeframes']
        
        logger.info(f"Manual retraining triggered for {ticker}")
        
        success = self._retrain_single_ticker(ticker)
        if success:
            self.last_training_times[ticker] = datetime.now()
        
        return success
    
    def get_training_status(self) -> Dict:
        """
        Get current training status and performance metrics.
        
        Returns:
            Dictionary with training status information
        """
        return {
            'scheduler_running': self.is_running,
            'last_training_times': {k: v.isoformat() for k, v in self.last_training_times.items()},
            'performance_history': self.performance_history,
            'config': self.config,
            'next_scheduled_check': schedule.next_run().isoformat() if schedule.jobs else None
        }

# Global scheduler instance
training_scheduler = TrainingScheduler()

def start_training_scheduler():
    """
    Start the global training scheduler.
    """
    training_scheduler.start_scheduler()

def stop_training_scheduler():
    """
    Stop the global training scheduler.
    """
    training_scheduler.stop_scheduler()

def get_scheduler_status():
    """
    Get the current scheduler status.
    """
    return training_scheduler.get_training_status()

def manual_retrain_ticker(ticker: str, models: List[str] = None, timeframes: List[str] = None):
    """
    Manually trigger retraining for a ticker.
    """
    return training_scheduler.manual_retrain(ticker, models, timeframes)