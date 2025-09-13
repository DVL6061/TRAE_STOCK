import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
from .config import get_settings

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry["extra"] = record.extra_data
            
        return json.dumps(log_entry, ensure_ascii=False)

class StockPredictionLogger:
    """Enhanced logger for stock prediction system"""
    
    def __init__(self, name: str = "stock_prediction"):
        self.settings = get_settings()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.settings.log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if self.settings.environment == "production":
            # Use JSON formatter for production
            console_handler.setFormatter(JSONFormatter())
        else:
            # Use simple formatter for development
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        
        # File handler for production
        if self.settings.environment == "production":
            self._setup_file_handler()
    
    def _setup_file_handler(self):
        """Setup rotating file handler for production"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Main application log
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "stock_prediction.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)
        
        # Error-only log
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, extra_data: Dict[str, Any] = None):
        """Log info message with optional extra data"""
        self._log_with_extra(logging.INFO, message, extra_data)
    
    def warning(self, message: str, extra_data: Dict[str, Any] = None):
        """Log warning message with optional extra data"""
        self._log_with_extra(logging.WARNING, message, extra_data)
    
    def error(self, message: str, extra_data: Dict[str, Any] = None, exc_info: bool = True):
        """Log error message with optional extra data and exception info"""
        self._log_with_extra(logging.ERROR, message, extra_data, exc_info)
    
    def debug(self, message: str, extra_data: Dict[str, Any] = None):
        """Log debug message with optional extra data"""
        self._log_with_extra(logging.DEBUG, message, extra_data)
    
    def critical(self, message: str, extra_data: Dict[str, Any] = None, exc_info: bool = True):
        """Log critical message with optional extra data and exception info"""
        self._log_with_extra(logging.CRITICAL, message, extra_data, exc_info)
    
    def _log_with_extra(self, level: int, message: str, extra_data: Dict[str, Any] = None, exc_info: bool = False):
        """Internal method to log with extra data"""
        if extra_data:
            # Create a custom LogRecord with extra data
            record = self.logger.makeRecord(
                self.logger.name, level, "", 0, message, (), None
            )
            record.extra_data = extra_data
            self.logger.handle(record)
        else:
            self.logger.log(level, message, exc_info=exc_info)
    
    def log_api_request(self, endpoint: str, method: str, status_code: int, 
                       response_time: float, user_id: str = None):
        """Log API request details"""
        extra_data = {
            "type": "api_request",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": response_time * 1000,
            "user_id": user_id
        }
        self.info(f"API {method} {endpoint} - {status_code}", extra_data)
    
    def log_prediction_request(self, symbol: str, prediction_type: str, 
                              model_used: str, confidence: float = None):
        """Log prediction request details"""
        extra_data = {
            "type": "prediction_request",
            "symbol": symbol,
            "prediction_type": prediction_type,
            "model_used": model_used,
            "confidence": confidence
        }
        self.info(f"Prediction request for {symbol} using {model_used}", extra_data)
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float], 
                             dataset_size: int = None):
        """Log model performance metrics"""
        extra_data = {
            "type": "model_performance",
            "model_name": model_name,
            "metrics": metrics,
            "dataset_size": dataset_size
        }
        self.info(f"Model performance for {model_name}", extra_data)
    
    def log_data_fetch(self, source: str, symbol: str, data_points: int, 
                      fetch_time: float, success: bool = True):
        """Log data fetching operations"""
        extra_data = {
            "type": "data_fetch",
            "source": source,
            "symbol": symbol,
            "data_points": data_points,
            "fetch_time_ms": fetch_time * 1000,
            "success": success
        }
        level = logging.INFO if success else logging.WARNING
        message = f"Data fetch from {source} for {symbol}: {'success' if success else 'failed'}"
        self._log_with_extra(level, message, extra_data)

# Global logger instance
logger = StockPredictionLogger()

# Convenience functions for backward compatibility
def get_logger(name: str = None) -> StockPredictionLogger:
    """Get logger instance"""
    if name:
        return StockPredictionLogger(name)
    return logger

def log_info(message: str, extra_data: Dict[str, Any] = None):
    """Log info message"""
    logger.info(message, extra_data)

def log_warning(message: str, extra_data: Dict[str, Any] = None):
    """Log warning message"""
    logger.warning(message, extra_data)

def log_error(message: str, extra_data: Dict[str, Any] = None, exc_info: bool = True):
    """Log error message"""
    logger.error(message, extra_data, exc_info)

def log_debug(message: str, extra_data: Dict[str, Any] = None):
    """Log debug message"""
    logger.debug(message, extra_data)

def log_critical(message: str, extra_data: Dict[str, Any] = None, exc_info: bool = True):
    """Log critical message"""
    logger.critical(message, extra_data, exc_info)