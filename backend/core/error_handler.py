import logging
import traceback
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import json
import asyncio
from contextlib import asynccontextmanager

from .config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StockPredictionError(Exception):
    """Base exception class for stock prediction system"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or "GENERAL_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)

class DataFetchError(StockPredictionError):
    """Exception raised when data fetching fails"""
    def __init__(self, message: str, source: str = None, symbol: str = None):
        details = {"source": source, "symbol": symbol}
        super().__init__(message, "DATA_FETCH_ERROR", details)

class ModelError(StockPredictionError):
    """Exception raised when model operations fail"""
    def __init__(self, message: str, model_type: str = None, operation: str = None):
        details = {"model_type": model_type, "operation": operation}
        super().__init__(message, "MODEL_ERROR", details)

class PredictionError(StockPredictionError):
    """Exception raised when prediction generation fails"""
    def __init__(self, message: str, symbol: str = None, timeframe: str = None):
        details = {"symbol": symbol, "timeframe": timeframe}
        super().__init__(message, "PREDICTION_ERROR", details)

class NewsProcessingError(StockPredictionError):
    """Exception raised when news processing fails"""
    def __init__(self, message: str, source: str = None, article_count: int = None):
        details = {"source": source, "article_count": article_count}
        super().__init__(message, "NEWS_PROCESSING_ERROR", details)

class APIError(StockPredictionError):
    """Exception raised when external API calls fail"""
    def __init__(self, message: str, api_name: str = None, status_code: int = None):
        details = {"api_name": api_name, "status_code": status_code}
        super().__init__(message, "API_ERROR", details)

class ValidationError(StockPredictionError):
    """Exception raised when data validation fails"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        details = {"field": field, "value": str(value) if value is not None else None}
        super().__init__(message, "VALIDATION_ERROR", details)

class CacheError(StockPredictionError):
    """Exception raised when cache operations fail"""
    def __init__(self, message: str, operation: str = None, key: str = None):
        details = {"operation": operation, "key": key}
        super().__init__(message, "CACHE_ERROR", details)

class DatabaseError(StockPredictionError):
    """Exception raised when database operations fail"""
    def __init__(self, message: str, operation: str = None, table: str = None):
        details = {"operation": operation, "table": table}
        super().__init__(message, "DATABASE_ERROR", details)

class RateLimitError(StockPredictionError):
    """Exception raised when API rate limits are exceeded"""
    def __init__(self, message: str, api_name: str = None, retry_after: int = None):
        details = {"api_name": api_name, "retry_after": retry_after}
        super().__init__(message, "RATE_LIMIT_ERROR", details)

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self):
        self.settings = get_settings()
        self.error_counts = {}
        self.error_history = []
        self.max_history_size = 1000
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None, request_id: str = None):
        """Log error with context information and tracking"""
        error_id = str(uuid.uuid4())
        request_id = request_id or str(uuid.uuid4())
        
        error_info = {
            "error_id": error_id,
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc() if self.settings.debug else None,
            "context": context or {},
            "environment": self.settings.environment
        }
        
        if isinstance(error, StockPredictionError):
            error_info.update({
                "error_code": error.error_code,
                "details": error.details
            })
        
        # Track error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to history (with size limit)
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
        
        logger.error(f"Error occurred: {json.dumps(error_info, indent=2)}")
        return error_info
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
            "recent_errors": len([e for e in self.error_history 
                                 if datetime.fromisoformat(e['timestamp']) > 
                                 datetime.now().replace(hour=datetime.now().hour-1)]),
            "history_size": len(self.error_history)
        }
    
    def handle_data_fetch_error(self, error: Exception, source: str = None, symbol: str = None) -> DataFetchError:
        """Handle data fetching errors"""
        if isinstance(error, DataFetchError):
            return error
        
        message = f"Failed to fetch data: {str(error)}"
        data_error = DataFetchError(message, source, symbol)
        self.log_error(data_error)
        return data_error
    
    @staticmethod
    def handle_model_error(error: Exception, model_type: str = None, operation: str = None) -> ModelError:
        """Handle model operation errors"""
        if isinstance(error, ModelError):
            return error
        
        message = f"Model operation failed: {str(error)}"
        model_error = ModelError(message, model_type, operation)
        ErrorHandler.log_error(model_error)
        return model_error
    
    @staticmethod
    def handle_prediction_error(error: Exception, symbol: str = None, timeframe: str = None) -> PredictionError:
        """Handle prediction generation errors"""
        if isinstance(error, PredictionError):
            return error
        
        message = f"Prediction generation failed: {str(error)}"
        pred_error = PredictionError(message, symbol, timeframe)
        ErrorHandler.log_error(pred_error)
        return pred_error
    
    @staticmethod
    def handle_news_processing_error(error: Exception, source: str = None, article_count: int = None) -> NewsProcessingError:
        """Handle news processing errors"""
        if isinstance(error, NewsProcessingError):
            return error
        
        message = f"News processing failed: {str(error)}"
        news_error = NewsProcessingError(message, source, article_count)
        ErrorHandler.log_error(news_error)
        return news_error
    
    @staticmethod
    def handle_api_error(error: Exception, api_name: str = None, status_code: int = None) -> APIError:
        """Handle external API errors"""
        if isinstance(error, APIError):
            return error
        
        message = f"API call failed: {str(error)}"
        api_error = APIError(message, api_name, status_code)
        ErrorHandler.log_error(api_error)
        return api_error

# FastAPI exception handlers
async def stock_prediction_exception_handler(request: Request, exc: StockPredictionError):
    """Handle custom stock prediction exceptions"""
    return JSONResponse(
        status_code=400,
        content={
            "error": True,
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": exc.timestamp
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    error_info = ErrorHandler.log_error(exc, {"path": str(request.url)})
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Decorator for error handling
def handle_errors(error_type: str = "general"):
    """Decorator to handle errors in functions"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if error_type == "data_fetch":
                    raise ErrorHandler.handle_data_fetch_error(e)
                elif error_type == "model":
                    raise ErrorHandler.handle_model_error(e)
                elif error_type == "prediction":
                    raise ErrorHandler.handle_prediction_error(e)
                elif error_type == "news":
                    raise ErrorHandler.handle_news_processing_error(e)
                elif error_type == "api":
                    raise ErrorHandler.handle_api_error(e)
                else:
                    ErrorHandler.log_error(e)
                    raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_type == "data_fetch":
                    raise ErrorHandler.handle_data_fetch_error(e)
                elif error_type == "model":
                    raise ErrorHandler.handle_model_error(e)
                elif error_type == "prediction":
                    raise ErrorHandler.handle_prediction_error(e)
                elif error_type == "news":
                    raise ErrorHandler.handle_news_processing_error(e)
                elif error_type == "api":
                    raise ErrorHandler.handle_api_error(e)
                else:
                    ErrorHandler.log_error(e)
                    raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Utility functions
def create_error_response(message: str, error_code: str = "ERROR", status_code: int = 400, details: Dict[str, Any] = None):
    """Create standardized error response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "error_code": error_code,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
    )

def log_info(message: str, context: Dict[str, Any] = None):
    """Log informational message with context"""
    log_data = {
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "context": context or {}
    }
    logger.info(json.dumps(log_data))

def log_warning(message: str, context: Dict[str, Any] = None):
    """Log warning message"""
    log_data = {
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "context": context or {}
    }
    logger.warning(json.dumps(log_data))

class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise StockPredictionError("Circuit breaker is OPEN", "CIRCUIT_BREAKER_OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Global instances
error_handler = ErrorHandler()
circuit_breaker = CircuitBreaker()