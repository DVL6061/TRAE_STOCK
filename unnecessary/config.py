import os
from dotenv import load_dotenv
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "historical").mkdir(exist_ok=True)
(DATA_DIR / "news").mkdir(exist_ok=True)

# API configurations
API_PREFIX = "/api"
API_V1_STR = "/v1"

# Yahoo Finance API configuration
YAHOO_FINANCE_INTERVAL_MAPPING = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
    "1h": "60m",
    "1d": "1d",
    "5d": "5d",
    "1wk": "1wk",
    "1mo": "1mo",
    "3mo": "3mo"
}

# Angel One API configuration
ANGEL_ONE_API_KEY = os.getenv("ANGEL_ONE_API_KEY", "")
ANGEL_ONE_CLIENT_ID = os.getenv("ANGEL_ONE_CLIENT_ID", "")
ANGEL_ONE_PASSWORD = os.getenv("ANGEL_ONE_PASSWORD", "")

# News API configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Model configurations
DEFAULT_MODEL_CONFIG = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "objective": "reg:squarederror"
    },
    "informer": {
        "enc_in": 7,  # Number of input features
        "dec_in": 7,  # Number of input features for decoder
        "c_out": 1,   # Number of outputs
        "seq_len": 60,  # Input sequence length
        "label_len": 30,  # Label sequence length
        "pred_len": 30,  # Prediction sequence length
        "factor": 5,  # Probsparse attention factor
        "d_model": 512,  # Dimension of model
        "n_heads": 8,  # Number of heads
        "e_layers": 2,  # Number of encoder layers
        "d_layers": 1,  # Number of decoder layers
        "d_ff": 2048,  # Dimension of FCN
        "dropout": 0.05,  # Dropout
        "attn": "prob",  # Attention used in encoder
        "embed": "timeF",  # Time features encoding
        "activation": "gelu",  # Activation
        "output_attention": False,  # Whether to output attention in encoder
        "distil": True,  # Whether to use distilling in encoder
    },
    "dqn": {
        "gamma": 0.99,  # Discount factor
        "epsilon_start": 1.0,  # Starting epsilon for epsilon-greedy
        "epsilon_end": 0.01,  # Final epsilon for epsilon-greedy
        "epsilon_decay": 500,  # Decay rate for epsilon
        "memory_size": 10000,  # Size of replay memory
        "batch_size": 64,  # Batch size for training
        "target_update": 10,  # Update frequency for target network
        "hidden_size": 128  # Size of hidden layers
    }
}

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    "rsi": {
        "window": 14,
        "name": "Relative Strength Index"
    },
    "macd": {
        "window_slow": 26,
        "window_fast": 12,
        "window_sign": 9,
        "name": "Moving Average Convergence Divergence"
    },
    "sma": {
        "window": 20,
        "name": "Simple Moving Average"
    },
    "ema": {
        "window": 20,
        "name": "Exponential Moving Average"
    },
    "bb": {
        "window": 20,
        "window_dev": 2,
        "name": "Bollinger Bands"
    },
    "adx": {
        "window": 14,
        "name": "Average Directional Index"
    },
    "stoch": {
        "window": 14,
        "smooth_window": 3,
        "name": "Stochastic Oscillator"
    }
}

# Prediction window mapping
PREDICTION_WINDOW_MAPPING = {
    "1d": {"days": 1, "name": "1 Day"},
    "3d": {"days": 3, "name": "3 Days"},
    "1w": {"days": 7, "name": "1 Week"},
    "2w": {"days": 14, "name": "2 Weeks"},
    "1m": {"days": 30, "name": "1 Month"},
    "3m": {"days": 90, "name": "3 Months"}
}

# Trading timeframe mapping
TRADING_TIMEFRAME_MAPPING = {
    "scalping": {"hours": 1, "name": "Scalping (Intraday)"},
    "intraday": {"hours": 6, "name": "Intraday"},
    "short_term": {"days": 5, "name": "Short-term (1-5 days)"},
    "medium_term": {"days": 30, "name": "Medium-term (1-4 weeks)"},
    "long_term": {"days": 90, "name": "Long-term (1-3 months)"}
}

# Risk tolerance mapping
RISK_TOLERANCE_MAPPING = {
    "low": {"description": "Conservative approach with emphasis on capital preservation"},
    "moderate": {"description": "Balanced approach with moderate risk for moderate returns"},
    "high": {"description": "Aggressive approach with higher risk for potentially higher returns"}
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "default",
            "filename": str(BASE_DIR / "logs" / "app.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5
        },
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    },
}

# Create logs directory if it doesn't exist
(BASE_DIR / "logs").mkdir(exist_ok=True)

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Application settings
APP_NAME = "Stock Prediction System"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Enterprise-grade stock prediction system using ML, RL, and Transformers"
APP_AUTHOR = "AI Stock Prediction Team"

# Default stock ticker
DEFAULT_TICKER = "TATAMOTORS.NS"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# CORS settings
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Initialize configuration
def init_config():
    """Initialize application configuration"""
    logger.info(f"Initializing {APP_NAME} v{APP_VERSION}")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Server running on {HOST}:{PORT} (Debug: {DEBUG})")
    
    # Check for required environment variables
    missing_vars = []
    if not ANGEL_ONE_API_KEY:
        missing_vars.append("ANGEL_ONE_API_KEY")
    if not ANGEL_ONE_CLIENT_ID:
        missing_vars.append("ANGEL_ONE_CLIENT_ID")
    if not ANGEL_ONE_PASSWORD:
        missing_vars.append("ANGEL_ONE_PASSWORD")
    if not NEWS_API_KEY:
        missing_vars.append("NEWS_API_KEY")
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Some features may not work properly without these variables")
    
    return True