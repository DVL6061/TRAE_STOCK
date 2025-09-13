import os
from typing import Optional, List
from pydantic import BaseSettings, validator
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Settings
    APP_NAME: str = "TRAE Stock Prediction System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Database Settings
    DATABASE_URL: Optional[str] = None
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "trae_stock"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    
    # Redis Settings
    REDIS_URL: Optional[str] = None
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL: int = 3600  # 1 hour default cache TTL
    
    # API Keys and External Services
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    ANGEL_ONE_API_KEY: Optional[str] = None
    ANGEL_ONE_CLIENT_ID: Optional[str] = None
    ANGEL_ONE_PASSWORD: Optional[str] = None
    ANGEL_ONE_TOTP_SECRET: Optional[str] = None
    
    # OpenAI/FinGPT Settings
    OPENAI_API_KEY: Optional[str] = None
    FINGPT_MODEL_PATH: Optional[str] = None
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    ALLOWED_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # ML Model Settings
    MODEL_CACHE_DIR: str = "./models/cache"
    XGBOOST_MODEL_PATH: Optional[str] = None
    INFORMER_MODEL_PATH: Optional[str] = None
    DQN_MODEL_PATH: Optional[str] = None
    
    # Data Settings
    DATA_CACHE_DIR: str = "./data/cache"
    MAX_HISTORICAL_DAYS: int = 365 * 5  # 5 years
    DEFAULT_PREDICTION_WINDOW: str = "1w"
    
    # News and Sentiment Settings
    NEWS_SOURCES: List[str] = [
        "cnbc", "moneycontrol", "mint", "economic-times", "business-standard"
    ]
    SENTIMENT_BATCH_SIZE: int = 32
    NEWS_CACHE_HOURS: int = 6
    
    # Technical Indicators Settings
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0
    
    # Risk Management Settings
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    STOP_LOSS_PERCENTAGE: float = 0.05  # 5%
    TAKE_PROFIT_PERCENTAGE: float = 0.15  # 15%
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = None
    
    # Rate Limiting Settings
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    
    # Celery Settings (for background tasks)
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    
    # Monitoring and Health Check Settings
    HEALTH_CHECK_INTERVAL: int = 300  # 5 minutes
    METRICS_ENABLED: bool = True
    
    @validator('DATABASE_URL', pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        return (
            f"postgresql://{values.get('POSTGRES_USER')}:"
            f"{values.get('POSTGRES_PASSWORD')}@"
            f"{values.get('POSTGRES_HOST')}:"
            f"{values.get('POSTGRES_PORT')}/"
            f"{values.get('POSTGRES_DB')}"
        )
    
    @validator('REDIS_URL', pre=True)
    def assemble_redis_connection(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        
        password_part = ""
        if values.get('REDIS_PASSWORD'):
            password_part = f":{values.get('REDIS_PASSWORD')}@"
        
        return (
            f"redis://{password_part}"
            f"{values.get('REDIS_HOST')}:"
            f"{values.get('REDIS_PORT')}/"
            f"{values.get('REDIS_DB')}"
        )
    
    @validator('CELERY_BROKER_URL', pre=True)
    def assemble_celery_broker(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        # Default to Redis if not specified
        return values.get('REDIS_URL', 'redis://localhost:6379/1')
    
    @validator('CELERY_RESULT_BACKEND', pre=True)
    def assemble_celery_result_backend(cls, v: Optional[str], values: dict) -> str:
        if isinstance(v, str):
            return v
        # Default to Redis if not specified
        return values.get('REDIS_URL', 'redis://localhost:6379/2')
    
    @validator('SECRET_KEY')
    def validate_secret_key(cls, v: str, values: dict) -> str:
        if values.get('ENVIRONMENT') == 'production' and v == 'your-secret-key-change-in-production':
            raise ValueError('SECRET_KEY must be changed in production environment')
        return v
    
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()
    
    def get_api_credentials(self) -> dict:
        """Get all API credentials in a secure format"""
        credentials = {}
        
        # Only include non-None API keys
        if self.ALPHA_VANTAGE_API_KEY:
            credentials['alpha_vantage'] = {
                'api_key': self.ALPHA_VANTAGE_API_KEY,
                'base_url': 'https://www.alphavantage.co/query'
            }
        
        if self.NEWS_API_KEY:
            credentials['news_api'] = {
                'api_key': self.NEWS_API_KEY,
                'base_url': 'https://newsapi.org/v2'
            }
        
        if self.ANGEL_ONE_API_KEY:
            credentials['angel_one'] = {
                'api_key': self.ANGEL_ONE_API_KEY,
                'client_id': self.ANGEL_ONE_CLIENT_ID,
                'password': self.ANGEL_ONE_PASSWORD,
                'totp_secret': self.ANGEL_ONE_TOTP_SECRET,
                'base_url': 'https://apiconnect.angelbroking.com'
            }
        
        if self.OPENAI_API_KEY:
            credentials['openai'] = {
                'api_key': self.OPENAI_API_KEY,
                'base_url': 'https://api.openai.com/v1'
            }
        
        return credentials
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT.lower() == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT.lower() == 'development'
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration"""
        return {
            'allow_origins': self.ALLOWED_ORIGINS,
            'allow_methods': self.ALLOWED_METHODS,
            'allow_headers': self.ALLOWED_HEADERS,
            'allow_credentials': True
        }
    
    def get_logging_config(self) -> dict:
        """Get logging configuration"""
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': self.LOG_FORMAT,
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                    'level': self.LOG_LEVEL,
                },
            },
            'root': {
                'level': self.LOG_LEVEL,
                'handlers': ['console'],
            },
        }
        
        # Add file handler if log file is specified
        if self.LOG_FILE:
            config['handlers']['file'] = {
                'class': 'logging.FileHandler',
                'filename': self.LOG_FILE,
                'formatter': 'default',
                'level': self.LOG_LEVEL,
            }
            config['root']['handlers'].append('file')
        
        return config
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Convenience function to get specific configurations
def get_database_url() -> str:
    """Get database URL"""
    return get_settings().DATABASE_URL

def get_redis_url() -> str:
    """Get Redis URL"""
    return get_settings().REDIS_URL

def get_api_credentials() -> dict:
    """Get API credentials"""
    return get_settings().get_api_credentials()

def is_production() -> bool:
    """Check if running in production"""
    return get_settings().is_production()

def is_development() -> bool:
    """Check if running in development"""
    return get_settings().is_development()

# Export settings instance
settings = get_settings()