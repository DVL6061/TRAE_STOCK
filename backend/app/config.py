import os
from typing import Dict, List

# Angel One API Configuration
ANGEL_ONE_API_KEY = os.getenv('ANGEL_ONE_API_KEY')
ANGEL_ONE_CLIENT_ID = os.getenv('ANGEL_ONE_CLIENT_ID')
ANGEL_ONE_PASSWORD = os.getenv('ANGEL_ONE_PASSWORD')

# Model Paths
MODEL_PATHS = {
    'xgboost': 'models/xgboost_model.json',
    'informer': 'models/informer_model',
    'dqn': 'models/dqn_model.zip'
}

# News Sources Configuration
NEWS_SOURCES: List[Dict] = [
    {
        'name': 'Moneycontrol',
        'url': 'https://www.moneycontrol.com/news/business/markets/',
        'article_selector': 'div.article-list article',
        'title_selector': 'h2',
        'link_selector': 'a',
        'timestamp_selector': 'span.article-date',
        'content_selector': 'div.content-article',
        'timestamp_format': '%B %d, %Y %I:%M %p'
    },
    {
        'name': 'Economic Times Markets',
        'url': 'https://economictimes.indiatimes.com/markets/stocks/news',
        'article_selector': 'div.eachStory',
        'title_selector': 'h3',
        'link_selector': 'a',
        'timestamp_selector': 'time',
        'content_selector': 'div.artText',
        'timestamp_format': '%d %b, %Y, %I:%M %p'
    },
    {
        'name': 'LiveMint Markets',
        'url': 'https://www.livemint.com/market/stock-market-news',
        'article_selector': 'div.listingNew article',
        'title_selector': 'h2',
        'link_selector': 'a',
        'timestamp_selector': 'span.dateTime',
        'content_selector': 'div.mainArea',
        'timestamp_format': '%d %b %Y, %I:%M %p'
    },
    {
        'name': 'Business Standard Markets',
        'url': 'https://www.business-standard.com/markets',
        'article_selector': 'div.article-list article',
        'title_selector': 'h2',
        'link_selector': 'a',
        'timestamp_selector': 'span.time',
        'content_selector': 'div.article-content',
        'timestamp_format': '%d %B %Y %I:%M %p'
    }
]

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    'SMA': [20, 50, 200],  # Simple Moving Average periods
    'EMA': [12, 26],       # Exponential Moving Average periods
    'RSI': 14,             # Relative Strength Index period
    'MACD': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'Bollinger': {
        'period': 20,
        'std_dev': 2
    }
}

# Model Training Configuration
TRAINING_CONFIG = {
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'objective': 'reg:squarederror',
        'early_stopping_rounds': 50
    },
    'informer': {
        'n_encoder_layers': 3,
        'n_decoder_layers': 2,
        'embedding_dim': 512,
        'dropout': 0.1,
        'attention_dropout': 0.1
    },
    'dqn': {
        'learning_rate': 0.0001,
        'buffer_size': 100000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.02,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'learning_starts': 50000,
        'batch_size': 32,
        'gamma': 0.99
    }
}

# Prediction Windows Configuration
PREDICTION_WINDOWS = {
    'scalping': '5m',      # 5 minutes
    'intraday': '1h',      # 1 hour
    'swing': '1d',         # 1 day
    'position': '1wk',     # 1 week
    'long_term': '1mo'     # 1 month
}