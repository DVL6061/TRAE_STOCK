import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel, BertTokenizer, BertForSequenceClassification
)
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinGPTSentimentAnalyzer:
    """
    Advanced Financial Sentiment Analysis using FinGPT and ensemble methods.
    Supports multiple models and real-time news processing.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", cache_dir: str = "./models/cache"):
        """
        Initialize FinGPT Sentiment Analyzer.
        
        Args:
            model_name: Pre-trained model name for financial sentiment
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        
        # News sources configuration
        self.news_sources = {
            'moneycontrol': {
                'base_url': 'https://www.moneycontrol.com',
                'search_url': 'https://www.moneycontrol.com/news/tags/{}.html',
                'selectors': {
                    'title': 'h2 a, h3 a',
                    'content': '.arti-flow p',
                    'date': '.article_schedule'
                }
            },
            'cnbc': {
                'base_url': 'https://www.cnbc.com',
                'search_url': 'https://www.cnbc.com/search/?query={}',
                'selectors': {
                    'title': '.SearchResult-headline',
                    'content': '.ArticleBody-articleBody p',
                    'date': '.ArticleHeader-time'
                }
            },
            'mint': {
                'base_url': 'https://www.livemint.com',
                'search_url': 'https://www.livemint.com/search?q={}',
                'selectors': {
                    'title': '.headline a',
                    'content': '.mainSectionBody p',
                    'date': '.publish_on'
                }
            }
        }
        
        # Sentiment labels mapping
        self.sentiment_labels = {
            'POSITIVE': 1,
            'NEGATIVE': -1,
            'NEUTRAL': 0
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize all sentiment analysis models.
        """
        try:
            # Create cache directory
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Initialize FinBERT model
            logger.info(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("FinGPT models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing FinGPT models: {str(e)}")
            # Fallback to basic sentiment analysis
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """
        Initialize fallback models if FinBERT fails.
        """
        try:
            logger.info("Initializing fallback sentiment models")
            
            # Use a simpler model as fallback
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Fallback models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing fallback models: {str(e)}")
            self.sentiment_pipeline = None
    
    @lru_cache(maxsize=1000)
    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text using multiple methods.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            if not text or len(text.strip()) == 0:
                return {
                    'sentiment': 'NEUTRAL',
                    'confidence': 0.0,
                    'score': 0.0,
                    'method': 'empty_text'
                }
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            results = {}
            
            # Method 1: FinBERT/Transformer model
            if self.sentiment_pipeline:
                try:
                    transformer_result = self.sentiment_pipeline(cleaned_text[:512])
                    results['transformer'] = {
                        'sentiment': transformer_result[0]['label'],
                        'confidence': transformer_result[0]['score']
                    }
                except Exception as e:
                    logger.warning(f"Transformer sentiment analysis failed: {str(e)}")
            
            # Method 2: TextBlob sentiment
            try:
                blob = TextBlob(cleaned_text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    textblob_sentiment = 'POSITIVE'
                elif polarity < -0.1:
                    textblob_sentiment = 'NEGATIVE'
                else:
                    textblob_sentiment = 'NEUTRAL'
                
                results['textblob'] = {
                    'sentiment': textblob_sentiment,
                    'confidence': abs(polarity),
                    'polarity': polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except Exception as e:
                logger.warning(f"TextBlob sentiment analysis failed: {str(e)}")
            
            # Method 3: Financial keyword-based analysis
            keyword_result = self._analyze_financial_keywords(cleaned_text)
            results['keywords'] = keyword_result
            
            # Ensemble the results
            final_result = self._ensemble_sentiment_results(results)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'score': 0.0,
                'error': str(e)
            }
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^a-zA-Z0-9\s\$\%\+\-\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _analyze_financial_keywords(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment based on financial keywords.
        
        Args:
            text: Input text
            
        Returns:
            Keyword-based sentiment analysis
        """
        positive_keywords = [
            'profit', 'growth', 'increase', 'rise', 'gain', 'bull', 'bullish',
            'surge', 'rally', 'boom', 'strong', 'positive', 'upgrade', 'buy',
            'outperform', 'beat', 'exceed', 'record', 'high', 'momentum'
        ]
        
        negative_keywords = [
            'loss', 'decline', 'fall', 'drop', 'bear', 'bearish', 'crash',
            'plunge', 'weak', 'negative', 'downgrade', 'sell', 'underperform',
            'miss', 'below', 'low', 'concern', 'risk', 'volatility'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0
            }
        
        sentiment_score = (positive_count - negative_count) / total_keywords
        
        if sentiment_score > 0.2:
            sentiment = 'POSITIVE'
        elif sentiment_score < -0.2:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'
        
        confidence = abs(sentiment_score)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    def _ensemble_sentiment_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine multiple sentiment analysis results using ensemble method.
        
        Args:
            results: Dictionary of sentiment results from different methods
            
        Returns:
            Final ensemble sentiment result
        """
        if not results:
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'score': 0.0,
                'method': 'ensemble'
            }
        
        # Weights for different methods
        weights = {
            'transformer': 0.5,
            'textblob': 0.3,
            'keywords': 0.2
        }
        
        sentiment_scores = []
        confidences = []
        
        for method, weight in weights.items():
            if method in results:
                result = results[method]
                
                # Convert sentiment to score
                if result['sentiment'] == 'POSITIVE':
                    score = 1.0
                elif result['sentiment'] == 'NEGATIVE':
                    score = -1.0
                else:
                    score = 0.0
                
                # Weight the score by confidence and method weight
                weighted_score = score * result['confidence'] * weight
                sentiment_scores.append(weighted_score)
                confidences.append(result['confidence'] * weight)
        
        if not sentiment_scores:
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'score': 0.0,
                'method': 'ensemble'
            }
        
        # Calculate final sentiment
        final_score = sum(sentiment_scores)
        final_confidence = sum(confidences)
        
        if final_score > 0.1:
            final_sentiment = 'POSITIVE'
        elif final_score < -0.1:
            final_sentiment = 'NEGATIVE'
        else:
            final_sentiment = 'NEUTRAL'
        
        return {
            'sentiment': final_sentiment,
            'confidence': min(final_confidence, 1.0),
            'score': final_score,
            'method': 'ensemble',
            'component_results': results
        }
    
    def fetch_news_data(self, ticker: str, days_back: int = 7, max_articles: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch news data for a specific ticker from multiple sources.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back for news
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of news articles with metadata
        """
        try:
            all_articles = []
            
            # Search terms for the ticker
            search_terms = [ticker, ticker.replace('.NS', ''), ticker.replace('.BO', '')]
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                
                for source_name, source_config in self.news_sources.items():
                    for term in search_terms:
                        future = executor.submit(
                            self._fetch_from_source,
                            source_name,
                            source_config,
                            term,
                            days_back,
                            max_articles // len(self.news_sources)
                        )
                        futures.append(future)
                
                for future in as_completed(futures, timeout=30):
                    try:
                        articles = future.result()
                        all_articles.extend(articles)
                    except Exception as e:
                        logger.warning(f"Failed to fetch from source: {str(e)}")
            
            # Remove duplicates and sort by date
            unique_articles = self._deduplicate_articles(all_articles)
            
            # Limit to max_articles
            return unique_articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            return []
    
    def _fetch_from_source(self, source_name: str, source_config: Dict, 
                          search_term: str, days_back: int, max_articles: int) -> List[Dict[str, Any]]:
        """
        Fetch articles from a specific news source.
        
        Args:
            source_name: Name of the news source
            source_config: Configuration for the source
            search_term: Search term (ticker)
            days_back: Days to look back
            max_articles: Maximum articles to fetch
            
        Returns:
            List of articles from the source
        """
        articles = []
        
        try:
            # This is a simplified implementation
            # In a real implementation, you would use proper web scraping
            # or news APIs like NewsAPI, Alpha Vantage News, etc.
            
            # For demonstration, we'll create mock news data
            # In production, replace this with actual web scraping or API calls
            
            mock_articles = self._generate_mock_news(search_term, days_back, max_articles)
            articles.extend(mock_articles)
            
        except Exception as e:
            logger.error(f"Error fetching from {source_name}: {str(e)}")
        
        return articles
    
    def _generate_mock_news(self, ticker: str, days_back: int, max_articles: int) -> List[Dict[str, Any]]:
        """
        Generate mock news data for demonstration.
        In production, replace with actual news fetching.
        
        Args:
            ticker: Stock ticker
            days_back: Days back
            max_articles: Max articles
            
        Returns:
            Mock news articles
        """
        mock_headlines = [
            f"{ticker} reports strong quarterly earnings, beats estimates",
            f"{ticker} announces new product launch, shares surge",
            f"{ticker} faces regulatory challenges, stock declines",
            f"{ticker} CEO optimistic about future growth prospects",
            f"{ticker} expands operations to new markets",
            f"Analysts upgrade {ticker} rating to buy",
            f"{ticker} dividend announcement boosts investor confidence",
            f"Market volatility affects {ticker} trading volume",
            f"{ticker} partnership deal creates new opportunities",
            f"Industry trends favor {ticker} business model"
        ]
        
        articles = []
        
        for i in range(min(max_articles, len(mock_headlines))):
            article_date = datetime.now() - timedelta(days=np.random.randint(0, days_back))
            
            articles.append({
                'title': mock_headlines[i],
                'content': f"This is mock content for {mock_headlines[i]}. " * 10,
                'date': article_date,
                'source': 'mock_source',
                'url': f"https://example.com/news/{i}",
                'ticker': ticker
            })
        
        return articles
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate articles based on title similarity.
        
        Args:
            articles: List of articles
            
        Returns:
            Deduplicated articles
        """
        if not articles:
            return []
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title_key = article['title'].lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        # Sort by date (newest first)
        unique_articles.sort(key=lambda x: x['date'], reverse=True)
        
        return unique_articles
    
    def analyze_news_sentiment(self, ticker: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to analyze
            
        Returns:
            Comprehensive sentiment analysis results
        """
        try:
            # Fetch news articles
            articles = self.fetch_news_data(ticker, days_back)
            
            if not articles:
                return {
                    'ticker': ticker,
                    'overall_sentiment': 'NEUTRAL',
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'article_count': 0,
                    'error': 'No articles found'
                }
            
            # Analyze sentiment for each article
            article_sentiments = []
            
            for article in articles:
                # Combine title and content for analysis
                text = f"{article['title']} {article.get('content', '')}"
                
                sentiment_result = self.analyze_text_sentiment(text)
                
                article_sentiment = {
                    'title': article['title'],
                    'date': article['date'],
                    'source': article['source'],
                    'sentiment': sentiment_result['sentiment'],
                    'confidence': sentiment_result['confidence'],
                    'score': sentiment_result['score']
                }
                
                article_sentiments.append(article_sentiment)
            
            # Calculate overall sentiment
            overall_result = self._calculate_overall_sentiment(article_sentiments)
            
            return {
                'ticker': ticker,
                'analysis_date': datetime.now(),
                'days_analyzed': days_back,
                'article_count': len(articles),
                'overall_sentiment': overall_result['sentiment'],
                'sentiment_score': overall_result['score'],
                'confidence': overall_result['confidence'],
                'sentiment_distribution': overall_result['distribution'],
                'article_sentiments': article_sentiments,
                'trending_sentiment': self._calculate_sentiment_trend(article_sentiments)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {
                'ticker': ticker,
                'overall_sentiment': 'NEUTRAL',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'error': str(e)
            }
    
    def _calculate_overall_sentiment(self, article_sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall sentiment from individual article sentiments.
        
        Args:
            article_sentiments: List of article sentiment results
            
        Returns:
            Overall sentiment analysis
        """
        if not article_sentiments:
            return {
                'sentiment': 'NEUTRAL',
                'score': 0.0,
                'confidence': 0.0,
                'distribution': {'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0}
            }
        
        # Count sentiment distribution
        distribution = {'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0}
        weighted_scores = []
        confidences = []
        
        for article in article_sentiments:
            sentiment = article['sentiment']
            confidence = article['confidence']
            score = article['score']
            
            distribution[sentiment] += 1
            weighted_scores.append(score * confidence)
            confidences.append(confidence)
        
        # Calculate overall metrics
        avg_score = np.mean(weighted_scores) if weighted_scores else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Determine overall sentiment
        if avg_score > 0.1:
            overall_sentiment = 'POSITIVE'
        elif avg_score < -0.1:
            overall_sentiment = 'NEGATIVE'
        else:
            overall_sentiment = 'NEUTRAL'
        
        return {
            'sentiment': overall_sentiment,
            'score': avg_score,
            'confidence': avg_confidence,
            'distribution': distribution
        }
    
    def _calculate_sentiment_trend(self, article_sentiments: List[Dict[str, Any]]) -> str:
        """
        Calculate sentiment trend over time.
        
        Args:
            article_sentiments: List of article sentiments sorted by date
            
        Returns:
            Trend description
        """
        if len(article_sentiments) < 3:
            return 'insufficient_data'
        
        # Sort by date
        sorted_articles = sorted(article_sentiments, key=lambda x: x['date'])
        
        # Calculate trend using recent vs older articles
        mid_point = len(sorted_articles) // 2
        recent_scores = [a['score'] for a in sorted_articles[mid_point:]]
        older_scores = [a['score'] for a in sorted_articles[:mid_point]]
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        trend_diff = recent_avg - older_avg
        
        if trend_diff > 0.2:
            return 'improving'
        elif trend_diff < -0.2:
            return 'declining'
        else:
            return 'stable'
    
    def get_sentiment_features(self, ticker: str, days_back: int = 7) -> Dict[str, float]:
        """
        Get sentiment features for model training.
        
        Args:
            ticker: Stock ticker
            days_back: Days to analyze
            
        Returns:
            Dictionary of sentiment features
        """
        try:
            sentiment_analysis = self.analyze_news_sentiment(ticker, days_back)
            
            features = {
                'sentiment_score': sentiment_analysis.get('sentiment_score', 0.0),
                'sentiment_confidence': sentiment_analysis.get('confidence', 0.0),
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'article_count': sentiment_analysis.get('article_count', 0),
                'sentiment_volatility': 0.0
            }
            
            # Calculate ratios
            distribution = sentiment_analysis.get('sentiment_distribution', {})
            total_articles = sum(distribution.values()) if distribution else 0
            
            if total_articles > 0:
                features['positive_ratio'] = distribution.get('POSITIVE', 0) / total_articles
                features['negative_ratio'] = distribution.get('NEGATIVE', 0) / total_articles
                features['neutral_ratio'] = distribution.get('NEUTRAL', 0) / total_articles
            
            # Calculate sentiment volatility
            article_sentiments = sentiment_analysis.get('article_sentiments', [])
            if len(article_sentiments) > 1:
                scores = [a['score'] for a in article_sentiments]
                features['sentiment_volatility'] = np.std(scores)
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting sentiment features: {str(e)}")
            return {
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'article_count': 0,
                'sentiment_volatility': 0.0
            }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the sentiment analyzer configuration.
        
        Args:
            filepath: Path to save the model
        """
        try:
            model_data = {
                'model_name': self.model_name,
                'cache_dir': self.cache_dir,
                'news_sources': self.news_sources,
                'sentiment_labels': self.sentiment_labels,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"FinGPT model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving FinGPT model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load the sentiment analyzer configuration.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model_name = model_data.get('model_name', self.model_name)
            self.cache_dir = model_data.get('cache_dir', self.cache_dir)
            self.news_sources = model_data.get('news_sources', self.news_sources)
            self.sentiment_labels = model_data.get('sentiment_labels', self.sentiment_labels)
            
            # Reinitialize models with loaded configuration
            self._initialize_models()
            
            logger.info(f"FinGPT model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading FinGPT model: {str(e)}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Initialize FinGPT analyzer
    analyzer = FinGPTSentimentAnalyzer()
    
    # Test text sentiment analysis
    test_text = "The company reported strong quarterly earnings, beating analyst expectations by 15%."
    result = analyzer.analyze_text_sentiment(test_text)
    print(f"Text sentiment: {result}")
    
    # Test news sentiment analysis
    ticker = "RELIANCE.NS"
    news_sentiment = analyzer.analyze_news_sentiment(ticker, days_back=7)
    print(f"News sentiment for {ticker}: {news_sentiment}")
    
    # Get sentiment features
    features = analyzer.get_sentiment_features(ticker)
    print(f"Sentiment features: {features}")