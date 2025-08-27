import os
import logging
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import asdict
import time
from functools import lru_cache

# Import our custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fingpt_model import FinGPTSentimentAnalyzer
from data.news_fetcher import ComprehensiveNewsFetcher, NewsArticle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentService:
    """
    Comprehensive sentiment analysis service that integrates FinGPT with news fetching
    and provides real-time sentiment features for trading models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment service.
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config
        
        # Initialize FinGPT analyzer
        self.sentiment_analyzer = FinGPTSentimentAnalyzer(
            model_name=config.get('fingpt_model', 'ProsusAI/finbert'),
            cache_dir=config.get('model_cache_dir', './models/cache')
        )
        
        # Initialize news fetcher
        self.news_fetcher = ComprehensiveNewsFetcher({
            'newsapi_key': config.get('newsapi_key'),
            'alphavantage_key': config.get('alphavantage_key')
        })
        
        # Cache for sentiment results
        self.sentiment_cache = {}
        self.cache_ttl = config.get('cache_ttl_minutes', 30) * 60  # Convert to seconds
        
        # Sentiment feature weights
        self.feature_weights = {
            'recent_sentiment': 0.4,
            'sentiment_trend': 0.3,
            'news_volume': 0.2,
            'sentiment_volatility': 0.1
        }
        
        logger.info("Sentiment service initialized successfully")
    
    async def get_comprehensive_sentiment(self, ticker: str, 
                                        days_back: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to analyze
            
        Returns:
            Comprehensive sentiment analysis results
        """
        try:
            # Check cache first
            cache_key = f"{ticker}_{days_back}"
            if self._is_cache_valid(cache_key):
                logger.info(f"Returning cached sentiment for {ticker}")
                return self.sentiment_cache[cache_key]['data']
            
            # Fetch news articles
            logger.info(f"Fetching news for {ticker} (last {days_back} days)")
            articles = await self._fetch_news_async(ticker, days_back)
            
            if not articles:
                return self._get_neutral_sentiment_result(ticker, "No news articles found")
            
            # Analyze sentiment for all articles
            logger.info(f"Analyzing sentiment for {len(articles)} articles")
            sentiment_results = await self._analyze_articles_sentiment(articles)
            
            # Calculate comprehensive metrics
            comprehensive_result = self._calculate_comprehensive_metrics(
                ticker, articles, sentiment_results, days_back
            )
            
            # Cache the result
            self.sentiment_cache[cache_key] = {
                'data': comprehensive_result,
                'timestamp': time.time()
            }
            
            logger.info(f"Sentiment analysis completed for {ticker}")
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive sentiment analysis: {str(e)}")
            return self._get_neutral_sentiment_result(ticker, str(e))
    
    async def _fetch_news_async(self, ticker: str, days_back: int) -> List[NewsArticle]:
        """
        Fetch news articles asynchronously.
        
        Args:
            ticker: Stock ticker
            days_back: Days to look back
            
        Returns:
            List of news articles
        """
        loop = asyncio.get_event_loop()
        
        # Run news fetching in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.news_fetcher.fetch_news,
                ticker,
                days_back,
                max_articles=100
            )
            articles = await loop.run_in_executor(None, lambda: future.result())
        
        return articles
    
    async def _analyze_articles_sentiment(self, articles: List[NewsArticle]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple articles concurrently.
        
        Args:
            articles: List of news articles
            
        Returns:
            List of sentiment analysis results
        """
        loop = asyncio.get_event_loop()
        
        # Process articles in batches to avoid overwhelming the system
        batch_size = 10
        all_results = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = []
            for article in batch:
                task = loop.run_in_executor(
                    None,
                    self._analyze_single_article,
                    article
                )
                tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and add valid results
            for result in batch_results:
                if isinstance(result, dict):
                    all_results.append(result)
                else:
                    logger.warning(f"Sentiment analysis failed for article: {str(result)}")
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        return all_results
    
    def _analyze_single_article(self, article: NewsArticle) -> Dict[str, Any]:
        """
        Analyze sentiment for a single article.
        
        Args:
            article: News article to analyze
            
        Returns:
            Sentiment analysis result
        """
        try:
            # Combine title and content for analysis
            text = f"{article.title} {article.content}"
            
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer.analyze_text_sentiment(text)
            
            # Add article metadata
            result = {
                'article_id': getattr(article, 'article_id', None),
                'title': article.title,
                'source': article.source,
                'published_date': article.published_date,
                'url': article.url,
                'sentiment': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'score': sentiment_result['score'],
                'method': sentiment_result.get('method', 'unknown'),
                'relevance_score': getattr(article, 'relevance_score', 0.5)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing article sentiment: {str(e)}")
            return {
                'title': getattr(article, 'title', 'Unknown'),
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'score': 0.0,
                'error': str(e)
            }
    
    def _calculate_comprehensive_metrics(self, ticker: str, articles: List[NewsArticle],
                                       sentiment_results: List[Dict[str, Any]],
                                       days_back: int) -> Dict[str, Any]:
        """
        Calculate comprehensive sentiment metrics.
        
        Args:
            ticker: Stock ticker
            articles: Original articles
            sentiment_results: Sentiment analysis results
            days_back: Analysis period
            
        Returns:
            Comprehensive sentiment metrics
        """
        try:
            if not sentiment_results:
                return self._get_neutral_sentiment_result(ticker, "No sentiment results")
            
            # Basic sentiment statistics
            sentiments = [r['sentiment'] for r in sentiment_results]
            scores = [r['score'] for r in sentiment_results]
            confidences = [r['confidence'] for r in sentiment_results]
            
            # Sentiment distribution
            sentiment_counts = {
                'POSITIVE': sentiments.count('POSITIVE'),
                'NEUTRAL': sentiments.count('NEUTRAL'),
                'NEGATIVE': sentiments.count('NEGATIVE')
            }
            
            total_articles = len(sentiment_results)
            sentiment_distribution = {
                k: v / total_articles for k, v in sentiment_counts.items()
            }
            
            # Weighted sentiment score (considering confidence and relevance)
            weighted_scores = []
            for result in sentiment_results:
                weight = result['confidence'] * result.get('relevance_score', 0.5)
                weighted_score = result['score'] * weight
                weighted_scores.append(weighted_score)
            
            overall_sentiment_score = np.mean(weighted_scores) if weighted_scores else 0.0
            
            # Determine overall sentiment
            if overall_sentiment_score > 0.1:
                overall_sentiment = 'POSITIVE'
            elif overall_sentiment_score < -0.1:
                overall_sentiment = 'NEGATIVE'
            else:
                overall_sentiment = 'NEUTRAL'
            
            # Time-based analysis
            time_analysis = self._analyze_sentiment_over_time(sentiment_results, days_back)
            
            # Source analysis
            source_analysis = self._analyze_sentiment_by_source(sentiment_results)
            
            # Volatility and trend analysis
            volatility_metrics = self._calculate_sentiment_volatility(sentiment_results)
            
            # News volume analysis
            volume_metrics = self._analyze_news_volume(articles, days_back)
            
            # Generate trading features
            trading_features = self._generate_trading_features(
                overall_sentiment_score,
                sentiment_distribution,
                volatility_metrics,
                volume_metrics,
                time_analysis
            )
            
            # Compile comprehensive result
            result = {
                'ticker': ticker,
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_period_days': days_back,
                'total_articles': total_articles,
                
                # Overall sentiment
                'overall_sentiment': overall_sentiment,
                'overall_sentiment_score': float(overall_sentiment_score),
                'average_confidence': float(np.mean(confidences)) if confidences else 0.0,
                
                # Distribution
                'sentiment_distribution': sentiment_distribution,
                'sentiment_counts': sentiment_counts,
                
                # Time analysis
                'time_analysis': time_analysis,
                
                # Source analysis
                'source_analysis': source_analysis,
                
                # Volatility and trends
                'volatility_metrics': volatility_metrics,
                
                # Volume metrics
                'volume_metrics': volume_metrics,
                
                # Trading features
                'trading_features': trading_features,
                
                # Individual article results (limited for performance)
                'sample_articles': sentiment_results[:10],
                
                # Quality metrics
                'quality_metrics': {
                    'avg_relevance_score': np.mean([r.get('relevance_score', 0.5) for r in sentiment_results]),
                    'high_confidence_ratio': sum(1 for r in sentiment_results if r['confidence'] > 0.7) / total_articles,
                    'source_diversity': len(set(r.get('source', 'unknown') for r in sentiment_results))
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {str(e)}")
            return self._get_neutral_sentiment_result(ticker, str(e))
    
    def _analyze_sentiment_over_time(self, sentiment_results: List[Dict[str, Any]],
                                   days_back: int) -> Dict[str, Any]:
        """
        Analyze sentiment trends over time.
        
        Args:
            sentiment_results: Sentiment analysis results
            days_back: Analysis period
            
        Returns:
            Time-based sentiment analysis
        """
        try:
            # Group by date
            daily_sentiments = {}
            
            for result in sentiment_results:
                date_key = result['published_date'].date()
                if date_key not in daily_sentiments:
                    daily_sentiments[date_key] = []
                daily_sentiments[date_key].append(result['score'])
            
            # Calculate daily averages
            daily_averages = {
                str(date): np.mean(scores) for date, scores in daily_sentiments.items()
            }
            
            # Calculate trend
            if len(daily_averages) > 1:
                dates = sorted(daily_averages.keys())
                scores = [daily_averages[date] for date in dates]
                
                # Simple trend calculation
                recent_avg = np.mean(scores[-3:]) if len(scores) >= 3 else np.mean(scores)
                older_avg = np.mean(scores[:-3]) if len(scores) > 3 else np.mean(scores)
                
                trend_direction = 'improving' if recent_avg > older_avg + 0.1 else \
                                'declining' if recent_avg < older_avg - 0.1 else 'stable'
                
                trend_strength = abs(recent_avg - older_avg)
            else:
                trend_direction = 'insufficient_data'
                trend_strength = 0.0
            
            return {
                'daily_averages': daily_averages,
                'trend_direction': trend_direction,
                'trend_strength': float(trend_strength),
                'days_with_news': len(daily_sentiments),
                'coverage_consistency': len(daily_sentiments) / days_back
            }
            
        except Exception as e:
            logger.error(f"Error in time analysis: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_sentiment_by_source(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment by news source.
        
        Args:
            sentiment_results: Sentiment analysis results
            
        Returns:
            Source-based sentiment analysis
        """
        try:
            source_sentiments = {}
            
            for result in sentiment_results:
                source = result.get('source', 'unknown')
                if source not in source_sentiments:
                    source_sentiments[source] = []
                source_sentiments[source].append(result['score'])
            
            source_analysis = {}
            for source, scores in source_sentiments.items():
                source_analysis[source] = {
                    'article_count': len(scores),
                    'avg_sentiment': float(np.mean(scores)),
                    'sentiment_std': float(np.std(scores)),
                    'positive_ratio': sum(1 for s in scores if s > 0.1) / len(scores),
                    'negative_ratio': sum(1 for s in scores if s < -0.1) / len(scores)
                }
            
            return source_analysis
            
        except Exception as e:
            logger.error(f"Error in source analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_sentiment_volatility(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate sentiment volatility metrics.
        
        Args:
            sentiment_results: Sentiment analysis results
            
        Returns:
            Volatility metrics
        """
        try:
            scores = [r['score'] for r in sentiment_results]
            
            if len(scores) < 2:
                return {
                    'sentiment_volatility': 0.0,
                    'sentiment_range': 0.0,
                    'volatility_category': 'low'
                }
            
            volatility = float(np.std(scores))
            sentiment_range = float(max(scores) - min(scores))
            
            # Categorize volatility
            if volatility < 0.3:
                volatility_category = 'low'
            elif volatility < 0.6:
                volatility_category = 'medium'
            else:
                volatility_category = 'high'
            
            return {
                'sentiment_volatility': volatility,
                'sentiment_range': sentiment_range,
                'volatility_category': volatility_category,
                'score_std': volatility,
                'score_min': float(min(scores)),
                'score_max': float(max(scores))
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_news_volume(self, articles: List[NewsArticle], days_back: int) -> Dict[str, Any]:
        """
        Analyze news volume metrics.
        
        Args:
            articles: News articles
            days_back: Analysis period
            
        Returns:
            Volume metrics
        """
        try:
            total_articles = len(articles)
            articles_per_day = total_articles / days_back
            
            # Group by date for daily volume analysis
            daily_counts = {}
            for article in articles:
                date_key = article.published_date.date()
                daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
            
            daily_volumes = list(daily_counts.values())
            
            # Volume statistics
            volume_stats = {
                'total_articles': total_articles,
                'articles_per_day': float(articles_per_day),
                'days_with_news': len(daily_counts),
                'max_daily_volume': max(daily_volumes) if daily_volumes else 0,
                'min_daily_volume': min(daily_volumes) if daily_volumes else 0,
                'avg_daily_volume': float(np.mean(daily_volumes)) if daily_volumes else 0.0,
                'volume_volatility': float(np.std(daily_volumes)) if len(daily_volumes) > 1 else 0.0
            }
            
            # Volume category
            if articles_per_day < 2:
                volume_category = 'low'
            elif articles_per_day < 5:
                volume_category = 'medium'
            else:
                volume_category = 'high'
            
            volume_stats['volume_category'] = volume_category
            
            return volume_stats
            
        except Exception as e:
            logger.error(f"Error analyzing news volume: {str(e)}")
            return {'error': str(e)}
    
    def _generate_trading_features(self, overall_sentiment_score: float,
                                 sentiment_distribution: Dict[str, float],
                                 volatility_metrics: Dict[str, Any],
                                 volume_metrics: Dict[str, Any],
                                 time_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate features for trading models.
        
        Args:
            overall_sentiment_score: Overall sentiment score
            sentiment_distribution: Sentiment distribution
            volatility_metrics: Volatility metrics
            volume_metrics: Volume metrics
            time_analysis: Time analysis results
            
        Returns:
            Trading features dictionary
        """
        try:
            features = {
                # Core sentiment features
                'sentiment_score': float(overall_sentiment_score),
                'sentiment_strength': abs(float(overall_sentiment_score)),
                'positive_ratio': sentiment_distribution.get('POSITIVE', 0.0),
                'negative_ratio': sentiment_distribution.get('NEGATIVE', 0.0),
                'neutral_ratio': sentiment_distribution.get('NEUTRAL', 0.0),
                
                # Volatility features
                'sentiment_volatility': volatility_metrics.get('sentiment_volatility', 0.0),
                'sentiment_range': volatility_metrics.get('sentiment_range', 0.0),
                
                # Volume features
                'news_volume': volume_metrics.get('articles_per_day', 0.0),
                'volume_volatility': volume_metrics.get('volume_volatility', 0.0),
                'coverage_consistency': time_analysis.get('coverage_consistency', 0.0),
                
                # Trend features
                'trend_strength': time_analysis.get('trend_strength', 0.0),
                
                # Composite features
                'sentiment_momentum': 0.0,  # Will be calculated below
                'news_impact_score': 0.0,   # Will be calculated below
            }
            
            # Calculate sentiment momentum
            trend_direction = time_analysis.get('trend_direction', 'stable')
            if trend_direction == 'improving':
                features['sentiment_momentum'] = features['trend_strength']
            elif trend_direction == 'declining':
                features['sentiment_momentum'] = -features['trend_strength']
            else:
                features['sentiment_momentum'] = 0.0
            
            # Calculate news impact score (composite metric)
            impact_components = [
                features['sentiment_strength'] * self.feature_weights['recent_sentiment'],
                abs(features['sentiment_momentum']) * self.feature_weights['sentiment_trend'],
                min(features['news_volume'] / 5.0, 1.0) * self.feature_weights['news_volume'],
                (1.0 - min(features['sentiment_volatility'], 1.0)) * self.feature_weights['sentiment_volatility']
            ]
            
            features['news_impact_score'] = sum(impact_components)
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating trading features: {str(e)}")
            return {}
    
    def _get_neutral_sentiment_result(self, ticker: str, reason: str) -> Dict[str, Any]:
        """
        Get neutral sentiment result for error cases.
        
        Args:
            ticker: Stock ticker
            reason: Reason for neutral result
            
        Returns:
            Neutral sentiment result
        """
        return {
            'ticker': ticker,
            'analysis_timestamp': datetime.now().isoformat(),
            'overall_sentiment': 'NEUTRAL',
            'overall_sentiment_score': 0.0,
            'average_confidence': 0.0,
            'total_articles': 0,
            'sentiment_distribution': {'POSITIVE': 0.0, 'NEUTRAL': 1.0, 'NEGATIVE': 0.0},
            'trading_features': {
                'sentiment_score': 0.0,
                'sentiment_strength': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 1.0,
                'sentiment_volatility': 0.0,
                'news_volume': 0.0,
                'news_impact_score': 0.0,
                'sentiment_momentum': 0.0
            },
            'reason': reason
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached result is still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid
        """
        if cache_key not in self.sentiment_cache:
            return False
        
        cache_time = self.sentiment_cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_ttl
    
    def get_sentiment_features_for_model(self, ticker: str, days_back: int = 7) -> Dict[str, float]:
        """
        Get sentiment features specifically formatted for ML models.
        
        Args:
            ticker: Stock ticker
            days_back: Days to analyze
            
        Returns:
            Dictionary of features for ML models
        """
        try:
            # Get comprehensive sentiment (this will use cache if available)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                sentiment_data = loop.run_until_complete(
                    self.get_comprehensive_sentiment(ticker, days_back)
                )
            finally:
                loop.close()
            
            # Extract trading features
            trading_features = sentiment_data.get('trading_features', {})
            
            # Ensure all required features are present with default values
            required_features = [
                'sentiment_score', 'sentiment_strength', 'positive_ratio',
                'negative_ratio', 'neutral_ratio', 'sentiment_volatility',
                'news_volume', 'news_impact_score', 'sentiment_momentum'
            ]
            
            features = {}
            for feature in required_features:
                features[f'news_{feature}'] = trading_features.get(feature, 0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting sentiment features for model: {str(e)}")
            # Return neutral features
            return {
                'news_sentiment_score': 0.0,
                'news_sentiment_strength': 0.0,
                'news_positive_ratio': 0.0,
                'news_negative_ratio': 0.0,
                'news_neutral_ratio': 1.0,
                'news_sentiment_volatility': 0.0,
                'news_news_volume': 0.0,
                'news_news_impact_score': 0.0,
                'news_sentiment_momentum': 0.0
            }
    
    def clear_cache(self) -> None:
        """
        Clear the sentiment cache.
        """
        self.sentiment_cache.clear()
        logger.info("Sentiment cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        current_time = time.time()
        valid_entries = sum(
            1 for entry in self.sentiment_cache.values()
            if (current_time - entry['timestamp']) < self.cache_ttl
        )
        
        return {
            'total_entries': len(self.sentiment_cache),
            'valid_entries': valid_entries,
            'cache_ttl_minutes': self.cache_ttl / 60,
            'cache_hit_ratio': valid_entries / max(len(self.sentiment_cache), 1)
        }

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Configuration
    config = {
        'fingpt_model': 'ProsusAI/finbert',
        'model_cache_dir': './models/cache',
        'newsapi_key': os.getenv('NEWSAPI_KEY'),
        'alphavantage_key': os.getenv('ALPHAVANTAGE_KEY'),
        'cache_ttl_minutes': 30
    }
    
    # Initialize service
    sentiment_service = SentimentService(config)
    
    # Test sentiment analysis
    ticker = "RELIANCE.NS"
    
    print(f"Testing sentiment analysis for {ticker}...")
    
    # Get comprehensive sentiment
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            sentiment_service.get_comprehensive_sentiment(ticker, days_back=7)
        )
        
        print(f"Overall sentiment: {result['overall_sentiment']}")
        print(f"Sentiment score: {result['overall_sentiment_score']:.3f}")
        print(f"Total articles: {result['total_articles']}")
        print(f"Trading features: {result['trading_features']}")
        
        # Test model features
        model_features = sentiment_service.get_sentiment_features_for_model(ticker)
        print(f"\nModel features: {model_features}")
        
        # Cache stats
        cache_stats = sentiment_service.get_cache_stats()
        print(f"\nCache stats: {cache_stats}")
        
    finally:
        loop.close()