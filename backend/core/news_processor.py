import requests
import pandas as pd
import numpy as np
import logging
import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import re
from bs4 import BeautifulSoup

# Import real news fetching and sentiment analysis models
from data.news_fetcher import NewsAPIFetcher, NewsDatabase, NewsArticle
from ML_models.sentiment_model import SentimentModel
from ML_models.fingpt_model import FinGPTSentimentAnalyzer

logger = logging.getLogger(__name__)

# Initialize news database and fetchers
news_db = NewsDatabase()
news_api_key = os.getenv('NEWS_API_KEY')
news_api_fetcher = NewsAPIFetcher(news_api_key) if news_api_key else None

# News sources configuration
NEWS_SOURCES = {
    "newsapi": "NewsAPI",
    "cnbc": "CNBC", 
    "moneycontrol": "Moneycontrol",
    "economic_times": "Economic Times",
    "mint": "Mint",
    "business_standard": "Business Standard"
}

# Mock sentiment analysis function
async def mock_analyze_sentiment(text: str) -> Dict[str, Any]:
    """Mock sentiment analysis function for development"""
    # Simple keyword-based sentiment analysis
    positive_keywords = [
        "growth", "increase", "surge", "positive", "strong", "launch", "partnership",
        "strategic", "opportunity", "profit", "upgrade", "success", "innovation"
    ]
    negative_keywords = [
        "shortage", "impact", "affect", "decline", "decrease", "challenge", "problem",
        "issue", "concern", "risk", "downgrade", "loss", "fail", "delay"
    ]
    
    # Count occurrences of positive and negative keywords
    text_lower = text.lower()
    positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    
    # Calculate sentiment score (-1 to 1)
    total_count = positive_count + negative_count
    if total_count == 0:
        sentiment_score = 0.0
    else:
        sentiment_score = (positive_count - negative_count) / total_count
    
    # Determine sentiment label
    if sentiment_score > 0.2:
        sentiment_label = "positive"
    elif sentiment_score < -0.2:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    
    # Return sentiment analysis result
    return {
        "score": round(sentiment_score, 2),
        "label": sentiment_label,
        "confidence": 0.7 + abs(sentiment_score) * 0.3,  # Mock confidence score
        "positive_aspects": [keyword for keyword in positive_keywords if keyword in text_lower],
        "negative_aspects": [keyword for keyword in negative_keywords if keyword in text_lower]
    }

async def fetch_news(
    ticker: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = 10
) -> List[Dict[str, Any]]:
    """Fetch financial news using real NewsAPI integration"""
    try:
        # Parse date parameters
        from_date = None
        to_date = None
        
        if start_date:
            from_date = datetime.fromisoformat(start_date)
        else:
            from_date = datetime.now() - timedelta(days=7)  # Default to 7 days back
            
        if end_date:
            to_date = datetime.fromisoformat(end_date)
        else:
            to_date = datetime.now()
        
        # Try to get cached articles first
        if ticker:
            cached_articles = news_db.get_articles(ticker, from_date)
            
            if cached_articles and len(cached_articles) >= (limit or 10):
                logger.info(f"Using cached news for {ticker}: {len(cached_articles)} articles")
                formatted_articles = []
                
                for article in cached_articles[:(limit or 10)]:
                    # Filter by date range if specified
                    article_date = article.published_at
                    if article_date >= from_date and article_date <= to_date:
                        formatted_articles.append({
                            'title': article.title,
                            'description': article.description,
                            'source': article.source,
                            'source_name': NEWS_SOURCES.get(article.source, article.source),
                            'url': article.url,
                            'published_at': article.published_at.isoformat(),
                            'keywords': article.keywords.split(',') if article.keywords else [],
                            'sentiment_score': article.sentiment_score
                        })
                
                # Filter by keywords if provided
                if keywords:
                    keywords_lower = [k.lower() for k in keywords]
                    formatted_articles = [
                        article for article in formatted_articles
                        if any(k in article['title'].lower() or k in article['description'].lower() 
                               for k in keywords_lower)
                    ]
                
                if formatted_articles:
                    return formatted_articles
        
        # Fetch fresh articles if cache is insufficient
        if news_api_fetcher and ticker:
            logger.info(f"Fetching fresh news for {ticker} from NewsAPI")
            fresh_articles = news_api_fetcher.fetch_articles(
                ticker=ticker,
                from_date=from_date,
                limit=(limit or 10) * 2  # Fetch more to account for filtering
            )
            
            # Save to cache and return formatted results
            formatted_articles = []
            for article in fresh_articles:
                # Filter by date range
                if article.published_at >= from_date and article.published_at <= to_date:
                    # Save to database
                    news_db.save_article(article)
                    
                    # Format for response
                    formatted_article = {
                        'title': article.title,
                        'description': article.description,
                        'source': article.source,
                        'source_name': NEWS_SOURCES.get(article.source, article.source),
                        'url': article.url,
                        'published_at': article.published_at.isoformat(),
                        'keywords': article.keywords.split(',') if article.keywords else [],
                        'sentiment_score': article.sentiment_score
                    }
                    
                    # Filter by keywords if provided
                    if keywords:
                        keywords_lower = [k.lower() for k in keywords]
                        if any(k in formatted_article['title'].lower() or 
                               k in formatted_article['description'].lower() 
                               for k in keywords_lower):
                            formatted_articles.append(formatted_article)
                    else:
                        formatted_articles.append(formatted_article)
            
            # Sort by date (newest first) and limit results
            formatted_articles.sort(
                key=lambda x: datetime.fromisoformat(x['published_at']),
                reverse=True
            )
            
            return formatted_articles[:(limit or 10)]
        
        # Fallback: try cached articles without ticker filtering
        if not ticker and news_db:
            logger.warning("No ticker specified, fetching general cached articles")
            # This would require a different database method for general queries
            # For now, return empty list
            return []
        
        logger.warning(f"No news data available for ticker: {ticker}")
        return []
        
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return []  # Return empty list instead of raising exception

# Initialize FinGPT sentiment analyzer (singleton pattern)
_sentiment_analyzer = None

def get_sentiment_analyzer():
    """Get or create FinGPT sentiment analyzer instance"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            _sentiment_analyzer = FinGPTSentimentAnalyzer()
            logger.info("FinGPT sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FinGPT analyzer: {str(e)}")
            # Fallback to basic sentiment model
            try:
                _sentiment_analyzer = SentimentModel()
                logger.info("Fallback sentiment model initialized")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback model: {str(e2)}")
                _sentiment_analyzer = None
    return _sentiment_analyzer

async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of provided text using FinGPT or SentimentModel"""
    try:
        analyzer = get_sentiment_analyzer()
        
        if analyzer is None:
            # Ultimate fallback to mock analysis
            logger.warning("No sentiment analyzer available, using mock analysis")
            return await mock_analyze_sentiment(text)
        
        # Use FinGPT analyzer if available
        if isinstance(analyzer, FinGPTSentimentAnalyzer):
            # FinGPTSentimentAnalyzer
            result = analyzer.analyze_text_sentiment(text)
            
            # Convert FinGPT result format to our standard format
            return {
                "score": result.get('score', 0.0),
                "label": result.get('sentiment', 'neutral').lower(),
                "confidence": result.get('confidence', 0.0),
                "method": "fingpt",
                "positive_aspects": result.get('component_results', {}).get('keywords', {}).get('positive_keywords', []),
                "negative_aspects": result.get('component_results', {}).get('keywords', {}).get('negative_keywords', [])
            }
        elif isinstance(analyzer, SentimentModel):
            # SentimentModel fallback
            result = analyzer.analyze_sentiment(text)
            
            # Convert SentimentModel result format to our standard format
            return {
                "score": result.get('score', 0.0),
                "label": result.get('sentiment', 'neutral').lower(),
                "confidence": result.get('confidence', 0.0),
                "method": "finbert",
                "positive_aspects": [],
                "negative_aspects": []
            }
        else:
            # Unknown analyzer type, fallback to mock
            logger.warning(f"Unknown analyzer type: {type(analyzer)}, using mock analysis")
            return await mock_analyze_sentiment(text)
            
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        # Fallback to mock analysis on error
        return await mock_analyze_sentiment(text)

async def get_news_impact(
    ticker: str,
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """Get the impact of news on a stock over a period of time"""
    try:
        # Fetch news for the specified ticker and date range
        news_items = await fetch_news(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            limit=None  # No limit for impact analysis
        )
        
        # Analyze sentiment for each news item
        for item in news_items:
            sentiment = await analyze_sentiment(item["title"] + " " + item["description"])
            item["sentiment"] = sentiment
        
        # Calculate overall sentiment metrics
        sentiment_scores = [item["sentiment"]["score"] for item in news_items]
        sentiment_labels = [item["sentiment"]["label"] for item in news_items]
        
        # Count sentiment labels
        positive_count = sentiment_labels.count("positive")
        neutral_count = sentiment_labels.count("neutral")
        negative_count = sentiment_labels.count("negative")
        total_count = len(sentiment_labels)
        
        # Calculate average sentiment score
        avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Group news by date for timeline analysis
        news_by_date = {}
        for item in news_items:
            date = datetime.fromisoformat(item["published_at"]).strftime("%Y-%m-%d")
            if date not in news_by_date:
                news_by_date[date] = []
            news_by_date[date].append(item)
        
        # Calculate daily sentiment scores
        daily_sentiment = []
        for date, items in news_by_date.items():
            scores = [item["sentiment"]["score"] for item in items]
            avg_score = sum(scores) / len(scores)
            daily_sentiment.append({
                "date": date,
                "score": avg_score,
                "news_count": len(items)
            })
        
        # Sort daily sentiment by date
        daily_sentiment.sort(key=lambda x: x["date"])
        
        # Return news impact analysis
        return {
            "ticker": ticker,
            "period": f"{start_date} to {end_date}",
            "news_count": total_count,
            "sentiment_distribution": {
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count
            },
            "sentiment_percentage": {
                "positive": round(positive_count / total_count * 100, 2) if total_count > 0 else 0,
                "neutral": round(neutral_count / total_count * 100, 2) if total_count > 0 else 0,
                "negative": round(negative_count / total_count * 100, 2) if total_count > 0 else 0
            },
            "average_sentiment_score": round(avg_sentiment_score, 2),
            "daily_sentiment": daily_sentiment,
            "top_positive_news": sorted(
                [item for item in news_items if item["sentiment"]["label"] == "positive"],
                key=lambda x: x["sentiment"]["score"],
                reverse=True
            )[:3],
            "top_negative_news": sorted(
                [item for item in news_items if item["sentiment"]["label"] == "negative"],
                key=lambda x: x["sentiment"]["score"]
            )[:3]
        }
    except Exception as e:
        logger.error(f"Error getting news impact for {ticker}: {str(e)}")
        raise Exception(f"Failed to get news impact: {str(e)}")