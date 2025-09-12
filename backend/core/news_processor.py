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
    """Fetch financial news based on ticker, keywords, and date range"""
    try:
        # In a real implementation, this would scrape news from various sources
        # or use a news API. For now, we'll use mock data.
        
        # Filter by date range if provided
        filtered_news = MOCK_TATA_MOTORS_NEWS.copy()
        if start_date:
            start = datetime.fromisoformat(start_date)
            filtered_news = [news for news in filtered_news 
                            if datetime.fromisoformat(news["published_at"]) >= start]
        if end_date:
            end = datetime.fromisoformat(end_date)
            filtered_news = [news for news in filtered_news 
                            if datetime.fromisoformat(news["published_at"]) <= end]
        
        # Filter by ticker if provided
        if ticker and ticker.upper() != "TATAMOTORS.NS":
            # For now, we only have mock data for Tata Motors
            return []
        
        # Filter by keywords if provided
        if keywords:
            keywords_lower = [k.lower() for k in keywords]
            filtered_news = [
                news for news in filtered_news 
                if any(k.lower() in news["title"].lower() or 
                       k.lower() in news["description"].lower() 
                       for k in keywords_lower)
            ]
        
        # Sort by date (newest first) and limit results
        filtered_news.sort(
            key=lambda x: datetime.fromisoformat(x["published_at"]),
            reverse=True
        )
        if limit:
            filtered_news = filtered_news[:limit]
        
        # Add source name from source ID
        for news in filtered_news:
            news["source_name"] = MOCK_NEWS_SOURCES.get(news["source"], news["source"])
        
        return filtered_news
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        raise Exception(f"Failed to fetch news: {str(e)}")

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
    """Analyze sentiment of provided text using FinGPT"""
    try:
        analyzer = get_sentiment_analyzer()
        
        if analyzer is None:
            # Ultimate fallback to mock analysis
            logger.warning("No sentiment analyzer available, using mock analysis")
            return await mock_analyze_sentiment(text)
        
        # Use FinGPT analyzer if available
        if hasattr(analyzer, 'analyze_text_sentiment'):
            # FinGPTSentimentAnalyzer
            result = analyzer.analyze_text_sentiment(text)
            
            # Convert FinGPT result format to our standard format
            return {
                "score": result.get('score', 0.0),
                "label": result.get('sentiment', 'neutral').lower(),
                "confidence": result.get('confidence', 0.0),
                "method": result.get('method', 'fingpt'),
                "positive_aspects": result.get('component_results', {}).get('keywords', {}).get('positive_count', 0),
                "negative_aspects": result.get('component_results', {}).get('keywords', {}).get('negative_count', 0)
            }
        else:
            # SentimentModel fallback
            result = analyzer.analyze_sentiment(text)
            
            # Convert SentimentModel result format to our standard format
            sentiment_score = result.get('sentiment_score', 0.0)
            return {
                "score": sentiment_score,
                "label": result.get('sentiment', 'neutral'),
                "confidence": result.get('confidence', 0.0),
                "method": "finbert",
                "positive_aspects": [],
                "negative_aspects": []
            }
            
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