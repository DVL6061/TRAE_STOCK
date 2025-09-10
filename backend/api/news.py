from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import asyncio

# Import news fetching and sentiment analysis utilities
from backend.core.news_processor import (
    fetch_news,
    analyze_sentiment,
    get_news_impact
)
from backend.data.news_fetcher import ComprehensiveNewsFetcher
from backend.app.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Models for request and response
class NewsRequest(BaseModel):
    ticker: Optional[str] = None
    keywords: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: Optional[int] = 10
    include_sentiment: bool = True

class SentimentAnalysisRequest(BaseModel):
    text: str

# Endpoints
@router.post("/search")
async def search_news(request: NewsRequest):
    """Search for financial news based on ticker, keywords, and date range"""
    try:
        # Set default dates if not provided
        if not request.end_date:
            request.end_date = datetime.now().strftime("%Y-%m-%d")
        if not request.start_date:
            # Default to 7 days of news
            start_date = datetime.now() - timedelta(days=7)
            request.start_date = start_date.strftime("%Y-%m-%d")
        
        # Calculate days back for fetcher
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
        days_back = (end_dt - start_dt).days + 1
        
        # Use comprehensive news fetcher if ticker is provided
        if request.ticker:
            settings = get_settings()
            config = {
                'newsapi_key': settings.NEWSAPI_KEY,
                'alphavantage_key': settings.ALPHAVANTAGE_KEY
            }
            
            fetcher = ComprehensiveNewsFetcher(config)
            articles = await asyncio.get_event_loop().run_in_executor(
                None, fetcher.fetch_news, request.ticker, days_back, request.limit or 10
            )
            
            # Convert articles to API format
            news_items = []
            for article in articles:
                item = {
                    "title": article.title,
                    "description": article.content[:200] + "..." if len(article.content) > 200 else article.content,
                    "url": article.url,
                    "source": article.source,
                    "published_date": article.published_date.isoformat(),
                    "ticker": article.ticker
                }
                
                # Add sentiment analysis if requested
                if request.include_sentiment:
                    sentiment_text = article.title + " " + (article.content or "")
                    sentiment = analyze_sentiment(sentiment_text)
                    item["sentiment"] = sentiment
                
                news_items.append(item)
        else:
            # Use core news processor for keyword-based search
            news_items = fetch_news(
                ticker=request.ticker,
                keywords=request.keywords,
                start_date=request.start_date,
                end_date=request.end_date,
                limit=request.limit
            )
            
            # Add sentiment analysis if requested
            if request.include_sentiment and news_items:
                for item in news_items:
                    sentiment_text = item["title"] + " " + item.get("description", "")
                    sentiment = analyze_sentiment(sentiment_text)
                    item["sentiment"] = sentiment
                
        return {"news": news_items}
    except Exception as e:
        logger.error(f"Error searching news: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search news: {str(e)}")

@router.post("/sentiment")
async def analyze_text_sentiment(request: SentimentAnalysisRequest):
    """Analyze sentiment of provided text"""
    try:
        # Call synchronous sentiment analysis function
        sentiment = analyze_sentiment(request.text)
        return sentiment
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze sentiment: {str(e)}")

@router.get("/impact/{ticker}")
async def get_stock_news_impact(ticker: str, days: int = Query(7, ge=1, le=30)):
    """Get the impact of news on a stock over a period of time"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Call synchronous news impact function
        impact = get_news_impact(
            ticker=ticker,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        return impact
    except Exception as e:
        logger.error(f"Error getting news impact for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get news impact: {str(e)}")

@router.get("/sources")
async def get_news_sources():
    """Get list of available news sources"""
    try:
        # This would be implemented to list available news sources
        # For now, return a placeholder
        return {
            "sources": [
                {
                    "id": "cnbc",
                    "name": "CNBC",
                    "url": "https://www.cnbc.com/",
                    "category": "Financial News"
                },
                {
                    "id": "moneycontrol",
                    "name": "Moneycontrol",
                    "url": "https://www.moneycontrol.com/",
                    "category": "Indian Financial News"
                },
                {
                    "id": "economic_times",
                    "name": "Economic Times",
                    "url": "https://economictimes.indiatimes.com/",
                    "category": "Indian Business News"
                },
                {
                    "id": "mint",
                    "name": "Mint",
                    "url": "https://www.livemint.com/",
                    "category": "Indian Business News"
                },
                {
                    "id": "business_standard",
                    "name": "Business Standard",
                    "url": "https://www.business-standard.com/",
                    "category": "Indian Business News"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching news sources: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch news sources: {str(e)}")

