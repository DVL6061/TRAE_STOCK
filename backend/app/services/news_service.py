import aiohttp
import asyncio
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from urllib.parse import urljoin, quote
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our models
from app.models.news import (
    NewsArticle, NewsWithSentiment, NewsSummary, 
    NewsSource, NewsCategory, SentimentLabel,
    NewsSearchRequest, NewsSearchResponse
)
from app.models.prediction import SentimentAnalysis
from backend.models.sentiment_model import SentimentModel
from app.config import NEWS_SOURCES

logger = logging.getLogger(__name__)

class NewsService:
    """Enhanced news service for collecting and analyzing financial news."""
    
    def __init__(self):
        # Initialize sentiment analyzer
        self.sentiment_model = SentimentModel()
        
        # News sources configuration with detailed scraping info
        self.sources = {
            NewsSource.MONEYCONTROL: {
                'base_url': 'https://www.moneycontrol.com',
                'search_url': 'https://www.moneycontrol.com/news/tags/{}.html',
                'selectors': {
                    'articles': '.news_common, .clearfix',
                    'title': 'h2 a, h3 a, .news_title a',
                    'link': 'h2 a, h3 a, .news_title a',
                    'date': '.article_schedule, .news_date',
                    'content': '.arti-flow p, .content_wrapper p'
                },
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            },
            NewsSource.ECONOMIC_TIMES: {
                'base_url': 'https://economictimes.indiatimes.com',
                'search_url': 'https://economictimes.indiatimes.com/topic/{}',
                'selectors': {
                    'articles': '.eachStory, .story-box',
                    'title': 'h3 a, h4 a, .story-title a',
                    'link': 'h3 a, h4 a, .story-title a',
                    'date': '.time, .story-date',
                    'content': '.Normal p, .article_content p'
                },
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            },
            NewsSource.MINT: {
                'base_url': 'https://www.livemint.com',
                'search_url': 'https://www.livemint.com/search?q={}',
                'selectors': {
                    'articles': '.searchResult, .listingNew',
                    'title': '.headline a, .headlineLink',
                    'link': '.headline a, .headlineLink',
                    'date': '.publish_on, .dateTime',
                    'content': '.mainSectionContent p, .paywall p'
                },
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            },
            NewsSource.CNBC: {
                'base_url': 'https://www.cnbc.com',
                'search_url': 'https://www.cnbc.com/search/?query={}',
                'selectors': {
                    'articles': '.SearchResult, .Card',
                    'title': '.SearchResult-headline, .Card-title',
                    'link': '.SearchResult-headline, .Card-title',
                    'date': '.SearchResult-publishedDate, .Card-time',
                    'content': '.ArticleBody-articleBody p, .InlineContent p'
                },
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            }
        }
        
        # Cache for news and sentiment
        self._news_cache = {}
        self._sentiment_cache = {}
        self._last_update = {}
        
        # Update interval (15 minutes)
        self._update_interval = timedelta(minutes=15)
        
        # Rate limiting
        self._request_delays = {
            NewsSource.MONEYCONTROL: 2,
            NewsSource.ECONOMIC_TIMES: 1.5,
            NewsSource.MINT: 2,
            NewsSource.CNBC: 1
        }
        
        # Session for requests
        self.session = requests.Session()
    
    async def search_news(
        self, 
        ticker: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        sources: Optional[List[NewsSource]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        include_sentiment: bool = True
    ) -> NewsSearchResponse:
        """Search for financial news with advanced filtering."""
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=7)
            
            # Use all sources if none specified
            if not sources:
                sources = list(self.sources.keys())
            
            # Build search terms
            search_terms = []
            if ticker:
                # Add variations of ticker
                search_terms.extend([
                    ticker,
                    ticker.replace('.NS', ''),
                    ticker.replace('.BO', ''),
                    ticker.replace('_', ' ')
                ])
            if keywords:
                search_terms.extend(keywords)
            
            # Fetch news from all sources
            all_articles = []
            with ThreadPoolExecutor(max_workers=len(sources)) as executor:
                futures = {
                    executor.submit(
                        self._fetch_from_source, 
                        source, 
                        search_terms, 
                        start_date, 
                        end_date, 
                        limit // len(sources)
                    ): source for source in sources
                }
                
                for future in as_completed(futures, timeout=60):
                    try:
                        articles = future.result()
                        all_articles.extend(articles)
                    except Exception as e:
                        source = futures[future]
                        logger.error(f"Error fetching from {source}: {str(e)}")
            
            # Remove duplicates based on title similarity
            unique_articles = self._remove_duplicates(all_articles)
            
            # Sort by date (newest first)
            unique_articles.sort(key=lambda x: x.published_date, reverse=True)
            
            # Limit results
            unique_articles = unique_articles[:limit]
            
            # Add sentiment analysis if requested
            if include_sentiment:
                unique_articles = await self._add_sentiment_analysis(unique_articles)
            
            return NewsSearchResponse(
                articles=unique_articles,
                total_count=len(unique_articles),
                search_terms=search_terms,
                date_range={
                    'start_date': start_date,
                    'end_date': end_date
                },
                sources_used=[s.value for s in sources]
            )
            
        except Exception as e:
            logger.error(f"Error in news search: {str(e)}")
            return NewsSearchResponse(
                articles=[],
                total_count=0,
                search_terms=search_terms if 'search_terms' in locals() else [],
                date_range={'start_date': start_date, 'end_date': end_date},
                sources_used=[]
            )
    
    async def get_sentiment_score(self, ticker: Optional[str] = None) -> float:
        """Get aggregated sentiment score from financial news."""
        try:
            cache_key = ticker or 'general'
            
            # Check if cache is valid
            if (cache_key in self._sentiment_cache and 
                cache_key in self._last_update and
                datetime.now() - self._last_update[cache_key] < self._update_interval):
                return self._sentiment_cache[cache_key]
            
            # Fetch recent news
            news_response = await self.search_news(
                ticker=ticker,
                start_date=datetime.now() - timedelta(days=3),
                limit=20,
                include_sentiment=True
            )
            
            if not news_response.articles:
                return 0.0
            
            # Calculate weighted sentiment score
            sentiment_scores = []
            for article in news_response.articles:
                if hasattr(article, 'sentiment') and article.sentiment:
                    # Weight by recency (more recent = higher weight)
                    hours_old = (datetime.now() - article.published_date).total_seconds() / 3600
                    weight = max(0.1, 1.0 - (hours_old / 72))  # Decay over 3 days
                    
                    sentiment_scores.append({
                        'score': article.sentiment.sentiment_score,
                        'weight': weight,
                        'confidence': article.sentiment.confidence
                    })
            
            if not sentiment_scores:
                return 0.0
            
            # Calculate weighted average
            total_weight = sum(s['weight'] * s['confidence'] for s in sentiment_scores)
            weighted_sum = sum(s['score'] * s['weight'] * s['confidence'] for s in sentiment_scores)
            
            weighted_sentiment = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Update cache
            self._sentiment_cache[cache_key] = weighted_sentiment
            self._last_update[cache_key] = datetime.now()
            
            return weighted_sentiment
            
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {str(e)}")
            # Return cached sentiment if available
            cache_key = ticker or 'general'
            if cache_key in self._sentiment_cache:
                return self._sentiment_cache[cache_key]
            return 0.0  # Neutral sentiment as fallback
    
    async def _fetch_news(self) -> List[Dict]:
        """Fetch news from multiple sources"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source in self.sources:
                tasks.append(self._fetch_source(session, source))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out errors and combine results
            news_articles = []
            for result in results:
                if isinstance(result, list):
                    news_articles.extend(result)
            
            return news_articles
    
    async def _fetch_source(self, session: aiohttp.ClientSession, source: Dict) -> List[Dict]:
        """Fetch news from a single source"""
        try:
            async with session.get(source['url']) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                articles = []
                for article in soup.select(source['article_selector']):
                    title = article.select_one(source['title_selector']).text.strip()
                    link = article.select_one(source['link_selector'])['href']
                    timestamp = self._parse_timestamp(
                        article.select_one(source['timestamp_selector']).text.strip(),
                        source['timestamp_format']
                    )
                    
                    # Get article content if needed
                    content = await self._fetch_article_content(session, link, source)
                    
                    articles.append({
                        'title': title,
                        'content': content,
                        'link': link,
                        'timestamp': timestamp,
                        'source': source['name']
                    })
                
                return articles
                
        except Exception as e:
            print(f"Error fetching from {source['name']}: {str(e)}")
            return []
    
    async def _fetch_article_content(self, session: aiohttp.ClientSession,
                                   url: str, source: Dict) -> str:
        """Fetch and extract content from an article"""
        try:
            async with session.get(url) as response:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                content_elements = soup.select(source['content_selector'])
                content = ' '.join(elem.text.strip() for elem in content_elements)
                
                return content
                
        except Exception as e:
            print(f"Error fetching article content: {str(e)}")
            return ""
    
    async def _analyze_sentiment(self, articles: List[Dict]) -> List[Dict]:
        """Analyze sentiment of news articles using FinGPT"""
        sentiment_scores = []
        
        for article in articles:
            # Combine title and content for analysis
            text = f"{article['title']}. {article['content']}"
            
            # Get sentiment prediction
            sentiment = self.sentiment_analyzer(text)[0]
            
            sentiment_scores.append({
                'score': self._normalize_sentiment_score(sentiment),
                'timestamp': article['timestamp'],
                'importance': self._calculate_importance(article)
            })
        
        return sentiment_scores
    
    def _normalize_sentiment_score(self, sentiment: Dict) -> float:
        """Normalize sentiment score to range [-1, 1]"""
        label = sentiment['label']
        score = sentiment['score']
        
        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        return 0.0
    
    def _calculate_importance(self, article: Dict) -> float:
        """Calculate importance weight for an article"""
        # Implement importance calculation based on:
        # - Source reliability
        # - Article freshness
        # - Content length
        # - Keyword relevance
        pass
    
    def _calculate_weighted_sentiment(self, sentiment_scores: List[Dict]) -> float:
        """Calculate weighted average sentiment score"""
        if not sentiment_scores:
            return 0.0
        
        total_weight = sum(score['importance'] for score in sentiment_scores)
        weighted_sum = sum(score['score'] * score['importance'] 
                          for score in sentiment_scores)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _parse_timestamp(self, timestamp_str: str, timestamp_format: str) -> datetime:
        """Parse timestamp string to datetime object"""
        try:
            return datetime.strptime(timestamp_str, timestamp_format)
        except Exception:
            return datetime.now()