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
from backend.ML_models.sentiment_model import SentimentModel
from core.news_processor import analyze_sentiment, fetch_news, get_news_impact
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
        """Search for financial news with advanced filtering using core news processor."""
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=7)
            
            # Build search query
            query_parts = []
            if ticker:
                query_parts.append(ticker)
            if keywords:
                query_parts.extend(keywords)
            
            query = " ".join(query_parts) if query_parts else "stock market"
            
            # Use core news processor for fetching news
            news_data = await fetch_news(
                query=query,
                sources=[source.value for source in (sources or list(self.sources.keys()))],
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            # Convert to NewsArticle objects
            articles = []
            for item in news_data:
                try:
                    article = NewsArticle(
                        title=item.get('title', ''),
                        content=item.get('content', ''),
                        url=item.get('url', ''),
                        source=NewsSource.MONEYCONTROL,  # Default source
                        published_date=datetime.fromisoformat(item.get('published_date', datetime.now().isoformat())),
                        category=NewsCategory.MARKET_NEWS,
                        keywords=item.get('keywords', [])
                    )
                    
                    # Add sentiment if requested
                    if include_sentiment:
                        article.sentiment_score = item.get('sentiment_score', 0.0)
                        article.sentiment_label = item.get('sentiment_label', 'neutral')
                    
                    articles.append(article)
                except Exception as e:
                    logger.debug(f"Error converting news item to NewsArticle: {str(e)}")
                    continue
            
            return NewsSearchResponse(
                articles=articles,
                total_count=len(articles),
                search_terms=query_parts,
                date_range={
                    'start_date': start_date,
                    'end_date': end_date
                },
                sources_used=[source.value for source in (sources or list(self.sources.keys()))]
            )
            
        except Exception as e:
            logger.error(f"Error in news search: {str(e)}")
            return NewsSearchResponse(
                articles=[],
                total_count=0,
                search_terms=[],
                date_range={'start_date': start_date, 'end_date': end_date},
                sources_used=[]
            )
    
    async def get_sentiment_score(self, ticker: Optional[str] = None) -> float:
        """Get aggregated sentiment score from financial news using core news processor."""
        try:
            cache_key = ticker or 'general'
            
            # Check if cache is valid
            if (cache_key in self._sentiment_cache and 
                cache_key in self._last_update and
                datetime.now() - self._last_update[cache_key] < self._update_interval):
                return self._sentiment_cache[cache_key]
            
            # Use core news processor for news impact analysis
            impact_data = await get_news_impact(ticker or "market", days=3)
            
            # Extract sentiment score from impact analysis
            sentiment_score = impact_data.get('overall_sentiment', 0.0)
            
            # Update cache
            self._sentiment_cache[cache_key] = sentiment_score
            self._last_update[cache_key] = datetime.now()
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error calculating sentiment score: {str(e)}")
            # Return cached sentiment if available
            cache_key = ticker or 'general'
            if cache_key in self._sentiment_cache:
                return self._sentiment_cache[cache_key]
            return 0.0  # Neutral sentiment as fallback
    
    def _fetch_from_source(
        self, 
        source: NewsSource, 
        search_terms: List[str], 
        start_date: datetime, 
        end_date: datetime, 
        limit: int
    ) -> List[NewsArticle]:
        """Fetch news from a specific source with rate limiting and parsing."""
        try:
            articles = []
            source_config = self.sources.get(source)
            if not source_config:
                return articles
            
            # Rate limiting
            time.sleep(self._request_delays.get(source, 1))
            
            for term in search_terms[:3]:  # Limit search terms to avoid too many requests
                try:
                    # Format search URL
                    search_url = source_config['search_url'].format(quote(term))
                    
                    # Make request
                    response = self.session.get(
                        search_url,
                        headers=source_config['headers'],
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    # Parse HTML
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract articles
                    article_elements = soup.select(source_config['selectors']['articles'])[:limit]
                    
                    for element in article_elements:
                        try:
                            # Extract title
                            title_elem = element.select_one(source_config['selectors']['title'])
                            if not title_elem:
                                continue
                            title = title_elem.get_text(strip=True)
                            
                            # Extract link
                            link_elem = element.select_one(source_config['selectors']['link'])
                            if not link_elem:
                                continue
                            link = link_elem.get('href', '')
                            if link.startswith('/'):
                                link = urljoin(source_config['base_url'], link)
                            
                            # Extract date
                            date_elem = element.select_one(source_config['selectors']['date'])
                            published_date = self._parse_date(date_elem.get_text(strip=True) if date_elem else '')
                            
                            # Filter by date range
                            if published_date < start_date or published_date > end_date:
                                continue
                            
                            # Create NewsArticle object
                            article = NewsArticle(
                                title=title,
                                content='',  # Will be fetched separately if needed
                                url=link,
                                source=source,
                                published_date=published_date,
                                category=NewsCategory.MARKET_NEWS,
                                keywords=[term]
                            )
                            
                            articles.append(article)
                            
                        except Exception as e:
                            logger.debug(f"Error parsing article element: {str(e)}")
                            continue
                    
                except Exception as e:
                    logger.error(f"Error fetching from {source.value} with term '{term}': {str(e)}")
                    continue
            
            return articles[:limit]
            
        except Exception as e:
            logger.error(f"Error in _fetch_from_source for {source.value}: {str(e)}")
            return []
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string from various formats used by news sources."""
        if not date_str:
            return datetime.now()
        
        # Common date formats used by Indian financial news sites
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%d-%m-%Y %H:%M:%S',
            '%d-%m-%Y',
            '%d %b %Y %H:%M',
            '%d %B %Y %H:%M',
            '%d %b %Y',
            '%d %B %Y',
            '%b %d, %Y %H:%M',
            '%B %d, %Y %H:%M',
            '%b %d, %Y',
            '%B %d, %Y'
        ]
        
        # Clean the date string
        date_str = re.sub(r'\s+', ' ', date_str.strip())
        date_str = re.sub(r'[^\w\s:,-]', '', date_str)
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try to extract relative dates (e.g., "2 hours ago", "1 day ago")
        relative_patterns = [
            (r'(\d+)\s*hour[s]?\s*ago', lambda m: datetime.now() - timedelta(hours=int(m.group(1)))),
            (r'(\d+)\s*day[s]?\s*ago', lambda m: datetime.now() - timedelta(days=int(m.group(1)))),
            (r'(\d+)\s*minute[s]?\s*ago', lambda m: datetime.now() - timedelta(minutes=int(m.group(1)))),
            (r'yesterday', lambda m: datetime.now() - timedelta(days=1)),
            (r'today', lambda m: datetime.now())
        ]
        
        for pattern, func in relative_patterns:
            match = re.search(pattern, date_str.lower())
            if match:
                return func(match)
        
        # Default to current time if parsing fails
        logger.warning(f"Could not parse date: {date_str}")
        return datetime.now()
    
    def _remove_duplicates(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity."""
        if not articles:
            return articles
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Normalize title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', article.title.lower())
            normalized_title = re.sub(r'\s+', ' ', normalized_title).strip()
            
            # Check for similarity with existing titles
            is_duplicate = False
            for seen_title in seen_titles:
                # Simple similarity check - if 80% of words match
                title_words = set(normalized_title.split())
                seen_words = set(seen_title.split())
                
                if len(title_words) > 0 and len(seen_words) > 0:
                    intersection = len(title_words.intersection(seen_words))
                    union = len(title_words.union(seen_words))
                    similarity = intersection / union
                    
                    if similarity > 0.8:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(normalized_title)
        
        return unique_articles
    
    async def _add_sentiment_analysis(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Add sentiment analysis to news articles using core news processor"""
        analyzed_articles = []
        
        for article in articles:
            try:
                # Combine title and content for analysis
                text = f"{article.title} {article.content}"
                
                # Use core news processor for sentiment analysis
                sentiment_result = analyze_sentiment(text)
                
                # Add sentiment data to article
                article.sentiment_score = sentiment_result.get('compound', 0.0)
                article.sentiment_label = sentiment_result.get('label', 'neutral')
                
                analyzed_articles.append(article)
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment: {str(e)}")
                # Add neutral sentiment as fallback
                article.sentiment_score = 0.0
                article.sentiment_label = 'neutral'
                analyzed_articles.append(article)
        
        return analyzed_articles
    
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