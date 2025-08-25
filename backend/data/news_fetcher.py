import os
import logging
import requests
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, quote
import hashlib
from dataclasses import dataclass
from functools import lru_cache
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data class for news articles."""
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    ticker: str
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    article_id: Optional[str] = None

class NewsDatabase:
    """SQLite database for caching news articles."""
    
    def __init__(self, db_path: str = "./data/news_cache.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the news database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS news_articles (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        content TEXT,
                        url TEXT UNIQUE,
                        source TEXT,
                        published_date TIMESTAMP,
                        ticker TEXT,
                        sentiment_score REAL,
                        relevance_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(url, ticker)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ticker_date 
                    ON news_articles(ticker, published_date)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_source_date 
                    ON news_articles(source, published_date)
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing news database: {str(e)}")
    
    def save_article(self, article: NewsArticle) -> bool:
        """Save an article to the database."""
        try:
            article_id = hashlib.md5(f"{article.url}_{article.ticker}".encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO news_articles 
                    (id, title, content, url, source, published_date, ticker, 
                     sentiment_score, relevance_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article_id, article.title, article.content, article.url,
                    article.source, article.published_date, article.ticker,
                    article.sentiment_score, article.relevance_score
                ))
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving article: {str(e)}")
            return False
    
    def get_articles(self, ticker: str, days_back: int = 7, 
                    limit: int = 100) -> List[NewsArticle]:
        """Retrieve articles from the database."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM news_articles 
                    WHERE ticker = ? AND published_date >= ?
                    ORDER BY published_date DESC
                    LIMIT ?
                """, (ticker, cutoff_date, limit))
                
                articles = []
                for row in cursor.fetchall():
                    article = NewsArticle(
                        title=row['title'],
                        content=row['content'],
                        url=row['url'],
                        source=row['source'],
                        published_date=datetime.fromisoformat(row['published_date']),
                        ticker=row['ticker'],
                        sentiment_score=row['sentiment_score'],
                        relevance_score=row['relevance_score'],
                        article_id=row['id']
                    )
                    articles.append(article)
                
                return articles
                
        except Exception as e:
            logger.error(f"Error retrieving articles: {str(e)}")
            return []

class NewsAPIFetcher:
    """Fetcher for NewsAPI.org."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'User-Agent': 'FinancialNewsAnalyzer/1.0'
        })
    
    def fetch_articles(self, ticker: str, days_back: int = 7, 
                      max_articles: int = 50) -> List[NewsArticle]:
        """Fetch articles from NewsAPI."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search queries for the ticker
            queries = [
                f'"{ticker}"',
                f'{ticker.replace(".NS", "")} AND (stock OR shares OR earnings)',
                f'{ticker.replace(".BO", "")} AND (financial OR market)'
            ]
            
            all_articles = []
            
            for query in queries:
                params = {
                    'q': query,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': min(100, max_articles // len(queries))
                }
                
                response = self.session.get(
                    f"{self.base_url}/everything",
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for article_data in data.get('articles', []):
                        if self._is_relevant_article(article_data, ticker):
                            article = self._parse_newsapi_article(article_data, ticker)
                            if article:
                                all_articles.append(article)
                
                # Rate limiting
                time.sleep(0.1)
            
            return self._deduplicate_articles(all_articles)[:max_articles]
            
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {str(e)}")
            return []
    
    def _is_relevant_article(self, article_data: Dict, ticker: str) -> bool:
        """Check if article is relevant to the ticker."""
        title = article_data.get('title', '').lower()
        description = article_data.get('description', '').lower()
        
        ticker_variants = [
            ticker.lower(),
            ticker.replace('.ns', '').lower(),
            ticker.replace('.bo', '').lower()
        ]
        
        for variant in ticker_variants:
            if variant in title or variant in description:
                return True
        
        return False
    
    def _parse_newsapi_article(self, article_data: Dict, ticker: str) -> Optional[NewsArticle]:
        """Parse NewsAPI article data."""
        try:
            published_date = datetime.fromisoformat(
                article_data['publishedAt'].replace('Z', '+00:00')
            )
            
            return NewsArticle(
                title=article_data['title'],
                content=article_data.get('description', '') + ' ' + 
                       article_data.get('content', ''),
                url=article_data['url'],
                source=article_data['source']['name'],
                published_date=published_date,
                ticker=ticker
            )
            
        except Exception as e:
            logger.error(f"Error parsing NewsAPI article: {str(e)}")
            return None
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles."""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return sorted(unique_articles, key=lambda x: x.published_date, reverse=True)

class AlphaVantageNewsFetcher:
    """Fetcher for Alpha Vantage News API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()
    
    def fetch_articles(self, ticker: str, days_back: int = 7, 
                      max_articles: int = 50) -> List[NewsArticle]:
        """Fetch articles from Alpha Vantage."""
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.api_key,
                'limit': max_articles
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                articles = []
                for item in data.get('feed', []):
                    article = self._parse_alphavantage_article(item, ticker)
                    if article and self._is_within_date_range(article, days_back):
                        articles.append(article)
                
                return articles[:max_articles]
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {str(e)}")
            return []
    
    def _parse_alphavantage_article(self, item: Dict, ticker: str) -> Optional[NewsArticle]:
        """Parse Alpha Vantage article data."""
        try:
            # Parse timestamp
            timestamp_str = item.get('time_published', '')
            published_date = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
            
            # Calculate relevance score
            relevance_score = 0.0
            for ticker_sentiment in item.get('ticker_sentiment', []):
                if ticker_sentiment.get('ticker') == ticker:
                    relevance_score = float(ticker_sentiment.get('relevance_score', 0.0))
                    break
            
            return NewsArticle(
                title=item.get('title', ''),
                content=item.get('summary', ''),
                url=item.get('url', ''),
                source=item.get('source', 'Alpha Vantage'),
                published_date=published_date,
                ticker=ticker,
                relevance_score=relevance_score
            )
            
        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage article: {str(e)}")
            return None
    
    def _is_within_date_range(self, article: NewsArticle, days_back: int) -> bool:
        """Check if article is within the specified date range."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        return article.published_date >= cutoff_date

class WebScrapingFetcher:
    """Web scraping fetcher for Indian financial news sites."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.sources = {
            'moneycontrol': {
                'search_url': 'https://www.moneycontrol.com/news/tags/{}.html',
                'selectors': {
                    'articles': '.news_common',
                    'title': 'h2 a, h3 a',
                    'link': 'h2 a, h3 a',
                    'date': '.article_schedule'
                }
            },
            'economictimes': {
                'search_url': 'https://economictimes.indiatimes.com/topic/{}',
                'selectors': {
                    'articles': '.eachStory',
                    'title': 'h3 a, h4 a',
                    'link': 'h3 a, h4 a',
                    'date': '.time'
                }
            },
            'livemint': {
                'search_url': 'https://www.livemint.com/search?q={}',
                'selectors': {
                    'articles': '.searchResult',
                    'title': '.headline a',
                    'link': '.headline a',
                    'date': '.publish_on'
                }
            }
        }
    
    def fetch_articles(self, ticker: str, days_back: int = 7, 
                      max_articles: int = 50) -> List[NewsArticle]:
        """Fetch articles using web scraping."""
        all_articles = []
        
        # Clean ticker for search
        search_term = ticker.replace('.NS', '').replace('.BO', '')
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for source_name, source_config in self.sources.items():
                future = executor.submit(
                    self._scrape_source,
                    source_name,
                    source_config,
                    search_term,
                    ticker,
                    max_articles // len(self.sources)
                )
                futures.append(future)
            
            for future in as_completed(futures, timeout=60):
                try:
                    articles = future.result()
                    all_articles.extend(articles)
                except Exception as e:
                    logger.warning(f"Scraping failed: {str(e)}")
        
        # Filter by date and deduplicate
        filtered_articles = self._filter_by_date(all_articles, days_back)
        return self._deduplicate_articles(filtered_articles)[:max_articles]
    
    def _scrape_source(self, source_name: str, source_config: Dict, 
                      search_term: str, ticker: str, max_articles: int) -> List[NewsArticle]:
        """Scrape articles from a specific source."""
        articles = []
        
        try:
            url = source_config['search_url'].format(quote(search_term))
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                selectors = source_config['selectors']
                
                article_elements = soup.select(selectors['articles'])[:max_articles]
                
                for element in article_elements:
                    try:
                        title_elem = element.select_one(selectors['title'])
                        link_elem = element.select_one(selectors['link'])
                        date_elem = element.select_one(selectors['date'])
                        
                        if title_elem and link_elem:
                            title = title_elem.get_text(strip=True)
                            link = link_elem.get('href')
                            
                            # Make absolute URL
                            if link and not link.startswith('http'):
                                base_url = f"https://{source_name}.com"
                                link = urljoin(base_url, link)
                            
                            # Parse date (simplified)
                            published_date = self._parse_date(date_elem, source_name)
                            
                            # Get article content
                            content = self._get_article_content(link)
                            
                            article = NewsArticle(
                                title=title,
                                content=content,
                                url=link,
                                source=source_name,
                                published_date=published_date,
                                ticker=ticker
                            )
                            
                            articles.append(article)
                            
                    except Exception as e:
                        logger.warning(f"Error parsing article element: {str(e)}")
                        continue
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {str(e)}")
        
        return articles
    
    def _parse_date(self, date_elem, source_name: str) -> datetime:
        """Parse date from different sources."""
        try:
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                # Simplified date parsing - in production, use more robust parsing
                # This is a placeholder implementation
                return datetime.now() - timedelta(hours=1)
            else:
                return datetime.now()
        except:
            return datetime.now()
    
    def _get_article_content(self, url: str) -> str:
        """Get full article content from URL."""
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Common content selectors
                content_selectors = [
                    '.article-content p',
                    '.story-content p',
                    '.content p',
                    'p'
                ]
                
                for selector in content_selectors:
                    paragraphs = soup.select(selector)
                    if paragraphs:
                        content = ' '.join([p.get_text(strip=True) for p in paragraphs[:5]])
                        if len(content) > 100:
                            return content[:1000]  # Limit content length
                
            return ""
            
        except Exception as e:
            logger.warning(f"Error getting article content: {str(e)}")
            return ""
    
    def _filter_by_date(self, articles: List[NewsArticle], days_back: int) -> List[NewsArticle]:
        """Filter articles by date range."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        return [a for a in articles if a.published_date >= cutoff_date]
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles."""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            title_key = article.title.lower().strip()
            if title_key not in seen_titles and len(title_key) > 10:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return sorted(unique_articles, key=lambda x: x.published_date, reverse=True)

class ComprehensiveNewsFetcher:
    """Main news fetcher that combines multiple sources."""
    
    def __init__(self, config: Dict[str, str]):
        """
        Initialize with API keys and configuration.
        
        Args:
            config: Dictionary containing API keys and settings
                   {'newsapi_key': 'xxx', 'alphavantage_key': 'xxx'}
        """
        self.config = config
        self.database = NewsDatabase()
        
        # Initialize fetchers
        self.fetchers = []
        
        if config.get('newsapi_key'):
            self.fetchers.append(NewsAPIFetcher(config['newsapi_key']))
        
        if config.get('alphavantage_key'):
            self.fetchers.append(AlphaVantageNewsFetcher(config['alphavantage_key']))
        
        # Always include web scraping as fallback
        self.fetchers.append(WebScrapingFetcher())
        
        logger.info(f"Initialized {len(self.fetchers)} news fetchers")
    
    def fetch_news(self, ticker: str, days_back: int = 7, 
                  max_articles: int = 100, use_cache: bool = True) -> List[NewsArticle]:
        """
        Fetch news from all available sources.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back
            max_articles: Maximum number of articles to return
            use_cache: Whether to use cached articles
            
        Returns:
            List of news articles
        """
        try:
            all_articles = []
            
            # Check cache first
            if use_cache:
                cached_articles = self.database.get_articles(ticker, days_back, max_articles)
                if cached_articles:
                    logger.info(f"Found {len(cached_articles)} cached articles for {ticker}")
                    all_articles.extend(cached_articles)
            
            # Fetch from external sources if we need more articles
            if len(all_articles) < max_articles:
                remaining_articles = max_articles - len(all_articles)
                articles_per_fetcher = remaining_articles // len(self.fetchers)
                
                with ThreadPoolExecutor(max_workers=len(self.fetchers)) as executor:
                    futures = []
                    
                    for fetcher in self.fetchers:
                        future = executor.submit(
                            fetcher.fetch_articles,
                            ticker,
                            days_back,
                            articles_per_fetcher
                        )
                        futures.append(future)
                    
                    for future in as_completed(futures, timeout=120):
                        try:
                            articles = future.result()
                            all_articles.extend(articles)
                            
                            # Cache new articles
                            for article in articles:
                                self.database.save_article(article)
                                
                        except Exception as e:
                            logger.warning(f"Fetcher failed: {str(e)}")
            
            # Deduplicate and sort
            unique_articles = self._deduplicate_articles(all_articles)
            
            logger.info(f"Fetched {len(unique_articles)} unique articles for {ticker}")
            return unique_articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Error in comprehensive news fetch: {str(e)}")
            return []
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles across all sources."""
        seen_urls = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            url_key = article.url
            title_key = article.title.lower().strip()
            
            if url_key not in seen_urls and title_key not in seen_titles:
                seen_urls.add(url_key)
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return sorted(unique_articles, key=lambda x: x.published_date, reverse=True)
    
    def get_news_summary(self, ticker: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Get a summary of news coverage for a ticker.
        
        Args:
            ticker: Stock ticker
            days_back: Days to analyze
            
        Returns:
            News summary statistics
        """
        try:
            articles = self.fetch_news(ticker, days_back)
            
            if not articles:
                return {
                    'ticker': ticker,
                    'total_articles': 0,
                    'sources': [],
                    'date_range': f"Last {days_back} days",
                    'coverage_trend': 'no_data'
                }
            
            # Calculate statistics
            sources = list(set([a.source for a in articles]))
            
            # Group by date
            daily_counts = {}
            for article in articles:
                date_key = article.published_date.date()
                daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
            
            # Calculate trend
            if len(daily_counts) > 1:
                dates = sorted(daily_counts.keys())
                recent_avg = sum(daily_counts[d] for d in dates[-3:]) / min(3, len(dates))
                older_avg = sum(daily_counts[d] for d in dates[:-3]) / max(1, len(dates) - 3)
                
                if recent_avg > older_avg * 1.2:
                    trend = 'increasing'
                elif recent_avg < older_avg * 0.8:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'insufficient_data'
            
            return {
                'ticker': ticker,
                'total_articles': len(articles),
                'sources': sources,
                'source_distribution': {s: sum(1 for a in articles if a.source == s) for s in sources},
                'date_range': f"Last {days_back} days",
                'daily_counts': {str(k): v for k, v in daily_counts.items()},
                'coverage_trend': trend,
                'avg_articles_per_day': len(articles) / days_back,
                'latest_article': articles[0].published_date.isoformat() if articles else None
            }
            
        except Exception as e:
            logger.error(f"Error getting news summary: {str(e)}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    # Configuration with API keys
    config = {
        'newsapi_key': os.getenv('NEWSAPI_KEY'),  # Get from environment
        'alphavantage_key': os.getenv('ALPHAVANTAGE_KEY')  # Get from environment
    }
    
    # Initialize fetcher
    fetcher = ComprehensiveNewsFetcher(config)
    
    # Fetch news for a ticker
    ticker = "RELIANCE.NS"
    articles = fetcher.fetch_news(ticker, days_back=7, max_articles=50)
    
    print(f"Fetched {len(articles)} articles for {ticker}")
    for article in articles[:3]:
        print(f"- {article.title} ({article.source})")
    
    # Get news summary
    summary = fetcher.get_news_summary(ticker)
    print(f"\nNews summary: {summary}")