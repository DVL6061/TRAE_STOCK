import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from transformers import pipeline
from bs4 import BeautifulSoup
from app.config import NEWS_SOURCES

class NewsService:
    def __init__(self):
        # Initialize FinGPT sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
        
        # News sources configuration
        self.sources = NEWS_SOURCES
        
        # Cache for news and sentiment
        self._news_cache = []
        self._sentiment_cache = None
        self._last_update = None
        
        # Update interval (15 minutes)
        self._update_interval = timedelta(minutes=15)
    
    async def get_sentiment_score(self) -> float:
        """Get aggregated sentiment score from financial news"""
        try:
            # Check if cache is valid
            if (self._sentiment_cache is not None and self._last_update and 
                datetime.now() - self._last_update < self._update_interval):
                return self._sentiment_cache
            
            # Fetch and analyze news
            news_articles = await self._fetch_news()
            sentiment_scores = await self._analyze_sentiment(news_articles)
            
            # Calculate weighted average sentiment
            weighted_sentiment = self._calculate_weighted_sentiment(sentiment_scores)
            
            # Update cache
            self._sentiment_cache = weighted_sentiment
            self._last_update = datetime.now()
            self._news_cache = news_articles
            
            return weighted_sentiment
            
        except Exception as e:
            # Return cached sentiment if available
            if self._sentiment_cache is not None:
                return self._sentiment_cache
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