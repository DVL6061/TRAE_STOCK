from pydantic import BaseModel, Field, validator, HttpUrl
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum

class NewsSource(str, Enum):
    """Supported news sources."""
    CNBC = "cnbc"
    MONEYCONTROL = "moneycontrol"
    MINT = "mint"
    ECONOMIC_TIMES = "economic_times"
    BUSINESS_STANDARD = "business_standard"
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    YAHOO_FINANCE = "yahoo_finance"

class NewsCategory(str, Enum):
    """News categories."""
    EARNINGS = "earnings"
    DIVIDENDS = "dividends"
    POLITICS = "politics"
    ECONOMY = "economy"
    MARKET = "market"
    COMPANY = "company"
    SECTOR = "sector"
    GLOBAL = "global"
    REGULATORY = "regulatory"
    MERGER_ACQUISITION = "merger_acquisition"

class SentimentLabel(str, Enum):
    """Sentiment analysis labels."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class ImportanceLevel(str, Enum):
    """News importance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NewsArticle(BaseModel):
    """Individual news article model."""
    id: str = Field(..., description="Unique article identifier")
    title: str = Field(..., min_length=1, max_length=500, description="Article title")
    content: str = Field(..., min_length=1, description="Article content")
    summary: Optional[str] = Field(None, max_length=1000, description="Article summary")
    url: HttpUrl = Field(..., description="Article URL")
    source: NewsSource = Field(..., description="News source")
    author: Optional[str] = Field(None, description="Article author")
    published_at: datetime = Field(..., description="Publication timestamp")
    scraped_at: datetime = Field(..., description="Scraping timestamp")
    
    # Content analysis
    category: Optional[NewsCategory] = Field(None, description="News category")
    tags: List[str] = Field(default_factory=list, description="Article tags")
    mentioned_tickers: List[str] = Field(default_factory=list, description="Stock tickers mentioned")
    mentioned_companies: List[str] = Field(default_factory=list, description="Companies mentioned")
    
    # Metadata
    language: str = Field(default="en", description="Article language")
    word_count: Optional[int] = Field(None, ge=0, description="Article word count")
    reading_time: Optional[int] = Field(None, ge=0, description="Estimated reading time in minutes")
    
class SentimentScore(BaseModel):
    """Sentiment analysis result for an article."""
    article_id: str = Field(..., description="Associated article ID")
    label: SentimentLabel = Field(..., description="Sentiment label")
    score: float = Field(..., ge=-1, le=1, description="Sentiment score (-1 to 1)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in sentiment prediction")
    
    # Detailed scores
    positive_score: float = Field(..., ge=0, le=1, description="Positive sentiment probability")
    negative_score: float = Field(..., ge=0, le=1, description="Negative sentiment probability")
    neutral_score: float = Field(..., ge=0, le=1, description="Neutral sentiment probability")
    
    # Analysis metadata
    model_version: str = Field(..., description="Sentiment model version")
    processed_at: datetime = Field(..., description="Processing timestamp")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    
class NewsImportance(BaseModel):
    """News importance assessment."""
    article_id: str = Field(..., description="Associated article ID")
    importance_level: ImportanceLevel = Field(..., description="Importance level")
    importance_score: float = Field(..., ge=0, le=1, description="Importance score (0 to 1)")
    
    # Importance factors
    source_credibility: float = Field(..., ge=0, le=1, description="Source credibility score")
    market_relevance: float = Field(..., ge=0, le=1, description="Market relevance score")
    timeliness: float = Field(..., ge=0, le=1, description="Timeliness score")
    company_impact: float = Field(..., ge=0, le=1, description="Company impact score")
    sector_impact: float = Field(..., ge=0, le=1, description="Sector impact score")
    
    # Metadata
    calculated_at: datetime = Field(..., description="Calculation timestamp")
    
class NewsWithSentiment(BaseModel):
    """News article with sentiment analysis."""
    article: NewsArticle
    sentiment: SentimentScore
    importance: Optional[NewsImportance] = Field(None, description="Importance assessment")
    
class NewsSummary(BaseModel):
    """Summary of news for a specific ticker/timeframe."""
    ticker: str = Field(..., description="Stock ticker symbol")
    timeframe: str = Field(..., description="Time period for summary")
    total_articles: int = Field(..., ge=0, description="Total number of articles")
    
    # Sentiment distribution
    positive_articles: int = Field(..., ge=0, description="Number of positive articles")
    negative_articles: int = Field(..., ge=0, description="Number of negative articles")
    neutral_articles: int = Field(..., ge=0, description="Number of neutral articles")
    
    # Aggregate sentiment
    overall_sentiment: SentimentLabel = Field(..., description="Overall sentiment")
    average_sentiment_score: float = Field(..., ge=-1, le=1, description="Average sentiment score")
    weighted_sentiment_score: float = Field(..., ge=-1, le=1, description="Importance-weighted sentiment")
    sentiment_volatility: float = Field(..., ge=0, description="Sentiment volatility measure")
    
    # Source distribution
    source_distribution: Dict[str, int] = Field(..., description="Articles per source")
    category_distribution: Dict[str, int] = Field(..., description="Articles per category")
    
    # Time-based metrics
    summary_generated_at: datetime = Field(..., description="Summary generation timestamp")
    earliest_article: datetime = Field(..., description="Earliest article timestamp")
    latest_article: datetime = Field(..., description="Latest article timestamp")
    
class NewsAlert(BaseModel):
    """News-based alert."""
    id: str = Field(..., description="Unique alert ID")
    ticker: str = Field(..., description="Related stock ticker")
    alert_type: str = Field(..., description="Type of alert")
    severity: ImportanceLevel = Field(..., description="Alert severity")
    
    # Alert content
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    related_articles: List[str] = Field(..., description="Related article IDs")
    
    # Trigger conditions
    trigger_condition: str = Field(..., description="Condition that triggered alert")
    threshold_value: Optional[float] = Field(None, description="Threshold value if applicable")
    actual_value: Optional[float] = Field(None, description="Actual value that triggered alert")
    
    # Metadata
    created_at: datetime = Field(..., description="Alert creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Alert expiration timestamp")
    is_active: bool = Field(default=True, description="Whether alert is active")
    acknowledged: bool = Field(default=False, description="Whether alert was acknowledged")
    
class NewsSearchRequest(BaseModel):
    """Request for news search."""
    query: Optional[str] = Field(None, description="Search query")
    tickers: Optional[List[str]] = Field(None, description="Filter by tickers")
    sources: Optional[List[NewsSource]] = Field(None, description="Filter by sources")
    categories: Optional[List[NewsCategory]] = Field(None, description="Filter by categories")
    
    # Time filters
    start_date: Optional[datetime] = Field(None, description="Start date for search")
    end_date: Optional[datetime] = Field(None, description="End date for search")
    
    # Result options
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Results offset for pagination")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_importance: bool = Field(default=False, description="Include importance scores")
    
class NewsSearchResponse(BaseModel):
    """Response for news search."""
    total_count: int = Field(..., ge=0, description="Total matching articles")
    returned_count: int = Field(..., ge=0, description="Number of articles returned")
    offset: int = Field(..., ge=0, description="Current offset")
    
    articles: List[NewsWithSentiment] = Field(..., description="Matching articles")
    
    # Aggregated data
    sentiment_summary: Optional[Dict[str, Any]] = Field(None, description="Sentiment summary")
    source_summary: Optional[Dict[str, int]] = Field(None, description="Source distribution")
    category_summary: Optional[Dict[str, int]] = Field(None, description="Category distribution")
    
    # Metadata
    search_time: float = Field(..., ge=0, description="Search execution time")
    generated_at: datetime = Field(..., description="Response generation timestamp")
    
class NewsStreamUpdate(BaseModel):
    """Real-time news stream update."""
    update_type: str = Field(..., description="Type of update")
    timestamp: datetime = Field(..., description="Update timestamp")
    
    # Update content
    new_articles: Optional[List[NewsWithSentiment]] = Field(None, description="New articles")
    updated_sentiment: Optional[Dict[str, NewsSummary]] = Field(None, description="Updated sentiment summaries")
    alerts: Optional[List[NewsAlert]] = Field(None, description="New alerts")
    
    # Metadata
    total_new_articles: int = Field(default=0, ge=0, description="Total new articles in update")
    affected_tickers: List[str] = Field(default_factory=list, description="Tickers affected by update")
    
class ScrapingStatus(BaseModel):
    """Status of news scraping operations."""
    source: NewsSource = Field(..., description="News source")
    last_scrape: datetime = Field(..., description="Last successful scrape")
    next_scrape: datetime = Field(..., description="Next scheduled scrape")
    
    # Status metrics
    articles_scraped: int = Field(..., ge=0, description="Articles scraped in last run")
    success_rate: float = Field(..., ge=0, le=1, description="Scraping success rate")
    average_processing_time: float = Field(..., ge=0, description="Average processing time per article")
    
    # Error tracking
    last_error: Optional[str] = Field(None, description="Last error message")
    error_count: int = Field(default=0, ge=0, description="Error count in current period")
    
    # Configuration
    is_active: bool = Field(default=True, description="Whether scraping is active")
    scrape_interval: int = Field(..., ge=60, description="Scraping interval in seconds")
    
class NewsConfiguration(BaseModel):
    """Configuration for news collection and processing."""
    # Source configuration
    enabled_sources: List[NewsSource] = Field(..., description="Enabled news sources")
    source_weights: Dict[str, float] = Field(..., description="Weight for each source")
    
    # Processing configuration
    sentiment_model: str = Field(..., description="Sentiment analysis model")
    importance_threshold: float = Field(..., ge=0, le=1, description="Minimum importance threshold")
    
    # Filtering configuration
    min_word_count: int = Field(default=50, ge=0, description="Minimum article word count")
    max_article_age: int = Field(default=7, ge=1, description="Maximum article age in days")
    
    # Alert configuration
    alert_thresholds: Dict[str, float] = Field(..., description="Alert threshold values")
    
    # Update timestamps
    created_at: datetime = Field(..., description="Configuration creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    @validator('source_weights')
    def validate_source_weights(cls, v):
        """Ensure all weights are between 0 and 1."""
        for source, weight in v.items():
            if not 0 <= weight <= 1:
                raise ValueError(f'Weight for {source} must be between 0 and 1')
        return v