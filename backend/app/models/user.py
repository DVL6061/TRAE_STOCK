from pydantic import BaseModel, Field, EmailStr, validator
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    PREMIUM = "premium"
    BASIC = "basic"
    TRIAL = "trial"

class SubscriptionStatus(str, Enum):
    """User subscription status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    PENDING = "pending"
    TRIAL = "trial"

class NotificationPreference(str, Enum):
    """Notification preference types."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"

class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    HINDI = "hi"

class Theme(str, Enum):
    """UI theme preferences."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"

class User(BaseModel):
    """User model."""
    id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    full_name: str = Field(..., min_length=1, max_length=100, description="Full name")
    
    # Authentication
    hashed_password: str = Field(..., description="Hashed password")
    is_active: bool = Field(default=True, description="Whether user account is active")
    is_verified: bool = Field(default=False, description="Whether email is verified")
    
    # Profile information
    phone_number: Optional[str] = Field(None, description="Phone number")
    date_of_birth: Optional[datetime] = Field(None, description="Date of birth")
    country: Optional[str] = Field(None, description="Country")
    timezone: str = Field(default="UTC", description="User timezone")
    
    # Subscription and role
    role: UserRole = Field(default=UserRole.TRIAL, description="User role")
    subscription_status: SubscriptionStatus = Field(default=SubscriptionStatus.TRIAL, description="Subscription status")
    subscription_expires_at: Optional[datetime] = Field(None, description="Subscription expiration")
    
    # Preferences
    language: Language = Field(default=Language.ENGLISH, description="Preferred language")
    theme: Theme = Field(default=Theme.LIGHT, description="UI theme preference")
    notification_preferences: List[NotificationPreference] = Field(default_factory=list, description="Notification preferences")
    
    # Timestamps
    created_at: datetime = Field(..., description="Account creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    # API access
    api_key: Optional[str] = Field(None, description="API key for programmatic access")
    api_calls_remaining: int = Field(default=1000, ge=0, description="Remaining API calls")
    api_calls_reset_at: datetime = Field(..., description="API calls reset timestamp")
    
class UserCreate(BaseModel):
    """User creation model."""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    full_name: str = Field(..., min_length=1, max_length=100, description="Full name")
    password: str = Field(..., min_length=8, max_length=100, description="Password")
    
    # Optional fields
    phone_number: Optional[str] = Field(None, description="Phone number")
    country: Optional[str] = Field(None, description="Country")
    timezone: str = Field(default="UTC", description="User timezone")
    language: Language = Field(default=Language.ENGLISH, description="Preferred language")
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v.lower()

class UserUpdate(BaseModel):
    """User update model."""
    full_name: Optional[str] = Field(None, min_length=1, max_length=100, description="Full name")
    phone_number: Optional[str] = Field(None, description="Phone number")
    country: Optional[str] = Field(None, description="Country")
    timezone: Optional[str] = Field(None, description="User timezone")
    language: Optional[Language] = Field(None, description="Preferred language")
    theme: Optional[Theme] = Field(None, description="UI theme preference")
    notification_preferences: Optional[List[NotificationPreference]] = Field(None, description="Notification preferences")

class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(default=False, description="Remember login session")

class UserResponse(BaseModel):
    """User response model (excludes sensitive data)."""
    id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., description="Username")
    full_name: str = Field(..., description="Full name")
    
    # Profile information
    phone_number: Optional[str] = Field(None, description="Phone number")
    country: Optional[str] = Field(None, description="Country")
    timezone: str = Field(..., description="User timezone")
    
    # Subscription and role
    role: UserRole = Field(..., description="User role")
    subscription_status: SubscriptionStatus = Field(..., description="Subscription status")
    subscription_expires_at: Optional[datetime] = Field(None, description="Subscription expiration")
    
    # Preferences
    language: Language = Field(..., description="Preferred language")
    theme: Theme = Field(..., description="UI theme preference")
    notification_preferences: List[NotificationPreference] = Field(..., description="Notification preferences")
    
    # Status
    is_active: bool = Field(..., description="Whether user account is active")
    is_verified: bool = Field(..., description="Whether email is verified")
    
    # Timestamps
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    # API access
    api_calls_remaining: int = Field(..., description="Remaining API calls")
    api_calls_reset_at: datetime = Field(..., description="API calls reset timestamp")

class UserWatchlist(BaseModel):
    """User watchlist model."""
    id: str = Field(..., description="Unique watchlist identifier")
    user_id: str = Field(..., description="User identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Watchlist name")
    description: Optional[str] = Field(None, max_length=500, description="Watchlist description")
    
    # Watchlist items
    tickers: List[str] = Field(..., description="List of stock tickers")
    is_default: bool = Field(default=False, description="Whether this is the default watchlist")
    
    # Metadata
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
class WatchlistCreate(BaseModel):
    """Watchlist creation model."""
    name: str = Field(..., min_length=1, max_length=100, description="Watchlist name")
    description: Optional[str] = Field(None, max_length=500, description="Watchlist description")
    tickers: List[str] = Field(default_factory=list, description="Initial list of stock tickers")
    is_default: bool = Field(default=False, description="Whether this is the default watchlist")

class WatchlistUpdate(BaseModel):
    """Watchlist update model."""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Watchlist name")
    description: Optional[str] = Field(None, max_length=500, description="Watchlist description")
    tickers: Optional[List[str]] = Field(None, description="List of stock tickers")
    is_default: Optional[bool] = Field(None, description="Whether this is the default watchlist")

class UserAlert(BaseModel):
    """User alert configuration."""
    id: str = Field(..., description="Unique alert identifier")
    user_id: str = Field(..., description="User identifier")
    ticker: str = Field(..., description="Stock ticker symbol")
    
    # Alert conditions
    alert_type: str = Field(..., description="Type of alert")
    condition: str = Field(..., description="Alert condition")
    threshold_value: float = Field(..., description="Threshold value")
    
    # Alert settings
    is_active: bool = Field(default=True, description="Whether alert is active")
    notification_methods: List[NotificationPreference] = Field(..., description="Notification methods")
    
    # Metadata
    created_at: datetime = Field(..., description="Creation timestamp")
    triggered_at: Optional[datetime] = Field(None, description="Last trigger timestamp")
    trigger_count: int = Field(default=0, ge=0, description="Number of times triggered")
    
class AlertCreate(BaseModel):
    """Alert creation model."""
    ticker: str = Field(..., description="Stock ticker symbol")
    alert_type: str = Field(..., description="Type of alert")
    condition: str = Field(..., description="Alert condition")
    threshold_value: float = Field(..., description="Threshold value")
    notification_methods: List[NotificationPreference] = Field(..., description="Notification methods")
    
    @validator('alert_type')
    def validate_alert_type(cls, v):
        allowed_types = ['price', 'volume', 'sentiment', 'prediction', 'news']
        if v.lower() not in allowed_types:
            raise ValueError(f'Alert type must be one of {allowed_types}')
        return v.lower()
    
    @validator('condition')
    def validate_condition(cls, v):
        allowed_conditions = ['above', 'below', 'equals', 'change_percent']
        if v.lower() not in allowed_conditions:
            raise ValueError(f'Condition must be one of {allowed_conditions}')
        return v.lower()

class UserSession(BaseModel):
    """User session model."""
    id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    
    # Session metadata
    created_at: datetime = Field(..., description="Session creation timestamp")
    expires_at: datetime = Field(..., description="Session expiration timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    
    # Device information
    device_info: Optional[Dict[str, Any]] = Field(None, description="Device information")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    
    # Status
    is_active: bool = Field(default=True, description="Whether session is active")
    
class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    
class UserStats(BaseModel):
    """User statistics model."""
    user_id: str = Field(..., description="User identifier")
    
    # Usage statistics
    total_predictions: int = Field(default=0, ge=0, description="Total predictions requested")
    total_api_calls: int = Field(default=0, ge=0, description="Total API calls made")
    watchlist_count: int = Field(default=0, ge=0, description="Number of watchlists")
    alert_count: int = Field(default=0, ge=0, description="Number of active alerts")
    
    # Activity metrics
    last_prediction_at: Optional[datetime] = Field(None, description="Last prediction timestamp")
    most_viewed_ticker: Optional[str] = Field(None, description="Most frequently viewed ticker")
    favorite_timeframe: Optional[str] = Field(None, description="Most used prediction timeframe")
    
    # Performance metrics
    prediction_accuracy: Optional[float] = Field(None, ge=0, le=1, description="User's prediction accuracy")
    successful_alerts: int = Field(default=0, ge=0, description="Number of successful alerts")
    
    # Timestamps
    stats_updated_at: datetime = Field(..., description="Statistics last updated")
    
class PasswordReset(BaseModel):
    """Password reset model."""
    email: EmailStr = Field(..., description="User email address")
    
class PasswordResetConfirm(BaseModel):
    """Password reset confirmation model."""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    
    @validator('new_password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class EmailVerification(BaseModel):
    """Email verification model."""
    token: str = Field(..., description="Verification token")
    
class UserPreferences(BaseModel):
    """User preferences model."""
    user_id: str = Field(..., description="User identifier")
    
    # Display preferences
    default_timeframe: str = Field(default="intraday", description="Default prediction timeframe")
    default_chart_type: str = Field(default="candlestick", description="Default chart type")
    show_technical_indicators: bool = Field(default=True, description="Show technical indicators")
    show_sentiment_analysis: bool = Field(default=True, description="Show sentiment analysis")
    
    # Notification preferences
    email_notifications: bool = Field(default=True, description="Enable email notifications")
    push_notifications: bool = Field(default=True, description="Enable push notifications")
    news_alerts: bool = Field(default=True, description="Enable news alerts")
    prediction_alerts: bool = Field(default=True, description="Enable prediction alerts")
    
    # Privacy preferences
    share_analytics: bool = Field(default=False, description="Share usage analytics")
    public_watchlists: bool = Field(default=False, description="Make watchlists public")
    
    # Advanced preferences
    auto_refresh_interval: int = Field(default=30, ge=5, le=300, description="Auto refresh interval in seconds")
    max_watchlist_items: int = Field(default=50, ge=1, le=200, description="Maximum watchlist items")
    
    # Timestamps
    updated_at: datetime = Field(..., description="Last update timestamp")