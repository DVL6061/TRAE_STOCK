from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
import yfinance as yf
from smartapi import SmartConnect
import pyotp
import json
from app.models.market import (
    MarketDataPoint, TechnicalIndicators, MarketSummary, 
    StockData, PortfolioSummary, MarketStatus, MarketOverview
)
from app.config import ANGEL_ONE_API_KEY, ANGEL_ONE_CLIENT_ID, ANGEL_ONE_PASSWORD

logger = logging.getLogger(__name__)

class MarketService:
    def __init__(self):
        self.smart_api = None
        self.session_data = None
        self.refresh_token = None
        self.access_token = None
        self.session_expires_at = None
        
        # Initialize market symbols with proper Angel One symbols
        self.symbols = {
            'NIFTY': {'symbol': 'NIFTY 50', 'token': '99926000', 'exchange': 'NSE'},
            'SENSEX': {'symbol': 'SENSEX', 'token': '99919000', 'exchange': 'BSE'},
            'BANKNIFTY': {'symbol': 'NIFTY BANK', 'token': '99926009', 'exchange': 'NSE'},
            'FINNIFTY': {'symbol': 'NIFTY FIN SERVICE', 'token': '99926037', 'exchange': 'NSE'}
        }
        
        # Cache for market data with TTL
        self._market_data_cache = {}
        self._portfolio_cache = None
        self._cache_ttl = timedelta(seconds=30)  # 30 seconds cache
        self._last_cache_update = None
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Angel One API connection with proper authentication."""
        try:
            if not all([ANGEL_ONE_API_KEY, ANGEL_ONE_CLIENT_ID, ANGEL_ONE_PASSWORD]):
                logger.warning("Angel One API credentials not found. Using mock data.")
                return
                
            self.smart_api = SmartConnect(api_key=ANGEL_ONE_API_KEY)
            
            # Generate TOTP for 2FA (if required)
            totp = None
            if hasattr(self, 'totp_secret') and self.totp_secret:
                totp = pyotp.TOTP(self.totp_secret).now()
            
            # Generate session
            self.session_data = self.smart_api.generateSession(
                clientCode=ANGEL_ONE_CLIENT_ID,
                password=ANGEL_ONE_PASSWORD,
                totp=totp
            )
            
            if self.session_data and self.session_data.get('status'):
                self.refresh_token = self.session_data['data']['refreshToken']
                self.access_token = self.session_data['data']['jwtToken']
                self.session_expires_at = datetime.now() + timedelta(hours=8)  # 8 hour session
                logger.info("Angel One API connection established successfully")
            else:
                logger.error(f"Failed to establish Angel One connection: {self.session_data}")
                
        except Exception as e:
            logger.error(f"Error initializing Angel One connection: {str(e)}")
            self.smart_api = None
    
    def _refresh_session(self):
        """Refresh the Angel One API session if expired."""
        try:
            if not self.smart_api or not self.refresh_token:
                self._initialize_connection()
                return
                
            if self.session_expires_at and datetime.now() >= self.session_expires_at:
                refresh_data = self.smart_api.renewAccessToken(
                    refreshToken=self.refresh_token
                )
                
                if refresh_data and refresh_data.get('status'):
                    self.access_token = refresh_data['data']['jwtToken']
                    self.session_expires_at = datetime.now() + timedelta(hours=8)
                    logger.info("Angel One session refreshed successfully")
                else:
                    logger.error("Failed to refresh Angel One session")
                    self._initialize_connection()
                    
        except Exception as e:
            logger.error(f"Error refreshing Angel One session: {str(e)}")
            self._initialize_connection()
    
    async def get_real_time_data(self, ticker: str = None) -> MarketOverview:
        """Fetch real-time market data from Angel One Smart API"""
        try:
            # Check session validity
            self._refresh_session()
            
            market_data = {}
            
            if self.smart_api:
                # Get data from Angel One API
                for index_name, symbol_info in self.symbols.items():
                    try:
                        ltp_data = self.smart_api.ltpData(
                            exchange=symbol_info['exchange'],
                            tradingsymbol=symbol_info['symbol'],
                            symboltoken=symbol_info['token']
                        )
                        
                        if ltp_data and ltp_data.get('status'):
                            ltp = float(ltp_data['data']['ltp'])
                            
                            # Get additional OHLC data
                            ohlc_data = self.smart_api.getCandleData(
                                exchange=symbol_info['exchange'],
                                symboltoken=symbol_info['token'],
                                interval='ONE_DAY',
                                fromdate='2024-01-01 09:15',
                                todate=datetime.now().strftime('%Y-%m-%d %H:%M')
                            )
                            
                            if ohlc_data and ohlc_data.get('status') and ohlc_data['data']:
                                latest_candle = ohlc_data['data'][-1]
                                open_price = float(latest_candle[1])
                                high_price = float(latest_candle[2])
                                low_price = float(latest_candle[3])
                                
                                change = ltp - open_price
                                change_percent = (change / open_price) * 100 if open_price > 0 else 0
                                
                                market_data[index_name.lower()] = MarketSummary(
                                    current_price=ltp,
                                    open_price=open_price,
                                    high_price=high_price,
                                    low_price=low_price,
                                    change=change,
                                    change_percent=change_percent,
                                    volume=int(latest_candle[5]) if len(latest_candle) > 5 else 0
                                )
                            
                    except Exception as e:
                        logger.error(f"Error fetching data for {index_name}: {str(e)}")
                        continue
            
            # Fallback to yfinance if Angel One fails or for additional data
            if not market_data:
                market_data = await self._get_yfinance_data()
            
            # Create market overview
            market_overview = MarketOverview(
                nifty=market_data.get('nifty'),
                sensex=market_data.get('sensex'),
                banknifty=market_data.get('banknifty'),
                finnifty=market_data.get('finnifty'),
                market_status=await self._get_market_status(),
                last_updated=datetime.now()
            )
            
            # Cache the data
            self._market_data_cache = market_overview
            self._last_cache_update = datetime.now()
            
            return market_overview
            
        except Exception as e:
            logger.error(f"Error fetching real-time market data: {str(e)}")
            # Return cached data if available
            if self._market_data_cache and self._is_cache_valid():
                return self._market_data_cache
            
            # Return fallback data
            return await self._get_fallback_market_data()
    
    async def _get_yfinance_data(self) -> Dict:
        """Get market data from yfinance as fallback."""
        market_data = {}
        yf_symbols = {
            'nifty': '^NSEI',
            'sensex': '^BSESN',
            'banknifty': '^NSEBANK'
        }
        
        for index_name, yf_symbol in yf_symbols.items():
            try:
                ticker = yf.Ticker(yf_symbol)
                info = ticker.info
                hist = ticker.history(period='1d', interval='1m')
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    current_price = float(latest['Close'])
                    open_price = float(hist.iloc[0]['Open'])
                    
                    market_data[index_name] = MarketSummary(
                        current_price=current_price,
                        open_price=open_price,
                        high_price=float(hist['High'].max()),
                        low_price=float(hist['Low'].min()),
                        change=current_price - open_price,
                        change_percent=((current_price - open_price) / open_price) * 100,
                        volume=int(hist['Volume'].sum())
                    )
                    
            except Exception as e:
                logger.error(f"Error fetching yfinance data for {index_name}: {str(e)}")
                continue
                
        return market_data
    
    async def _get_market_status(self) -> MarketStatus:
        """Get current market status."""
        now = datetime.now()
        
        # Indian market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
        is_market_hours = market_open <= now <= market_close
        
        if is_weekend:
            status = "CLOSED"
            message = "Market is closed (Weekend)"
        elif is_market_hours:
            status = "OPEN"
            message = "Market is open"
        elif now < market_open:
            status = "PRE_MARKET"
            message = "Pre-market session"
        else:
            status = "CLOSED"
            message = "Market is closed"
        
        return MarketStatus(
            status=status,
            message=message,
            next_open=market_open if now < market_open else market_open + timedelta(days=1),
            next_close=market_close if is_market_hours else market_close + timedelta(days=1)
        )
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if not self._last_cache_update:
            return False
        return datetime.now() - self._last_cache_update < self._cache_ttl
    
    async def _get_fallback_market_data(self) -> MarketOverview:
        """Get fallback market data when all sources fail."""
        return MarketOverview(
            nifty=MarketSummary(
                current_price=22000.0,
                open_price=22000.0,
                high_price=22100.0,
                low_price=21900.0,
                change=0.0,
                change_percent=0.0,
                volume=0
            ),
            sensex=MarketSummary(
                current_price=72000.0,
                open_price=72000.0,
                high_price=72200.0,
                low_price=71800.0,
                change=0.0,
                change_percent=0.0,
                volume=0
            ),
            banknifty=MarketSummary(
                current_price=48000.0,
                open_price=48000.0,
                high_price=48200.0,
                low_price=47800.0,
                change=0.0,
                change_percent=0.0,
                volume=0
            ),
            market_status=await self._get_market_status(),
            last_updated=datetime.now()
        )
    
    async def get_current_data(self) -> MarketOverview:
        """Get current market data from cache or fetch new"""
        if not self._market_data_cache or not self._is_cache_valid():
            return await self.get_real_time_data()
        return self._market_data_cache
    
    async def get_stock_data(self, symbol: str, period: str = '1d', interval: str = '1m') -> StockData:
        """Get detailed stock data with technical indicators."""
        try:
            # First try Angel One API
            if self.smart_api:
                self._refresh_session()
                
                # Get symbol token (this would need proper symbol mapping)
                symbol_token = self._get_symbol_token(symbol)
                
                if symbol_token:
                    # Get OHLCV data
                    candle_data = self.smart_api.getCandleData(
                        exchange='NSE',
                        symboltoken=symbol_token,
                        interval=self._convert_interval(interval),
                        fromdate=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M'),
                        todate=datetime.now().strftime('%Y-%m-%d %H:%M')
                    )
                    
                    if candle_data and candle_data.get('status') and candle_data['data']:
                        # Convert to MarketDataPoint objects
                        data_points = []
                        for candle in candle_data['data']:
                            data_points.append(MarketDataPoint(
                                timestamp=datetime.strptime(candle[0], '%Y-%m-%dT%H:%M:%S%z'),
                                open=float(candle[1]),
                                high=float(candle[2]),
                                low=float(candle[3]),
                                close=float(candle[4]),
                                volume=int(candle[5]) if len(candle) > 5 else 0
                            ))
                        
                        # Calculate technical indicators
                        technical_indicators = self._calculate_technical_indicators(data_points)
                        
                        return StockData(
                            symbol=symbol,
                            data_points=data_points,
                            technical_indicators=technical_indicators,
                            last_updated=datetime.now()
                        )
            
            # Fallback to yfinance
            return await self._get_yfinance_stock_data(symbol, period, interval)
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            return await self._get_yfinance_stock_data(symbol, period, interval)
    
    async def _get_yfinance_stock_data(self, symbol: str, period: str, interval: str) -> StockData:
        """Get stock data from yfinance."""
        try:
            # Convert symbol to yfinance format if needed
            yf_symbol = self._convert_to_yf_symbol(symbol)
            
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Convert to MarketDataPoint objects
            data_points = []
            for timestamp, row in hist.iterrows():
                data_points.append(MarketDataPoint(
                    timestamp=timestamp,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume'])
                ))
            
            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(data_points)
            
            return StockData(
                symbol=symbol,
                data_points=data_points,
                technical_indicators=technical_indicators,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {str(e)}")
            raise
    
    def _calculate_technical_indicators(self, data_points: List[MarketDataPoint]) -> TechnicalIndicators:
        """Calculate technical indicators from price data."""
        if len(data_points) < 20:
            # Not enough data for calculations
            return TechnicalIndicators()
        
        closes = [dp.close for dp in data_points]
        highs = [dp.high for dp in data_points]
        lows = [dp.low for dp in data_points]
        volumes = [dp.volume for dp in data_points]
        
        # Simple Moving Averages
        sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
        
        # Exponential Moving Averages
        ema_12 = self._calculate_ema(closes, 12)
        ema_26 = self._calculate_ema(closes, 26)
        
        # RSI
        rsi = self._calculate_rsi(closes)
        
        # MACD
        macd_line = ema_12 - ema_26 if ema_12 and ema_26 else None
        macd_signal = self._calculate_ema([macd_line] * 9, 9) if macd_line else None
        macd_histogram = macd_line - macd_signal if macd_line and macd_signal else None
        
        # Bollinger Bands
        bb_middle = sma_20
        if bb_middle and len(closes) >= 20:
            std_dev = (sum([(x - bb_middle) ** 2 for x in closes[-20:]]) / 20) ** 0.5
            bb_upper = bb_middle + (2 * std_dev)
            bb_lower = bb_middle - (2 * std_dev)
        else:
            bb_upper = bb_lower = None
        
        return TechnicalIndicators(
            sma_20=sma_20,
            sma_50=sma_50,
            ema_12=ema_12,
            ema_26=ema_26,
            rsi=rsi,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower
        )
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return None
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _get_symbol_token(self, symbol: str) -> Optional[str]:
        """Get Angel One token for symbol."""
        # This would need a proper symbol-to-token mapping database
        # For now, return None to fallback to yfinance
        return None
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval to Angel One format."""
        interval_map = {
            '1m': 'ONE_MINUTE',
            '5m': 'FIVE_MINUTE',
            '15m': 'FIFTEEN_MINUTE',
            '1h': 'ONE_HOUR',
            '1d': 'ONE_DAY'
        }
        return interval_map.get(interval, 'ONE_MINUTE')
    
    def _convert_to_yf_symbol(self, symbol: str) -> str:
        """Convert symbol to yfinance format."""
        # Add .NS suffix for NSE stocks if not present
        if '.' not in symbol and symbol not in ['^NSEI', '^BSESN', '^NSEBANK']:
            return f"{symbol}.NS"
        return symbol
    
    async def get_portfolio_summary(self) -> PortfolioSummary:
        """Get portfolio summary from Angel One API"""
        if self._portfolio_cache:
            return self._portfolio_cache
            
        try:
            if self.smart_api:
                self._refresh_session()
                
                # Get holdings from Angel One API
                holdings = self.smart_api.holding()
                
                if holdings and holdings.get('status') and holdings.get('data'):
                    # Calculate portfolio metrics
                    total_value = 0
                    total_invested = 0
                    
                    for holding in holdings['data']:
                        current_value = float(holding['ltp']) * int(holding['quantity'])
                        invested_value = float(holding['averageprice']) * int(holding['quantity'])
                        
                        total_value += current_value
                        total_invested += invested_value
                    
                    day_change = total_value - total_invested
                    day_change_percent = (day_change / total_invested) * 100 if total_invested > 0 else 0
                    
                    portfolio = PortfolioSummary(
                        total_value=total_value,
                        total_invested=total_invested,
                        day_change=day_change,
                        day_change_percent=day_change_percent,
                        holdings_count=len(holdings['data'])
                    )
                    
                    self._portfolio_cache = portfolio
                    return portfolio
            
            # Return mock data if API call fails or no API
            return PortfolioSummary(
                total_value=100000.0,
                total_invested=95000.0,
                day_change=5000.0,
                day_change_percent=5.26,
                holdings_count=5
            )
            
        except Exception as e:
            logger.error(f"Error fetching portfolio summary: {str(e)}")
            # Return mock data if API call fails
            return PortfolioSummary(
                total_value=100000.0,
                total_invested=95000.0,
                day_change=5000.0,
                day_change_percent=5.26,
                holdings_count=5
            )