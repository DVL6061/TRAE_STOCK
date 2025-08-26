from typing import Dict, List
from datetime import datetime
import yfinance as yf
from smartapi import SmartConnect
from app.models.market import MarketData, PortfolioSummary
from app.config import ANGEL_ONE_API_KEY, ANGEL_ONE_CLIENT_ID, ANGEL_ONE_PASSWORD

class MarketService:
    def __init__(self):
        self.smart_api = SmartConnect(api_key=ANGEL_ONE_API_KEY)
        self.session_data = self.smart_api.generateSession(
            ANGEL_ONE_CLIENT_ID,
            ANGEL_ONE_PASSWORD
        )
        self.refresh_token = self.session_data['data']['refreshToken']
        
        # Initialize market symbols
        self.symbols = {
            'NIFTY': '^NSEI',
            'SENSEX': '^BSESN',
            'BANKNIFTY': '^NSEBANK'
        }
        
        # Cache for market data
        self._market_data_cache = {}
        self._portfolio_cache = None
    
    async def get_real_time_data(self) -> MarketData:
        """Fetch real-time market data from Angel One Smart API"""
        try:
            data = {}
            for index_name, symbol in self.symbols.items():
                # Get real-time LTP (Last Traded Price)
                ltp_data = self.smart_api.ltpData(
                    exchange='NSE',
                    tradingsymbol=symbol,
                    symboltoken=self._get_token(symbol)
                )
                
                # Get OHLC data from yfinance for additional info
                yf_data = yf.download(symbol, period='1d', interval='1m')
                latest = yf_data.iloc[-1]
                
                data[index_name.lower()] = {
                    'current': ltp_data['data']['ltp'],
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'change': ((ltp_data['data']['ltp'] - latest['Open']) / latest['Open']) * 100
                }
            
            self._market_data_cache = data
            return MarketData(**data)
            
        except Exception as e:
            # Return cached data if API call fails
            if self._market_data_cache:
                return MarketData(**self._market_data_cache)
            raise e
    
    async def get_current_data(self) -> MarketData:
        """Get current market data from cache or fetch new"""
        if not self._market_data_cache:
            return await self.get_real_time_data()
        return MarketData(**self._market_data_cache)
    
    async def get_portfolio_summary(self) -> PortfolioSummary:
        """Get portfolio summary from Angel One"""
        try:
            # Get holdings data
            holdings = self.smart_api.holding()
            
            # Calculate portfolio metrics
            total_investment = sum(float(h['averageprice']) * float(h['quantity']) for h in holdings['data'])
            current_value = sum(float(h['ltp']) * float(h['quantity']) for h in holdings['data'])
            total_pnl = current_value - total_investment
            day_pnl = sum(float(h['daychange']) * float(h['quantity']) for h in holdings['data'])
            
            portfolio_data = {
                'total_investment': total_investment,
                'current_value': current_value,
                'total_pnl': total_pnl,
                'total_pnl_percentage': (total_pnl / total_investment) * 100 if total_investment > 0 else 0,
                'day_pnl': day_pnl,
                'day_pnl_percentage': (day_pnl / total_investment) * 100 if total_investment > 0 else 0,
                'holdings': holdings['data']
            }
            
            self._portfolio_cache = portfolio_data
            return PortfolioSummary(**portfolio_data)
            
        except Exception as e:
            # Return cached data if API call fails
            if self._portfolio_cache:
                return PortfolioSummary(**self._portfolio_cache)
            raise e
    
    def _get_token(self, symbol: str) -> str:
        """Get trading token for a symbol"""
        # Implement token mapping logic here
        # This would typically involve maintaining a mapping of symbols to their tokens
        # or making an API call to get the token
        pass