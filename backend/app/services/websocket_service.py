import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor

# Import our services and clients
from ..data.angel_one_client import AngelOneClient
from .prediction_service import PredictionService
from .market_service import MarketService
from ..config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PRICE_UPDATE = "price_update"
    PREDICTION_UPDATE = "prediction_update"
    TECHNICAL_UPDATE = "technical_update"
    NEWS_UPDATE = "news_update"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class ClientSubscription:
    """Client subscription information."""
    client_id: str
    websocket: Any  # FastAPI WebSocket or websockets.WebSocketServerProtocol
    subscribed_tickers: Set[str] = field(default_factory=set)
    subscription_types: Set[str] = field(default_factory=set)
    prediction_interval: int = 30
    last_heartbeat: datetime = field(default_factory=datetime.now)
    last_price_update: Dict[str, datetime] = field(default_factory=dict)
    last_prediction_update: Dict[str, datetime] = field(default_factory=dict)

class WebSocketDataStreamer:
    """
    WebSocket data streaming service for real-time stock data.
    
    Handles:
    - Real-time price updates from Angel One API
    - ML prediction updates
    - Technical indicator calculations
    - News sentiment updates
    - Client subscription management
    """
    
    def __init__(self):
        """Initialize WebSocket data streamer."""
        self.settings = get_settings()
        
        # Services
        self.angel_client = AngelOneClient()
        self.prediction_service = PredictionService()
        self.market_service = MarketService()
        
        # Client management
        self.clients: Dict[str, ClientSubscription] = {}
        self.active_tickers: Set[str] = set()
        
        # Data caches
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.prediction_cache: Dict[str, Dict[str, Any]] = {}
        self.technical_cache: Dict[str, Dict[str, Any]] = {}
        self.news_cache: Dict[str, Dict[str, Any]] = {}
        
        # Threading
        self.streaming_active = False
        self.price_thread: Optional[threading.Thread] = None
        self.prediction_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Update intervals (seconds)
        self.price_update_interval = 1  # 1 second for real-time prices
        self.prediction_update_interval = 30  # 30 seconds for predictions
        self.technical_update_interval = 5  # 5 seconds for technical indicators
        self.heartbeat_interval = 30  # 30 seconds for heartbeat
        
        logger.info("WebSocket data streamer initialized")
    
    def start_streaming(self):
        """Start background streaming threads."""
        if self.streaming_active:
            logger.warning("Streaming already active")
            return
        
        self.streaming_active = True
        
        # Start background threads
        self.price_thread = threading.Thread(target=self._price_streaming_loop, daemon=True)
        self.prediction_thread = threading.Thread(target=self._prediction_streaming_loop, daemon=True)
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        
        self.price_thread.start()
        self.prediction_thread.start()
        self.heartbeat_thread.start()
        
        logger.info("WebSocket streaming threads started")
    
    def stop_streaming(self):
        """Stop background streaming threads."""
        self.streaming_active = False
        
        # Wait for threads to finish
        if self.price_thread and self.price_thread.is_alive():
            self.price_thread.join(timeout=5)
        
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=5)
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("WebSocket streaming stopped")
    
    def _price_streaming_loop(self):
        """Background thread for price updates."""
        logger.info("Price streaming loop started")
        
        while self.streaming_active:
            try:
                if self.active_tickers:
                    # Fetch prices for all active tickers
                    for ticker in list(self.active_tickers):
                        try:
                            price_data = self._fetch_current_price(ticker)
                            if price_data:
                                self.price_cache[ticker] = price_data
                                
                                # Calculate technical indicators
                                technical_data = self._calculate_technical_indicators(ticker, price_data)
                                if technical_data:
                                    self.technical_cache[ticker] = technical_data
                                
                                # Broadcast to subscribed clients
                                asyncio.create_task(self._broadcast_price_update(ticker, price_data))
                                
                                if technical_data:
                                    asyncio.create_task(self._broadcast_technical_update(ticker, technical_data))
                        
                        except Exception as e:
                            logger.error(f"Error fetching price for {ticker}: {str(e)}")
                
                time.sleep(self.price_update_interval)
                
            except Exception as e:
                logger.error(f"Error in price streaming loop: {str(e)}")
                time.sleep(5)
        
        logger.info("Price streaming loop stopped")
    
    def _prediction_streaming_loop(self):
        """Background thread for prediction updates."""
        logger.info("Prediction streaming loop started")
        
        while self.streaming_active:
            try:
                if self.active_tickers:
                    # Generate predictions for all active tickers
                    for ticker in list(self.active_tickers):
                        try:
                            # Check if clients want prediction updates for this ticker
                            clients_want_predictions = any(
                                'prediction' in client.subscription_types and ticker in client.subscribed_tickers
                                for client in self.clients.values()
                            )
                            
                            if clients_want_predictions:
                                prediction_data = self._generate_prediction(ticker)
                                if prediction_data:
                                    self.prediction_cache[ticker] = prediction_data
                                    
                                    # Broadcast to subscribed clients
                                    asyncio.create_task(self._broadcast_prediction_update(ticker, prediction_data))
                        
                        except Exception as e:
                            logger.error(f"Error generating prediction for {ticker}: {str(e)}")
                
                time.sleep(self.prediction_update_interval)
                
            except Exception as e:
                logger.error(f"Error in prediction streaming loop: {str(e)}")
                time.sleep(10)
        
        logger.info("Prediction streaming loop stopped")
    
    def _heartbeat_loop(self):
        """Background thread for client heartbeat monitoring."""
        logger.info("Heartbeat loop started")
        
        while self.streaming_active:
            try:
                current_time = datetime.now()
                disconnected_clients = []
                
                # Check for stale clients
                for client_id, client in self.clients.items():
                    time_since_heartbeat = current_time - client.last_heartbeat
                    
                    if time_since_heartbeat > timedelta(minutes=5):  # 5 minute timeout
                        logger.warning(f"Client {client_id} heartbeat timeout")
                        disconnected_clients.append(client_id)
                
                # Remove disconnected clients
                for client_id in disconnected_clients:
                    self._remove_client(client_id)
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                time.sleep(30)
        
        logger.info("Heartbeat loop stopped")
    
    def _fetch_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch current price data for a ticker."""
        try:
            # Use Angel One API to get real-time price
            if self.angel_client.is_authenticated():
                price_data = self.angel_client.get_ltp_data([ticker])
                
                if price_data and ticker in price_data:
                    data = price_data[ticker]
                    
                    return {
                        'ticker': ticker,
                        'ltp': data.get('ltp', 0),
                        'open': data.get('open', 0),
                        'high': data.get('high', 0),
                        'low': data.get('low', 0),
                        'close': data.get('close', 0),
                        'volume': data.get('volume', 0),
                        'change': data.get('change', 0),
                        'change_percent': data.get('change_percent', 0),
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Fallback to market service
            return self.market_service.get_current_price(ticker)
            
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {str(e)}")
            return None
    
    def _generate_prediction(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Generate ML prediction for a ticker."""
        try:
            # Get historical data for prediction
            historical_data = self.market_service.get_historical_data(
                ticker, 
                period='1mo'  # Last month for prediction
            )
            
            if historical_data is None or len(historical_data) < 30:
                logger.warning(f"Insufficient data for prediction: {ticker}")
                return None
            
            # Generate prediction using prediction service
            prediction_result = self.prediction_service.generate_price_prediction(
                ticker=ticker,
                timeframe='1d',
                prediction_days=5
            )
            
            if prediction_result:
                return {
                    'ticker': ticker,
                    'predictions': prediction_result.get('predictions', []),
                    'confidence': prediction_result.get('confidence', 0),
                    'signal': prediction_result.get('signal', 'hold'),
                    'target_price': prediction_result.get('target_price', 0),
                    'stop_loss': prediction_result.get('stop_loss', 0),
                    'model_used': prediction_result.get('model_used', 'ensemble'),
                    'timestamp': datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating prediction for {ticker}: {str(e)}")
            return None
    
    def _calculate_technical_indicators(self, ticker: str, price_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate technical indicators for a ticker."""
        try:
            # Get recent historical data for technical analysis
            historical_data = self.market_service.get_historical_data(
                ticker, 
                period='3mo'  # 3 months for technical indicators
            )
            
            if historical_data is None or len(historical_data) < 20:
                return None
            
            # Calculate technical indicators using market service
            indicators = self.market_service.calculate_technical_indicators(historical_data)
            
            if indicators:
                return {
                    'ticker': ticker,
                    'rsi': indicators.get('rsi', 0),
                    'macd': indicators.get('macd', {}),
                    'bollinger_bands': indicators.get('bollinger_bands', {}),
                    'sma_20': indicators.get('sma_20', 0),
                    'sma_50': indicators.get('sma_50', 0),
                    'ema_12': indicators.get('ema_12', 0),
                    'ema_26': indicators.get('ema_26', 0),
                    'volume_sma': indicators.get('volume_sma', 0),
                    'timestamp': datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {ticker}: {str(e)}")
            return None
    
    async def _broadcast_price_update(self, ticker: str, price_data: Dict[str, Any]):
        """Broadcast price update to subscribed clients."""
        try:
            message = {
                'type': MessageType.PRICE_UPDATE.value,
                'data': {
                    'ticker': ticker,
                    'price_data': price_data
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to subscribed clients
            for client_id, client in list(self.clients.items()):
                try:
                    if (ticker in client.subscribed_tickers and 
                        'price' in client.subscription_types):
                        
                        await self._send_message_to_client(client_id, message)
                        client.last_price_update[ticker] = datetime.now()
                
                except Exception as e:
                    logger.error(f"Error sending price update to client {client_id}: {str(e)}")
                    # Remove problematic client
                    self._remove_client(client_id)
            
        except Exception as e:
            logger.error(f"Error broadcasting price update: {str(e)}")
    
    async def _broadcast_prediction_update(self, ticker: str, prediction_data: Dict[str, Any]):
        """Broadcast prediction update to subscribed clients."""
        try:
            message = {
                'type': MessageType.PREDICTION_UPDATE.value,
                'data': {
                    'ticker': ticker,
                    'prediction_data': prediction_data
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to subscribed clients
            for client_id, client in list(self.clients.items()):
                try:
                    if (ticker in client.subscribed_tickers and 
                        'prediction' in client.subscription_types):
                        
                        await self._send_message_to_client(client_id, message)
                        client.last_prediction_update[ticker] = datetime.now()
                
                except Exception as e:
                    logger.error(f"Error sending prediction update to client {client_id}: {str(e)}")
                    # Remove problematic client
                    self._remove_client(client_id)
            
        except Exception as e:
            logger.error(f"Error broadcasting prediction update: {str(e)}")
    
    async def _broadcast_technical_update(self, ticker: str, technical_data: Dict[str, Any]):
        """Broadcast technical indicator update to subscribed clients."""
        try:
            message = {
                'type': MessageType.TECHNICAL_UPDATE.value,
                'data': {
                    'ticker': ticker,
                    'technical_data': technical_data
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to subscribed clients
            for client_id, client in list(self.clients.items()):
                try:
                    if (ticker in client.subscribed_tickers and 
                        'technical' in client.subscription_types):
                        
                        await self._send_message_to_client(client_id, message)
                
                except Exception as e:
                    logger.error(f"Error sending technical update to client {client_id}: {str(e)}")
                    # Remove problematic client
                    self._remove_client(client_id)
            
        except Exception as e:
            logger.error(f"Error broadcasting technical update: {str(e)}")
    
    async def _send_message_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to a specific client."""
        try:
            client = self.clients.get(client_id)
            if not client:
                return
            
            # Add client_id to message
            message['client_id'] = client_id
            
            # Send message (handle both FastAPI WebSocket and websockets library)
            if hasattr(client.websocket, 'send_text'):
                # FastAPI WebSocket
                await client.websocket.send_text(json.dumps(message))
            else:
                # websockets library WebSocket
                await client.websocket.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {str(e)}")
            # Remove problematic client
            self._remove_client(client_id)
    
    def _remove_client(self, client_id: str):
        """Remove a client and update active tickers."""
        try:
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"Removed client {client_id}")
                
                # Update active tickers
                self._update_active_tickers()
        
        except Exception as e:
            logger.error(f"Error removing client {client_id}: {str(e)}")
    
    def _update_active_tickers(self):
        """Update active tickers based on client subscriptions."""
        try:
            # Collect all subscribed tickers
            subscribed_tickers = set()
            for client in self.clients.values():
                subscribed_tickers.update(client.subscribed_tickers)
            
            # Update active tickers
            old_tickers = self.active_tickers.copy()
            self.active_tickers = subscribed_tickers
            
            # Clean up cache for removed tickers
            removed_tickers = old_tickers - subscribed_tickers
            for ticker in removed_tickers:
                self.price_cache.pop(ticker, None)
                self.prediction_cache.pop(ticker, None)
                self.technical_cache.pop(ticker, None)
                self.news_cache.pop(ticker, None)
            
            if removed_tickers:
                logger.info(f"Removed tickers from cache: {removed_tickers}")
        
        except Exception as e:
            logger.error(f"Error updating active tickers: {str(e)}")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        try:
            current_time = datetime.now()
            
            # Calculate client details
            client_details = {}
            for client_id, client in self.clients.items():
                client_details[client_id] = {
                    'subscribed_tickers': list(client.subscribed_tickers),
                    'subscription_types': list(client.subscription_types),
                    'last_heartbeat': client.last_heartbeat.isoformat(),
                    'prediction_interval': client.prediction_interval
                }
            
            return {
                'active_clients': len(self.clients),
                'active_tickers': len(self.active_tickers),
                'tickers_list': list(self.active_tickers),
                'streaming_active': self.streaming_active,
                'server_uptime': current_time.isoformat(),
                'client_details': client_details,
                'cache_stats': {
                    'price_cache_size': len(self.price_cache),
                    'prediction_cache_size': len(self.prediction_cache),
                    'technical_cache_size': len(self.technical_cache),
                    'news_cache_size': len(self.news_cache)
                }
            }
        
        except Exception as e:
            logger.error(f"Error getting server stats: {str(e)}")
            return {
                'active_clients': 0,
                'active_tickers': 0,
                'tickers_list': [],
                'streaming_active': False,
                'server_uptime': datetime.now().isoformat(),
                'client_details': {},
                'error': str(e)
            }