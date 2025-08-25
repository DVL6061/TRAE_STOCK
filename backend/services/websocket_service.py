import os
import sys
import asyncio
import json
import logging
import websockets
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import uuid
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our services and clients
from data.angel_one_client import AngelOneClient
from services.prediction_service import PredictionService
from data.technical_indicators import TechnicalIndicators

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
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    STATUS = "status"

@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    client_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        message_dict = asdict(self)
        message_dict['timestamp'] = self.timestamp.isoformat()
        return json.dumps(message_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create from JSON string."""
        data = json.loads(json_str)
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class ClientSubscription:
    """Client subscription information."""
    client_id: str
    websocket: websockets.WebSocketServerProtocol
    subscribed_tickers: Set[str]
    subscription_types: Set[str]  # price, prediction, technical, news
    last_heartbeat: datetime
    prediction_interval: int = 30  # seconds
    
class WebSocketDataStreamer:
    """
    Real-time WebSocket data streaming service for stock market data.
    Provides live price feeds, predictions, and technical analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the WebSocket streaming service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize data clients and services
        self.angel_client = AngelOneClient(
            api_key=config.get('angel_api_key'),
            client_id=config.get('angel_client_id'),
            password=config.get('angel_password'),
            totp_secret=config.get('angel_totp_secret')
        )
        
        self.prediction_service = PredictionService(config)
        self.technical_indicators = TechnicalIndicators()
        
        # WebSocket server configuration
        self.host = config.get('websocket_host', 'localhost')
        self.port = config.get('websocket_port', 8765)
        
        # Client management
        self.clients: Dict[str, ClientSubscription] = {}
        self.active_tickers: Set[str] = set()
        
        # Data streaming
        self.streaming_active = False
        self.data_queue = Queue()
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.prediction_cache: Dict[str, Dict[str, Any]] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.streaming_thread = None
        self.prediction_thread = None
        
        # Heartbeat configuration
        self.heartbeat_interval = 30  # seconds
        self.client_timeout = 60  # seconds
        
        logger.info(f"WebSocket service initialized on {self.host}:{self.port}")
    
    async def start_server(self):
        """
        Start the WebSocket server.
        """
        try:
            logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            
            # Start background tasks
            self.streaming_active = True
            
            # Start data streaming thread
            self.streaming_thread = threading.Thread(
                target=self._start_data_streaming_thread,
                daemon=True
            )
            self.streaming_thread.start()
            
            # Start prediction updates thread
            self.prediction_thread = threading.Thread(
                target=self._start_prediction_thread,
                daemon=True
            )
            self.prediction_thread.start()
            
            # Start WebSocket server
            async with websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10
            ) as server:
                logger.info("WebSocket server started successfully")
                
                # Start heartbeat task
                heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                # Keep server running
                await asyncio.Future()  # Run forever
                
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {str(e)}")
            raise
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """
        Handle new WebSocket client connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        client_id = str(uuid.uuid4())
        logger.info(f"New client connected: {client_id}")
        
        # Create client subscription
        client_subscription = ClientSubscription(
            client_id=client_id,
            websocket=websocket,
            subscribed_tickers=set(),
            subscription_types=set(),
            last_heartbeat=datetime.now()
        )
        
        self.clients[client_id] = client_subscription
        
        try:
            # Send welcome message
            welcome_message = WebSocketMessage(
                type=MessageType.STATUS.value,
                data={
                    'status': 'connected',
                    'client_id': client_id,
                    'server_time': datetime.now().isoformat(),
                    'available_message_types': [t.value for t in MessageType]
                },
                timestamp=datetime.now(),
                client_id=client_id
            )
            
            await websocket.send(welcome_message.to_json())
            
            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {str(e)}")
        finally:
            # Clean up client
            await self._cleanup_client(client_id)
    
    async def _handle_client_message(self, client_id: str, message: str):
        """
        Handle incoming client message.
        
        Args:
            client_id: Client identifier
            message: Raw message string
        """
        try:
            # Parse message
            data = json.loads(message)
            message_type = data.get('type')
            message_data = data.get('data', {})
            
            client = self.clients.get(client_id)
            if not client:
                return
            
            # Update heartbeat
            client.last_heartbeat = datetime.now()
            
            # Handle different message types
            if message_type == MessageType.SUBSCRIBE.value:
                await self._handle_subscribe(client_id, message_data)
            
            elif message_type == MessageType.UNSUBSCRIBE.value:
                await self._handle_unsubscribe(client_id, message_data)
            
            elif message_type == MessageType.HEARTBEAT.value:
                await self._handle_heartbeat(client_id)
            
            else:
                # Send error for unknown message type
                error_message = WebSocketMessage(
                    type=MessageType.ERROR.value,
                    data={
                        'error': f'Unknown message type: {message_type}',
                        'received_message': data
                    },
                    timestamp=datetime.now(),
                    client_id=client_id
                )
                
                await client.websocket.send(error_message.to_json())
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from client {client_id}: {str(e)}")
            await self._send_error(client_id, f"Invalid JSON: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error handling message from client {client_id}: {str(e)}")
            await self._send_error(client_id, f"Message handling error: {str(e)}")
    
    async def _handle_subscribe(self, client_id: str, data: Dict[str, Any]):
        """
        Handle subscription request.
        
        Args:
            client_id: Client identifier
            data: Subscription data
        """
        try:
            client = self.clients.get(client_id)
            if not client:
                return
            
            tickers = data.get('tickers', [])
            subscription_types = data.get('types', ['price'])
            prediction_interval = data.get('prediction_interval', 30)
            
            # Validate tickers
            valid_tickers = []
            for ticker in tickers:
                if isinstance(ticker, str) and ticker.strip():
                    valid_tickers.append(ticker.upper().strip())
            
            if not valid_tickers:
                await self._send_error(client_id, "No valid tickers provided")
                return
            
            # Validate subscription types
            valid_types = []
            for sub_type in subscription_types:
                if sub_type in ['price', 'prediction', 'technical', 'news']:
                    valid_types.append(sub_type)
            
            if not valid_types:
                await self._send_error(client_id, "No valid subscription types provided")
                return
            
            # Update client subscription
            client.subscribed_tickers.update(valid_tickers)
            client.subscription_types.update(valid_types)
            client.prediction_interval = max(10, min(300, prediction_interval))  # 10s to 5min
            
            # Update active tickers
            self.active_tickers.update(valid_tickers)
            
            # Send confirmation
            confirmation_message = WebSocketMessage(
                type=MessageType.STATUS.value,
                data={
                    'status': 'subscribed',
                    'tickers': list(client.subscribed_tickers),
                    'types': list(client.subscription_types),
                    'prediction_interval': client.prediction_interval
                },
                timestamp=datetime.now(),
                client_id=client_id
            )
            
            await client.websocket.send(confirmation_message.to_json())
            
            # Send initial data if available
            for ticker in valid_tickers:
                if ticker in self.price_cache:
                    await self._send_price_update(client_id, ticker, self.price_cache[ticker])
                
                if ticker in self.prediction_cache and 'prediction' in valid_types:
                    await self._send_prediction_update(client_id, ticker, self.prediction_cache[ticker])
            
            logger.info(f"Client {client_id} subscribed to {valid_tickers} for {valid_types}")
            
        except Exception as e:
            logger.error(f"Error handling subscription: {str(e)}")
            await self._send_error(client_id, f"Subscription error: {str(e)}")
    
    async def _handle_unsubscribe(self, client_id: str, data: Dict[str, Any]):
        """
        Handle unsubscription request.
        
        Args:
            client_id: Client identifier
            data: Unsubscription data
        """
        try:
            client = self.clients.get(client_id)
            if not client:
                return
            
            tickers = data.get('tickers', [])
            subscription_types = data.get('types', [])
            
            # Remove tickers
            if tickers:
                for ticker in tickers:
                    client.subscribed_tickers.discard(ticker.upper().strip())
            else:
                client.subscribed_tickers.clear()
            
            # Remove subscription types
            if subscription_types:
                for sub_type in subscription_types:
                    client.subscription_types.discard(sub_type)
            else:
                client.subscription_types.clear()
            
            # Update active tickers
            self._update_active_tickers()
            
            # Send confirmation
            confirmation_message = WebSocketMessage(
                type=MessageType.STATUS.value,
                data={
                    'status': 'unsubscribed',
                    'remaining_tickers': list(client.subscribed_tickers),
                    'remaining_types': list(client.subscription_types)
                },
                timestamp=datetime.now(),
                client_id=client_id
            )
            
            await client.websocket.send(confirmation_message.to_json())
            
            logger.info(f"Client {client_id} unsubscribed from {tickers}")
            
        except Exception as e:
            logger.error(f"Error handling unsubscription: {str(e)}")
            await self._send_error(client_id, f"Unsubscription error: {str(e)}")
    
    async def _handle_heartbeat(self, client_id: str):
        """
        Handle heartbeat message.
        
        Args:
            client_id: Client identifier
        """
        try:
            client = self.clients.get(client_id)
            if not client:
                return
            
            client.last_heartbeat = datetime.now()
            
            # Send heartbeat response
            heartbeat_message = WebSocketMessage(
                type=MessageType.HEARTBEAT.value,
                data={
                    'server_time': datetime.now().isoformat(),
                    'client_id': client_id
                },
                timestamp=datetime.now(),
                client_id=client_id
            )
            
            await client.websocket.send(heartbeat_message.to_json())
            
        except Exception as e:
            logger.error(f"Error handling heartbeat: {str(e)}")
    
    async def _send_error(self, client_id: str, error_message: str):
        """
        Send error message to client.
        
        Args:
            client_id: Client identifier
            error_message: Error message
        """
        try:
            client = self.clients.get(client_id)
            if not client:
                return
            
            error_msg = WebSocketMessage(
                type=MessageType.ERROR.value,
                data={
                    'error': error_message,
                    'timestamp': datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                client_id=client_id
            )
            
            await client.websocket.send(error_msg.to_json())
            
        except Exception as e:
            logger.error(f"Error sending error message: {str(e)}")
    
    async def _send_price_update(self, client_id: str, ticker: str, price_data: Dict[str, Any]):
        """
        Send price update to client.
        
        Args:
            client_id: Client identifier
            ticker: Stock ticker
            price_data: Price data
        """
        try:
            client = self.clients.get(client_id)
            if not client or ticker not in client.subscribed_tickers or 'price' not in client.subscription_types:
                return
            
            price_message = WebSocketMessage(
                type=MessageType.PRICE_UPDATE.value,
                data={
                    'ticker': ticker,
                    'price_data': price_data
                },
                timestamp=datetime.now(),
                client_id=client_id
            )
            
            await client.websocket.send(price_message.to_json())
            
        except Exception as e:
            logger.error(f"Error sending price update: {str(e)}")
    
    async def _send_prediction_update(self, client_id: str, ticker: str, prediction_data: Dict[str, Any]):
        """
        Send prediction update to client.
        
        Args:
            client_id: Client identifier
            ticker: Stock ticker
            prediction_data: Prediction data
        """
        try:
            client = self.clients.get(client_id)
            if not client or ticker not in client.subscribed_tickers or 'prediction' not in client.subscription_types:
                return
            
            prediction_message = WebSocketMessage(
                type=MessageType.PREDICTION_UPDATE.value,
                data={
                    'ticker': ticker,
                    'prediction_data': prediction_data
                },
                timestamp=datetime.now(),
                client_id=client_id
            )
            
            await client.websocket.send(prediction_message.to_json())
            
        except Exception as e:
            logger.error(f"Error sending prediction update: {str(e)}")
    
    async def _send_technical_update(self, client_id: str, ticker: str, technical_data: Dict[str, Any]):
        """
        Send technical analysis update to client.
        
        Args:
            client_id: Client identifier
            ticker: Stock ticker
            technical_data: Technical analysis data
        """
        try:
            client = self.clients.get(client_id)
            if not client or ticker not in client.subscribed_tickers or 'technical' not in client.subscription_types:
                return
            
            technical_message = WebSocketMessage(
                type=MessageType.TECHNICAL_UPDATE.value,
                data={
                    'ticker': ticker,
                    'technical_data': technical_data
                },
                timestamp=datetime.now(),
                client_id=client_id
            )
            
            await client.websocket.send(technical_message.to_json())
            
        except Exception as e:
            logger.error(f"Error sending technical update: {str(e)}")
    
    def _start_data_streaming_thread(self):
        """
        Start data streaming in a separate thread.
        """
        logger.info("Starting data streaming thread")
        
        while self.streaming_active:
            try:
                if self.active_tickers:
                    # Fetch real-time data for active tickers
                    for ticker in list(self.active_tickers):
                        try:
                            # Get current price data
                            price_data = self._fetch_current_price(ticker)
                            
                            if price_data:
                                self.price_cache[ticker] = price_data
                                
                                # Queue price update for all subscribed clients
                                asyncio.run_coroutine_threadsafe(
                                    self._broadcast_price_update(ticker, price_data),
                                    asyncio.get_event_loop()
                                )
                                
                                # Get technical analysis if needed
                                technical_data = self._calculate_technical_indicators(ticker, price_data)
                                if technical_data:
                                    asyncio.run_coroutine_threadsafe(
                                        self._broadcast_technical_update(ticker, technical_data),
                                        asyncio.get_event_loop()
                                    )
                        
                        except Exception as e:
                            logger.error(f"Error fetching data for {ticker}: {str(e)}")
                
                # Sleep between updates
                time.sleep(self.config.get('price_update_interval', 5))
                
            except Exception as e:
                logger.error(f"Error in data streaming thread: {str(e)}")
                time.sleep(5)
    
    def _start_prediction_thread(self):
        """
        Start prediction updates in a separate thread.
        """
        logger.info("Starting prediction thread")
        
        while self.streaming_active:
            try:
                if self.active_tickers:
                    for ticker in list(self.active_tickers):
                        try:
                            # Check if any client needs prediction updates
                            needs_prediction = any(
                                ticker in client.subscribed_tickers and 'prediction' in client.subscription_types
                                for client in self.clients.values()
                            )
                            
                            if needs_prediction:
                                # Generate prediction
                                prediction_data = self._generate_prediction(ticker)
                                
                                if prediction_data:
                                    self.prediction_cache[ticker] = prediction_data
                                    
                                    # Broadcast prediction update
                                    asyncio.run_coroutine_threadsafe(
                                        self._broadcast_prediction_update(ticker, prediction_data),
                                        asyncio.get_event_loop()
                                    )
                        
                        except Exception as e:
                            logger.error(f"Error generating prediction for {ticker}: {str(e)}")
                
                # Sleep between prediction updates
                time.sleep(self.config.get('prediction_update_interval', 30))
                
            except Exception as e:
                logger.error(f"Error in prediction thread: {str(e)}")
                time.sleep(10)
    
    def _fetch_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current price data for ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Current price data or None
        """
        try:
            # Try to get real-time data from Angel One
            current_data = self.angel_client.get_ltp(ticker)
            
            if current_data:
                return {
                    'ticker': ticker,
                    'ltp': current_data.get('ltp', 0.0),
                    'open': current_data.get('open', 0.0),
                    'high': current_data.get('high', 0.0),
                    'low': current_data.get('low', 0.0),
                    'close': current_data.get('close', 0.0),
                    'volume': current_data.get('volume', 0),
                    'change': current_data.get('change', 0.0),
                    'change_percent': current_data.get('change_percent', 0.0),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Generate mock data for testing
                return self._generate_mock_price_data(ticker)
                
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {str(e)}")
            return self._generate_mock_price_data(ticker)
    
    def _generate_mock_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Generate mock price data for testing.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Mock price data
        """
        # Use ticker hash for consistent mock data
        np.random.seed(hash(ticker) % 2**32)
        
        base_price = 100.0 + (hash(ticker) % 1000)
        change_percent = np.random.uniform(-2.0, 2.0)
        current_price = base_price * (1 + change_percent / 100)
        
        return {
            'ticker': ticker,
            'ltp': round(current_price, 2),
            'open': round(base_price, 2),
            'high': round(current_price * 1.02, 2),
            'low': round(current_price * 0.98, 2),
            'close': round(base_price, 2),
            'volume': int(np.random.uniform(100000, 1000000)),
            'change': round(current_price - base_price, 2),
            'change_percent': round(change_percent, 2),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_technical_indicators(self, ticker: str, price_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Calculate technical indicators for the ticker.
        
        Args:
            ticker: Stock ticker
            price_data: Current price data
            
        Returns:
            Technical indicators or None
        """
        try:
            # This would typically use historical data to calculate indicators
            # For now, return mock technical data
            
            current_price = price_data.get('ltp', 100.0)
            
            return {
                'ticker': ticker,
                'rsi': round(np.random.uniform(30, 70), 2),
                'macd': round(np.random.uniform(-2, 2), 4),
                'macd_signal': round(np.random.uniform(-2, 2), 4),
                'bb_upper': round(current_price * 1.02, 2),
                'bb_lower': round(current_price * 0.98, 2),
                'sma_20': round(current_price * np.random.uniform(0.98, 1.02), 2),
                'ema_20': round(current_price * np.random.uniform(0.98, 1.02), 2),
                'volume_sma': int(np.random.uniform(500000, 1500000)),
                'atr': round(current_price * 0.02, 2),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {ticker}: {str(e)}")
            return None
    
    def _generate_prediction(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Generate prediction for ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Prediction data or None
        """
        try:
            # This would use the actual prediction service
            # For now, return mock prediction data
            
            signals = ['BUY', 'SELL', 'HOLD']
            signal = np.random.choice(signals)
            confidence = np.random.uniform(0.6, 0.95)
            price_change = np.random.uniform(-5.0, 5.0)
            
            return {
                'ticker': ticker,
                'signal': signal,
                'confidence': round(confidence, 3),
                'price_change_forecast': round(price_change, 2),
                'prediction_horizon': 'intraday',
                'model_ensemble': {
                    'xgboost_signal': np.random.choice(signals),
                    'informer_signal': np.random.choice(signals),
                    'dqn_signal': np.random.choice(signals)
                },
                'risk_score': round(np.random.uniform(0.2, 0.8), 3),
                'sentiment_score': round(np.random.uniform(-0.5, 0.5), 3),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction for {ticker}: {str(e)}")
            return None
    
    async def _broadcast_price_update(self, ticker: str, price_data: Dict[str, Any]):
        """
        Broadcast price update to all subscribed clients.
        
        Args:
            ticker: Stock ticker
            price_data: Price data
        """
        for client_id, client in list(self.clients.items()):
            if ticker in client.subscribed_tickers and 'price' in client.subscription_types:
                await self._send_price_update(client_id, ticker, price_data)
    
    async def _broadcast_prediction_update(self, ticker: str, prediction_data: Dict[str, Any]):
        """
        Broadcast prediction update to all subscribed clients.
        
        Args:
            ticker: Stock ticker
            prediction_data: Prediction data
        """
        for client_id, client in list(self.clients.items()):
            if ticker in client.subscribed_tickers and 'prediction' in client.subscription_types:
                await self._send_prediction_update(client_id, ticker, prediction_data)
    
    async def _broadcast_technical_update(self, ticker: str, technical_data: Dict[str, Any]):
        """
        Broadcast technical update to all subscribed clients.
        
        Args:
            ticker: Stock ticker
            technical_data: Technical data
        """
        for client_id, client in list(self.clients.items()):
            if ticker in client.subscribed_tickers and 'technical' in client.subscription_types:
                await self._send_technical_update(client_id, ticker, technical_data)
    
    async def _heartbeat_loop(self):
        """
        Periodic heartbeat and client cleanup.
        """
        while self.streaming_active:
            try:
                current_time = datetime.now()
                
                # Check for inactive clients
                inactive_clients = []
                for client_id, client in self.clients.items():
                    time_since_heartbeat = (current_time - client.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.client_timeout:
                        inactive_clients.append(client_id)
                
                # Remove inactive clients
                for client_id in inactive_clients:
                    await self._cleanup_client(client_id)
                    logger.info(f"Removed inactive client: {client_id}")
                
                # Update active tickers
                self._update_active_tickers()
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(5)
    
    async def _cleanup_client(self, client_id: str):
        """
        Clean up client connection.
        
        Args:
            client_id: Client identifier
        """
        try:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                # Close WebSocket connection if still open
                if not client.websocket.closed:
                    await client.websocket.close()
                
                # Remove client
                del self.clients[client_id]
                
                # Update active tickers
                self._update_active_tickers()
                
                logger.info(f"Client {client_id} cleaned up")
                
        except Exception as e:
            logger.error(f"Error cleaning up client {client_id}: {str(e)}")
    
    def _update_active_tickers(self):
        """
        Update the set of active tickers based on client subscriptions.
        """
        active_tickers = set()
        
        for client in self.clients.values():
            active_tickers.update(client.subscribed_tickers)
        
        self.active_tickers = active_tickers
    
    def stop_server(self):
        """
        Stop the WebSocket server and all background tasks.
        """
        logger.info("Stopping WebSocket server")
        
        self.streaming_active = False
        
        # Close all client connections
        for client in self.clients.values():
            if not client.websocket.closed:
                asyncio.create_task(client.websocket.close())
        
        self.clients.clear()
        self.active_tickers.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("WebSocket server stopped")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Server statistics
        """
        return {
            'active_clients': len(self.clients),
            'active_tickers': len(self.active_tickers),
            'tickers_list': list(self.active_tickers),
            'streaming_active': self.streaming_active,
            'server_uptime': datetime.now().isoformat(),
            'client_details': {
                client_id: {
                    'subscribed_tickers': list(client.subscribed_tickers),
                    'subscription_types': list(client.subscription_types),
                    'last_heartbeat': client.last_heartbeat.isoformat()
                }
                for client_id, client in self.clients.items()
            }
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Configuration
    config = {
        'angel_api_key': os.getenv('ANGEL_API_KEY'),
        'angel_client_id': os.getenv('ANGEL_CLIENT_ID'),
        'angel_password': os.getenv('ANGEL_PASSWORD'),
        'angel_totp_secret': os.getenv('ANGEL_TOTP_SECRET'),
        'websocket_host': 'localhost',
        'websocket_port': 8765,
        'price_update_interval': 5,  # seconds
        'prediction_update_interval': 30,  # seconds
        'newsapi_key': os.getenv('NEWSAPI_KEY'),
        'alphavantage_key': os.getenv('ALPHAVANTAGE_KEY')
    }
    
    # Initialize and start WebSocket service
    websocket_service = WebSocketDataStreamer(config)
    
    try:
        print(f"Starting WebSocket server on {config['websocket_host']}:{config['websocket_port']}")
        print("Press Ctrl+C to stop the server")
        
        # Start the server
        asyncio.run(websocket_service.start_server())
        
    except KeyboardInterrupt:
        print("\nShutting down WebSocket server...")
        websocket_service.stop_server()
        print("Server stopped")
    
    except Exception as e:
        print(f"Error running WebSocket server: {str(e)}")
        websocket_service.stop_server()