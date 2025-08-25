import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketClient:
    """
    Example WebSocket client for testing the stock prediction streaming service.
    """
    
    def __init__(self, uri: str = "ws://localhost:8765"):
        """
        Initialize WebSocket client.
        
        Args:
            uri: WebSocket server URI
        """
        self.uri = uri
        self.websocket = None
        self.client_id = None
        self.connected = False
        
        # Message handlers
        self.message_handlers = {
            'status': self._handle_status,
            'price_update': self._handle_price_update,
            'prediction_update': self._handle_prediction_update,
            'technical_update': self._handle_technical_update,
            'news_update': self._handle_news_update,
            'error': self._handle_error,
            'heartbeat': self._handle_heartbeat
        }
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'price_updates': 0,
            'prediction_updates': 0,
            'technical_updates': 0,
            'errors': 0,
            'start_time': None
        }
    
    async def connect(self):
        """
        Connect to WebSocket server.
        """
        try:
            logger.info(f"Connecting to {self.uri}")
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            self.stats['start_time'] = datetime.now()
            logger.info("Connected to WebSocket server")
            
            # Start message handling
            await self._handle_messages()
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {str(e)}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """
        Disconnect from WebSocket server.
        """
        try:
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            
            self.connected = False
            logger.info("Disconnected from WebSocket server")
            
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")
    
    async def subscribe(self, tickers: List[str], types: List[str] = None, prediction_interval: int = 30):
        """
        Subscribe to ticker updates.
        
        Args:
            tickers: List of stock tickers
            types: List of subscription types (price, prediction, technical, news)
            prediction_interval: Prediction update interval in seconds
        """
        if not self.connected:
            raise Exception("Not connected to WebSocket server")
        
        if types is None:
            types = ['price', 'prediction', 'technical']
        
        message = {
            'type': 'subscribe',
            'data': {
                'tickers': tickers,
                'types': types,
                'prediction_interval': prediction_interval
            }
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Subscribed to {tickers} for {types}")
    
    async def unsubscribe(self, tickers: List[str] = None, types: List[str] = None):
        """
        Unsubscribe from ticker updates.
        
        Args:
            tickers: List of stock tickers (None for all)
            types: List of subscription types (None for all)
        """
        if not self.connected:
            raise Exception("Not connected to WebSocket server")
        
        message = {
            'type': 'unsubscribe',
            'data': {
                'tickers': tickers or [],
                'types': types or []
            }
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Unsubscribed from {tickers or 'all'} for {types or 'all'}")
    
    async def send_heartbeat(self):
        """
        Send heartbeat message.
        """
        if not self.connected:
            return
        
        message = {
            'type': 'heartbeat',
            'data': {
                'client_time': datetime.now().isoformat()
            }
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def _handle_messages(self):
        """
        Handle incoming WebSocket messages.
        """
        try:
            async for message in self.websocket:
                await self._process_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Error handling messages: {str(e)}")
            self.connected = False
    
    async def _process_message(self, message: str):
        """
        Process incoming message.
        
        Args:
            message: Raw message string
        """
        try:
            data = json.loads(message)
            message_type = data.get('type')
            message_data = data.get('data', {})
            timestamp = data.get('timestamp')
            client_id = data.get('client_id')
            
            # Update client ID if provided
            if client_id and not self.client_id:
                self.client_id = client_id
            
            # Update statistics
            self.stats['messages_received'] += 1
            
            # Handle message based on type
            handler = self.message_handlers.get(message_type)
            if handler:
                await handler(message_data, timestamp)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    async def _handle_status(self, data: Dict[str, Any], timestamp: str):
        """
        Handle status message.
        
        Args:
            data: Message data
            timestamp: Message timestamp
        """
        status = data.get('status')
        logger.info(f"Status update: {status}")
        
        if status == 'connected':
            self.client_id = data.get('client_id')
            logger.info(f"Assigned client ID: {self.client_id}")
            
            # Print available message types
            available_types = data.get('available_message_types', [])
            logger.info(f"Available message types: {available_types}")
        
        elif status in ['subscribed', 'unsubscribed']:
            tickers = data.get('tickers', [])
            types = data.get('types', [])
            logger.info(f"Subscription update - Tickers: {tickers}, Types: {types}")
    
    async def _handle_price_update(self, data: Dict[str, Any], timestamp: str):
        """
        Handle price update message.
        
        Args:
            data: Message data
            timestamp: Message timestamp
        """
        ticker = data.get('ticker')
        price_data = data.get('price_data', {})
        
        self.stats['price_updates'] += 1
        
        # Extract key price information
        ltp = price_data.get('ltp', 0.0)
        change = price_data.get('change', 0.0)
        change_percent = price_data.get('change_percent', 0.0)
        volume = price_data.get('volume', 0)
        
        logger.info(
            f"Price Update - {ticker}: ₹{ltp:.2f} "
            f"({change:+.2f}, {change_percent:+.2f}%) Vol: {volume:,}"
        )
    
    async def _handle_prediction_update(self, data: Dict[str, Any], timestamp: str):
        """
        Handle prediction update message.
        
        Args:
            data: Message data
            timestamp: Message timestamp
        """
        ticker = data.get('ticker')
        prediction_data = data.get('prediction_data', {})
        
        self.stats['prediction_updates'] += 1
        
        # Extract prediction information
        signal = prediction_data.get('signal', 'HOLD')
        confidence = prediction_data.get('confidence', 0.0)
        price_change = prediction_data.get('price_change_forecast', 0.0)
        risk_score = prediction_data.get('risk_score', 0.0)
        sentiment = prediction_data.get('sentiment_score', 0.0)
        
        logger.info(
            f"Prediction Update - {ticker}: {signal} "
            f"(Conf: {confidence:.1%}, Change: {price_change:+.2f}%, "
            f"Risk: {risk_score:.2f}, Sentiment: {sentiment:+.2f})"
        )
        
        # Show ensemble details if available
        ensemble = prediction_data.get('model_ensemble', {})
        if ensemble:
            xgb_signal = ensemble.get('xgboost_signal', 'N/A')
            inf_signal = ensemble.get('informer_signal', 'N/A')
            dqn_signal = ensemble.get('dqn_signal', 'N/A')
            
            logger.info(
                f"  Ensemble - XGB: {xgb_signal}, Informer: {inf_signal}, DQN: {dqn_signal}"
            )
    
    async def _handle_technical_update(self, data: Dict[str, Any], timestamp: str):
        """
        Handle technical analysis update message.
        
        Args:
            data: Message data
            timestamp: Message timestamp
        """
        ticker = data.get('ticker')
        technical_data = data.get('technical_data', {})
        
        self.stats['technical_updates'] += 1
        
        # Extract technical indicators
        rsi = technical_data.get('rsi', 0.0)
        macd = technical_data.get('macd', 0.0)
        bb_upper = technical_data.get('bb_upper', 0.0)
        bb_lower = technical_data.get('bb_lower', 0.0)
        sma_20 = technical_data.get('sma_20', 0.0)
        
        logger.info(
            f"Technical Update - {ticker}: RSI: {rsi:.1f}, MACD: {macd:.4f}, "
            f"BB: {bb_lower:.2f}-{bb_upper:.2f}, SMA20: {sma_20:.2f}"
        )
    
    async def _handle_news_update(self, data: Dict[str, Any], timestamp: str):
        """
        Handle news update message.
        
        Args:
            data: Message data
            timestamp: Message timestamp
        """
        ticker = data.get('ticker')
        news_data = data.get('news_data', {})
        
        logger.info(f"News Update - {ticker}: {news_data}")
    
    async def _handle_error(self, data: Dict[str, Any], timestamp: str):
        """
        Handle error message.
        
        Args:
            data: Message data
            timestamp: Message timestamp
        """
        error_message = data.get('error', 'Unknown error')
        self.stats['errors'] += 1
        
        logger.error(f"Server Error: {error_message}")
    
    async def _handle_heartbeat(self, data: Dict[str, Any], timestamp: str):
        """
        Handle heartbeat message.
        
        Args:
            data: Message data
            timestamp: Message timestamp
        """
        server_time = data.get('server_time')
        logger.debug(f"Heartbeat received - Server time: {server_time}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Client statistics
        """
        stats = self.stats.copy()
        
        if stats['start_time']:
            uptime = (datetime.now() - stats['start_time']).total_seconds()
            stats['uptime_seconds'] = uptime
            stats['messages_per_second'] = stats['messages_received'] / max(uptime, 1)
        
        stats['connected'] = self.connected
        stats['client_id'] = self.client_id
        
        return stats

# Example usage functions
async def basic_example():
    """
    Basic WebSocket client example.
    """
    client = WebSocketClient()
    
    try:
        # Connect to server
        await client.connect()
        
        # Subscribe to some tickers
        await client.subscribe(
            tickers=['RELIANCE', 'TCS', 'INFY', 'HDFCBANK'],
            types=['price', 'prediction', 'technical'],
            prediction_interval=30
        )
        
        # Run for 2 minutes
        await asyncio.sleep(120)
        
        # Print statistics
        stats = client.get_stats()
        print("\nClient Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except KeyboardInterrupt:
        print("\nStopping client...")
    finally:
        await client.disconnect()

async def interactive_example():
    """
    Interactive WebSocket client example.
    """
    client = WebSocketClient()
    
    try:
        # Connect to server
        await client.connect()
        
        print("\nWebSocket Client Connected!")
        print("Available commands:")
        print("  1. Subscribe to tickers")
        print("  2. Unsubscribe from tickers")
        print("  3. Send heartbeat")
        print("  4. Show statistics")
        print("  5. Quit")
        
        while client.connected:
            try:
                # This is a simplified example - in a real implementation,
                # you'd want to handle user input properly in an async way
                await asyncio.sleep(1)
                
                # For demo purposes, let's subscribe to some tickers automatically
                if client.stats['messages_received'] == 1:  # After welcome message
                    await client.subscribe(
                        tickers=['RELIANCE', 'TCS'],
                        types=['price', 'prediction'],
                        prediction_interval=20
                    )
                
                # Send periodic heartbeat
                if client.stats['messages_received'] % 30 == 0:
                    await client.send_heartbeat()
                
                # Stop after 60 seconds for demo
                if client.stats.get('uptime_seconds', 0) > 60:
                    break
                    
            except KeyboardInterrupt:
                break
        
        # Print final statistics
        stats = client.get_stats()
        print("\nFinal Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error in interactive example: {str(e)}")
    finally:
        await client.disconnect()

async def stress_test_example():
    """
    Stress test with multiple clients.
    """
    clients = []
    num_clients = 5
    
    try:
        print(f"Starting stress test with {num_clients} clients...")
        
        # Create and connect multiple clients
        for i in range(num_clients):
            client = WebSocketClient()
            await client.connect()
            clients.append(client)
            
            # Subscribe each client to different tickers
            tickers = [f'STOCK{j}' for j in range(i*2, (i+1)*2)]
            await client.subscribe(
                tickers=tickers,
                types=['price', 'prediction'],
                prediction_interval=15
            )
            
            print(f"Client {i+1} connected and subscribed to {tickers}")
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Print statistics for all clients
        print("\nStress Test Results:")
        total_messages = 0
        
        for i, client in enumerate(clients):
            stats = client.get_stats()
            total_messages += stats['messages_received']
            print(f"Client {i+1}: {stats['messages_received']} messages, "
                  f"{stats.get('messages_per_second', 0):.2f} msg/sec")
        
        print(f"Total messages across all clients: {total_messages}")
        
    except Exception as e:
        print(f"Error in stress test: {str(e)}")
    finally:
        # Disconnect all clients
        for client in clients:
            await client.disconnect()

# Main execution
if __name__ == "__main__":
    import sys
    
    print("WebSocket Client Examples")
    print("========================")
    print("1. Basic Example (2 minutes)")
    print("2. Interactive Example (1 minute)")
    print("3. Stress Test (30 seconds, 5 clients)")
    
    try:
        choice = input("\nSelect example (1-3): ").strip()
        
        if choice == '1':
            print("\nRunning basic example...")
            asyncio.run(basic_example())
        elif choice == '2':
            print("\nRunning interactive example...")
            asyncio.run(interactive_example())
        elif choice == '3':
            print("\nRunning stress test...")
            asyncio.run(stress_test_example())
        else:
            print("Invalid choice. Running basic example...")
            asyncio.run(basic_example())
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")