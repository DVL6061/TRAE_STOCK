import os
import sys
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our services
from app.services.websocket_service import WebSocketDataStreamer, MessageType
from app.services.prediction_service import PredictionService
from app.data.angel_one_client import AngelOneClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global WebSocket service instance
websocket_service: Optional[WebSocketDataStreamer] = None

# Pydantic models for API
class SubscriptionRequest(BaseModel):
    """WebSocket subscription request model."""
    tickers: List[str] = Field(..., description="List of stock tickers to subscribe to")
    types: List[str] = Field(default=['price'], description="Subscription types: price, prediction, technical, news")
    prediction_interval: int = Field(default=30, ge=10, le=300, description="Prediction update interval in seconds")

class UnsubscriptionRequest(BaseModel):
    """WebSocket unsubscription request model."""
    tickers: Optional[List[str]] = Field(default=None, description="List of tickers to unsubscribe from (None for all)")
    types: Optional[List[str]] = Field(default=None, description="Subscription types to remove (None for all)")

class ServerStatsResponse(BaseModel):
    """Server statistics response model."""
    active_clients: int
    active_tickers: int
    tickers_list: List[str]
    streaming_active: bool
    server_uptime: str
    client_details: Dict[str, Any]

class TickerDataResponse(BaseModel):
    """Ticker data response model."""
    ticker: str
    price_data: Optional[Dict[str, Any]] = None
    prediction_data: Optional[Dict[str, Any]] = None
    technical_data: Optional[Dict[str, Any]] = None
    timestamp: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    global websocket_service
    
    # Startup
    logger.info("Starting WebSocket service...")
    
    # Configuration
    config = {
        'angel_api_key': os.getenv('ANGEL_API_KEY'),
        'angel_client_id': os.getenv('ANGEL_CLIENT_ID'),
        'angel_password': os.getenv('ANGEL_PASSWORD'),
        'angel_totp_secret': os.getenv('ANGEL_TOTP_SECRET'),
        'websocket_host': os.getenv('WEBSOCKET_HOST', 'localhost'),
        'websocket_port': int(os.getenv('WEBSOCKET_PORT', 8765)),
        'price_update_interval': int(os.getenv('PRICE_UPDATE_INTERVAL', 5)),
        'prediction_update_interval': int(os.getenv('PREDICTION_UPDATE_INTERVAL', 30)),
        'newsapi_key': os.getenv('NEWSAPI_KEY'),
        'alphavantage_key': os.getenv('ALPHAVANTAGE_KEY')
    }
    
    # Initialize WebSocket service
    websocket_service = WebSocketDataStreamer(config)
    
    # Start background streaming tasks
    websocket_service.streaming_active = True
    
    # Start data streaming thread
    import threading
    streaming_thread = threading.Thread(
        target=websocket_service._start_data_streaming_thread,
        daemon=True
    )
    streaming_thread.start()
    
    # Start prediction thread
    prediction_thread = threading.Thread(
        target=websocket_service._start_prediction_thread,
        daemon=True
    )
    prediction_thread.start()
    
    logger.info("WebSocket service started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down WebSocket service...")
    if websocket_service:
        websocket_service.stop_server()
    logger.info("WebSocket service stopped")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Stock Prediction WebSocket API",
    description="Real-time stock prediction and streaming API with WebSocket support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get WebSocket service
def get_websocket_service() -> WebSocketDataStreamer:
    """Get the global WebSocket service instance."""
    if websocket_service is None:
        raise HTTPException(status_code=503, detail="WebSocket service not available")
    return websocket_service

# REST API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Stock Prediction WebSocket API",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws",
            "stats": "/api/stats",
            "ticker_data": "/api/ticker/{ticker}",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    service = get_websocket_service()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "streaming_active": service.streaming_active,
        "active_clients": len(service.clients),
        "active_tickers": len(service.active_tickers)
    }

@app.get("/api/stats", response_model=ServerStatsResponse)
async def get_server_stats(service: WebSocketDataStreamer = Depends(get_websocket_service)):
    """Get server statistics."""
    stats = service.get_server_stats()
    return ServerStatsResponse(**stats)

@app.get("/api/ticker/{ticker}", response_model=TickerDataResponse)
async def get_ticker_data(
    ticker: str,
    include_prediction: bool = True,
    include_technical: bool = True,
    service: WebSocketDataStreamer = Depends(get_websocket_service)
):
    """Get current data for a specific ticker."""
    ticker = ticker.upper().strip()
    
    # Get cached data
    price_data = service.price_cache.get(ticker)
    prediction_data = service.prediction_cache.get(ticker) if include_prediction else None
    
    # Generate technical data if requested
    technical_data = None
    if include_technical and price_data:
        technical_data = service._calculate_technical_indicators(ticker, price_data)
    
    # If no cached data, try to fetch fresh data
    if not price_data:
        price_data = service._fetch_current_price(ticker)
        if price_data:
            service.price_cache[ticker] = price_data
    
    if not prediction_data and include_prediction:
        prediction_data = service._generate_prediction(ticker)
        if prediction_data:
            service.prediction_cache[ticker] = prediction_data
    
    return TickerDataResponse(
        ticker=ticker,
        price_data=price_data,
        prediction_data=prediction_data,
        technical_data=technical_data,
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/tickers")
async def get_active_tickers(service: WebSocketDataStreamer = Depends(get_websocket_service)):
    """Get list of active tickers."""
    return {
        "active_tickers": list(service.active_tickers),
        "count": len(service.active_tickers),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/ticker/{ticker}/subscribe")
async def add_ticker_to_stream(
    ticker: str,
    background_tasks: BackgroundTasks,
    service: WebSocketDataStreamer = Depends(get_websocket_service)
):
    """Add a ticker to the streaming service."""
    ticker = ticker.upper().strip()
    
    # Add to active tickers
    service.active_tickers.add(ticker)
    
    # Fetch initial data in background
    background_tasks.add_task(service._fetch_current_price, ticker)
    
    return {
        "message": f"Ticker {ticker} added to streaming",
        "ticker": ticker,
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/api/ticker/{ticker}/subscribe")
async def remove_ticker_from_stream(
    ticker: str,
    service: WebSocketDataStreamer = Depends(get_websocket_service)
):
    """Remove a ticker from the streaming service."""
    ticker = ticker.upper().strip()
    
    # Remove from active tickers if no clients are subscribed
    clients_subscribed = any(
        ticker in client.subscribed_tickers
        for client in service.clients.values()
    )
    
    if not clients_subscribed:
        service.active_tickers.discard(ticker)
        # Clear cached data
        service.price_cache.pop(ticker, None)
        service.prediction_cache.pop(ticker, None)
        
        return {
            "message": f"Ticker {ticker} removed from streaming",
            "ticker": ticker,
            "timestamp": datetime.now().isoformat()
        }
    else:
        return {
            "message": f"Ticker {ticker} still has active subscriptions",
            "ticker": ticker,
            "active_subscriptions": sum(
                1 for client in service.clients.values()
                if ticker in client.subscribed_tickers
            ),
            "timestamp": datetime.now().isoformat()
        }

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time data streaming."""
    service = get_websocket_service()
    
    # Accept WebSocket connection
    await websocket.accept()
    
    # Handle the connection using our WebSocket service
    # We need to adapt the websockets library interface to FastAPI's WebSocket
    await handle_fastapi_websocket(websocket, service)

async def handle_fastapi_websocket(websocket: WebSocket, service: WebSocketDataStreamer):
    """
    Handle FastAPI WebSocket connection using our WebSocket service logic.
    
    Args:
        websocket: FastAPI WebSocket connection
        service: WebSocket data streaming service
    """
    import uuid
    from app.services.websocket_service import ClientSubscription
    
    client_id = str(uuid.uuid4())
    logger.info(f"New FastAPI WebSocket client connected: {client_id}")
    
    # Create client subscription (adapted for FastAPI WebSocket)
    client_subscription = ClientSubscription(
        client_id=client_id,
        websocket=websocket,  # This will be a FastAPI WebSocket, not websockets.WebSocketServerProtocol
        subscribed_tickers=set(),
        subscription_types=set(),
        last_heartbeat=datetime.now()
    )
    
    service.clients[client_id] = client_subscription
    
    try:
        # Send welcome message
        welcome_message = {
            'type': MessageType.STATUS.value,
            'data': {
                'status': 'connected',
                'client_id': client_id,
                'server_time': datetime.now().isoformat(),
                'available_message_types': [t.value for t in MessageType]
            },
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id
        }
        
        await websocket.send_text(json.dumps(welcome_message))
        
        # Handle client messages
        while True:
            try:
                # Receive message
                message = await websocket.receive_text()
                await handle_fastapi_client_message(client_id, message, service, websocket)
                
            except WebSocketDisconnect:
                logger.info(f"FastAPI WebSocket client {client_id} disconnected")
                break
            except Exception as e:
                logger.error(f"Error handling FastAPI WebSocket message from {client_id}: {str(e)}")
                # Send error message
                error_message = {
                    'type': MessageType.ERROR.value,
                    'data': {
                        'error': f'Message handling error: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    },
                    'timestamp': datetime.now().isoformat(),
                    'client_id': client_id
                }
                
                try:
                    await websocket.send_text(json.dumps(error_message))
                except:
                    break  # Connection is probably closed
                
    except Exception as e:
        logger.error(f"Error in FastAPI WebSocket handler for client {client_id}: {str(e)}")
    finally:
        # Clean up client
        await cleanup_fastapi_client(client_id, service)

async def handle_fastapi_client_message(
    client_id: str, 
    message: str, 
    service: WebSocketDataStreamer,
    websocket: WebSocket
):
    """
    Handle incoming FastAPI WebSocket client message.
    
    Args:
        client_id: Client identifier
        message: Raw message string
        service: WebSocket service
        websocket: FastAPI WebSocket connection
    """
    try:
        # Parse message
        data = json.loads(message)
        message_type = data.get('type')
        message_data = data.get('data', {})
        
        client = service.clients.get(client_id)
        if not client:
            return
        
        # Update heartbeat
        client.last_heartbeat = datetime.now()
        
        # Handle different message types
        if message_type == MessageType.SUBSCRIBE.value:
            await handle_fastapi_subscribe(client_id, message_data, service, websocket)
        
        elif message_type == MessageType.UNSUBSCRIBE.value:
            await handle_fastapi_unsubscribe(client_id, message_data, service, websocket)
        
        elif message_type == MessageType.HEARTBEAT.value:
            await handle_fastapi_heartbeat(client_id, service, websocket)
        
        else:
            # Send error for unknown message type
            error_message = {
                'type': MessageType.ERROR.value,
                'data': {
                    'error': f'Unknown message type: {message_type}',
                    'received_message': data
                },
                'timestamp': datetime.now().isoformat(),
                'client_id': client_id
            }
            
            await websocket.send_text(json.dumps(error_message))
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from FastAPI client {client_id}: {str(e)}")
        await send_fastapi_error(client_id, f"Invalid JSON: {str(e)}", websocket)
    
    except Exception as e:
        logger.error(f"Error handling message from FastAPI client {client_id}: {str(e)}")
        await send_fastapi_error(client_id, f"Message handling error: {str(e)}", websocket)

async def handle_fastapi_subscribe(
    client_id: str, 
    data: Dict[str, Any], 
    service: WebSocketDataStreamer,
    websocket: WebSocket
):
    """Handle FastAPI WebSocket subscription request."""
    try:
        client = service.clients.get(client_id)
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
            await send_fastapi_error(client_id, "No valid tickers provided", websocket)
            return
        
        # Validate subscription types
        valid_types = []
        for sub_type in subscription_types:
            if sub_type in ['price', 'prediction', 'technical', 'news']:
                valid_types.append(sub_type)
        
        if not valid_types:
            await send_fastapi_error(client_id, "No valid subscription types provided", websocket)
            return
        
        # Update client subscription
        client.subscribed_tickers.update(valid_tickers)
        client.subscription_types.update(valid_types)
        client.prediction_interval = max(10, min(300, prediction_interval))
        
        # Update active tickers
        service.active_tickers.update(valid_tickers)
        
        # Send confirmation
        confirmation_message = {
            'type': MessageType.STATUS.value,
            'data': {
                'status': 'subscribed',
                'tickers': list(client.subscribed_tickers),
                'types': list(client.subscription_types),
                'prediction_interval': client.prediction_interval
            },
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id
        }
        
        await websocket.send_text(json.dumps(confirmation_message))
        
        # Send initial data if available
        for ticker in valid_tickers:
            if ticker in service.price_cache:
                await send_fastapi_price_update(client_id, ticker, service.price_cache[ticker], websocket)
            
            if ticker in service.prediction_cache and 'prediction' in valid_types:
                await send_fastapi_prediction_update(client_id, ticker, service.prediction_cache[ticker], websocket)
        
        logger.info(f"FastAPI client {client_id} subscribed to {valid_tickers} for {valid_types}")
        
    except Exception as e:
        logger.error(f"Error handling FastAPI subscription: {str(e)}")
        await send_fastapi_error(client_id, f"Subscription error: {str(e)}", websocket)

async def handle_fastapi_unsubscribe(
    client_id: str, 
    data: Dict[str, Any], 
    service: WebSocketDataStreamer,
    websocket: WebSocket
):
    """Handle FastAPI WebSocket unsubscription request."""
    try:
        client = service.clients.get(client_id)
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
        service._update_active_tickers()
        
        # Send confirmation
        confirmation_message = {
            'type': MessageType.STATUS.value,
            'data': {
                'status': 'unsubscribed',
                'remaining_tickers': list(client.subscribed_tickers),
                'remaining_types': list(client.subscription_types)
            },
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id
        }
        
        await websocket.send_text(json.dumps(confirmation_message))
        
        logger.info(f"FastAPI client {client_id} unsubscribed from {tickers}")
        
    except Exception as e:
        logger.error(f"Error handling FastAPI unsubscription: {str(e)}")
        await send_fastapi_error(client_id, f"Unsubscription error: {str(e)}", websocket)

async def handle_fastapi_heartbeat(
    client_id: str, 
    service: WebSocketDataStreamer,
    websocket: WebSocket
):
    """Handle FastAPI WebSocket heartbeat message."""
    try:
        client = service.clients.get(client_id)
        if not client:
            return
        
        client.last_heartbeat = datetime.now()
        
        # Send heartbeat response
        heartbeat_message = {
            'type': MessageType.HEARTBEAT.value,
            'data': {
                'server_time': datetime.now().isoformat(),
                'client_id': client_id
            },
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id
        }
        
        await websocket.send_text(json.dumps(heartbeat_message))
        
    except Exception as e:
        logger.error(f"Error handling FastAPI heartbeat: {str(e)}")

async def send_fastapi_error(client_id: str, error_message: str, websocket: WebSocket):
    """Send error message to FastAPI WebSocket client."""
    try:
        error_msg = {
            'type': MessageType.ERROR.value,
            'data': {
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            },
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id
        }
        
        await websocket.send_text(json.dumps(error_msg))
        
    except Exception as e:
        logger.error(f"Error sending FastAPI error message: {str(e)}")

async def send_fastapi_price_update(client_id: str, ticker: str, price_data: Dict[str, Any], websocket: WebSocket):
    """Send price update to FastAPI WebSocket client."""
    try:
        price_message = {
            'type': MessageType.PRICE_UPDATE.value,
            'data': {
                'ticker': ticker,
                'price_data': price_data
            },
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id
        }
        
        await websocket.send_text(json.dumps(price_message))
        
    except Exception as e:
        logger.error(f"Error sending FastAPI price update: {str(e)}")

async def send_fastapi_prediction_update(client_id: str, ticker: str, prediction_data: Dict[str, Any], websocket: WebSocket):
    """Send prediction update to FastAPI WebSocket client."""
    try:
        prediction_message = {
            'type': MessageType.PREDICTION_UPDATE.value,
            'data': {
                'ticker': ticker,
                'prediction_data': prediction_data
            },
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id
        }
        
        await websocket.send_text(json.dumps(prediction_message))
        
    except Exception as e:
        logger.error(f"Error sending FastAPI prediction update: {str(e)}")

async def cleanup_fastapi_client(client_id: str, service: WebSocketDataStreamer):
    """Clean up FastAPI WebSocket client connection."""
    try:
        if client_id in service.clients:
            # Remove client
            del service.clients[client_id]
            
            # Update active tickers
            service._update_active_tickers()
            
            logger.info(f"FastAPI client {client_id} cleaned up")
            
    except Exception as e:
        logger.error(f"Error cleaning up FastAPI client {client_id}: {str(e)}")

# Background task to broadcast updates to FastAPI WebSocket clients
async def broadcast_to_fastapi_clients(service: WebSocketDataStreamer):
    """Background task to broadcast updates to FastAPI WebSocket clients."""
    while service.streaming_active:
        try:
            # This would be called by the streaming threads to broadcast updates
            # For now, it's a placeholder for the broadcasting logic
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in FastAPI broadcast task: {str(e)}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "websocket_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )