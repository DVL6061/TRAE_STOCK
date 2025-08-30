from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import asyncio
import json
from datetime import datetime
from app.models.market import MarketData, PredictionData
from app.services.market_service import MarketService
from app.services.prediction_service import PredictionService

# Import API routers
from backend.api import news, predictions, stock_data, training
from backend.api.websocket_api import websocket_endpoint
from backend.core.training_scheduler import start_training_scheduler
from backend.core.error_handler import (
    StockPredictionError,
    stock_prediction_exception_handler,
    general_exception_handler,
    log_info
)

app = FastAPI(title="Stock Prediction API")

# Add exception handlers
app.add_exception_handler(StockPredictionError, stock_prediction_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Startup event to initialize training scheduler
@app.on_event("startup")
async def startup_event():
    """Initialize training scheduler on application startup"""
    log_info("Starting Stock Prediction API application")
    try:
        await start_training_scheduler()
        log_info("Training scheduler started successfully")
    except Exception as e:
        log_info(f"Failed to start training scheduler: {e}", {"error": str(e)})

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(news.router, prefix="/api/news", tags=["news"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(stock_data.router, prefix="/api/stock", tags=["stock_data"])
app.include_router(training.router, prefix="/api/training", tags=["training"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                self.disconnect(connection)

manager = ConnectionManager()
market_service = MarketService()
prediction_service = PredictionService()

# WebSocket endpoint for real-time market data
@app.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Get real-time market data
            market_data = await market_service.get_real_time_data()
            await manager.broadcast({
                "type": "market_update",
                "payload": market_data.dict()
            })
            
            # Get real-time predictions
            predictions = await prediction_service.get_latest_predictions()
            await manager.broadcast({
                "type": "prediction_update",
                "payload": predictions.dict()
            })
            
            await asyncio.sleep(1)  # Update every second
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# REST endpoints
@app.get("/api/market/current")
async def get_current_market_data():
    return await market_service.get_current_data()

@app.get("/api/portfolio/summary")
async def get_portfolio_summary():
    return await market_service.get_portfolio_summary()

@app.get("/api/predictions/latest")
async def get_latest_predictions():
    return await prediction_service.get_latest_predictions()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)